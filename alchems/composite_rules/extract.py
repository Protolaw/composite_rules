from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __package__ in (None, "") and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alchems.io import (
    read_json,
    resolve_existing_path,
    setup_runtime_cache_dirs,
    write_composite_errors as write_errors,
    write_composite_rules,
    write_composite_summary as write_summary,
)


@dataclass(frozen=True)
class MoleculeCenterProjection:
    molecule: Any
    center_atoms: frozenset[int]


@dataclass(frozen=True)
class ReactionRuleStep:
    """A route reaction annotated with its extracted rule and reaction center."""

    rule_smarts: str
    center_atoms: frozenset[int]
    reaction_smiles: str
    target_smiles: str = ""
    reactant_center_molecules: tuple[MoleculeCenterProjection, ...] = ()
    product_center_molecules: tuple[MoleculeCenterProjection, ...] = ()


@dataclass
class RouteProcessingStats:
    routes_seen: int = 0
    routes_with_composites: int = 0
    reactions_seen: int = 0
    reaction_rule_cache_hits: int = 0
    reaction_rule_cache_misses: int = 0
    skipped_reactions: int = 0
    errors: int = 0


class RuleExtractionError(Exception):
    """Raised when a reaction cannot produce exactly one usable rule."""


def reaction_smiles_from_node(node: dict[str, Any]) -> str:
    metadata = node.get("metadata") or {}
    smiles = (
        node.get("smiles")
        or metadata.get("smiles")
        or metadata.get("mapped_reaction_smiles")
        or metadata.get("rsmi")
    )
    if not smiles:
        raise ValueError("reaction node has no mapped reaction SMILES")
    return smiles


@lru_cache(maxsize=8192)
def parse_route_reaction(reaction_smiles: str) -> Any:
    from chython import smiles as parse_smiles

    return parse_smiles(reaction_smiles)


def route_reaction_atom_ids(reaction: Any) -> set[int]:
    atom_ids: set[int] = set()
    for molecule in reaction.reactants + reaction.products + reaction.reagents:
        atom_ids.update(int(atom_id) for atom_id in molecule)
    return atom_ids


def same_route_molecule(left: Any, right: Any) -> bool:
    if left.atoms_count != right.atoms_count or left.bonds_count != right.bonds_count:
        return False
    try:
        if any(True for _mapping in left.get_mapping(right)):
            return True
    except Exception:
        pass

    left_copy = left.copy()
    right_copy = right.copy()
    left_copy.canonicalize()
    right_copy.canonicalize()
    if left_copy == right_copy:
        return True

    left_copy = left.copy()
    right_copy = right.copy()
    try:
        left_copy.clean_stereo()
        right_copy.clean_stereo()
    except Exception:
        return False
    left_copy.canonicalize()
    right_copy.canonicalize()
    return left_copy == right_copy


def remap_route_molecule(molecule: Any, mapping: dict[int, int]) -> Any:
    molecule_copy = molecule.copy()
    molecule_mapping = {
        atom_id: mapping[atom_id]
        for atom_id in molecule_copy
        if atom_id in mapping and atom_id != mapping[atom_id]
    }
    if molecule_mapping:
        molecule_copy.remap(molecule_mapping)
    return molecule_copy


def remap_route_reaction(reaction: Any, mapping: dict[int, int]) -> Any:
    from chython.containers import ReactionContainer

    return ReactionContainer(
        tuple(remap_route_molecule(molecule, mapping) for molecule in reaction.reactants),
        tuple(remap_route_molecule(molecule, mapping) for molecule in reaction.products),
        tuple(remap_route_molecule(molecule, mapping) for molecule in reaction.reagents),
        meta=dict(getattr(reaction, "meta", {}) or {}),
        name=getattr(reaction, "name", None),
    )


def molecule_node_smiles(node: dict[str, Any]) -> str:
    metadata = node.get("metadata") or {}
    return node.get("smiles") or metadata.get("smiles") or ""


def set_molecule_node_mapped_smiles(node: dict[str, Any], molecule: Any) -> None:
    metadata = node.setdefault("metadata", {})
    metadata.setdefault("original_smiles", molecule_node_smiles(node))
    metadata["mapped_smiles"] = format(molecule, "m")
    node["smiles"] = metadata["mapped_smiles"]


def find_route_side_molecule(
    candidates: Iterable[Any],
    reference: Any | None,
    *,
    fallback_smiles: str = "",
    excluded_indexes: set[int] | None = None,
) -> tuple[int, Any] | tuple[None, None]:
    excluded = excluded_indexes or set()
    reference_molecule = reference
    if reference_molecule is None and fallback_smiles:
        try:
            reference_molecule = parse_route_molecule(fallback_smiles)
        except Exception:
            reference_molecule = None
    if reference_molecule is None:
        return None, None

    for index, candidate in enumerate(candidates):
        if index in excluded:
            continue
        try:
            if same_route_molecule(candidate, reference_molecule):
                return index, candidate
        except Exception:
            continue
    return None, None


def normalize_route_tree(route: dict[str, Any]) -> dict[str, Any]:
    """Return a route copy with globally consistent atom maps.

    PaRoutes stores each reaction with step-local atom maps. This normalizer
    walks the route from the target toward stock molecules, aligns every child
    reaction product to the mapped molecule expected by its parent reaction, and
    assigns fresh map numbers to atoms that are newly introduced in each branch.
    The original molecule ``smiles`` fields are preserved for display; the
    globally mapped molecule representation is stored in ``metadata.mapped_smiles``.
    """

    route = copy.deepcopy(route)
    all_original_atom_ids: set[int] = set()

    def collect_original_atom_ids(node: dict[str, Any]) -> None:
        if node.get("type") == "reaction":
            node["smiles"] = reaction_smiles_from_node(node)
            try:
                all_original_atom_ids.update(
                    route_reaction_atom_ids(parse_route_reaction(node["smiles"]))
                )
            except Exception:
                pass
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                collect_original_atom_ids(child)

    collect_original_atom_ids(route)
    used_atom_ids: set[int] = set()
    next_atom_id = max(all_original_atom_ids or {0}) + 1

    def fresh_atom_id() -> int:
        nonlocal next_atom_id
        while next_atom_id in used_atom_ids:
            next_atom_id += 1
        atom_id = next_atom_id
        used_atom_ids.add(atom_id)
        next_atom_id += 1
        return atom_id

    def complete_mapping(reaction: Any, alignment: dict[int, int]) -> dict[int, int]:
        mapping: dict[int, int] = dict(alignment)
        for target_atom_id in alignment.values():
            used_atom_ids.add(int(target_atom_id))

        for atom_id in sorted(route_reaction_atom_ids(reaction)):
            if atom_id in mapping:
                continue
            if atom_id in used_atom_ids:
                mapping[atom_id] = fresh_atom_id()
            else:
                mapping[atom_id] = atom_id
                used_atom_ids.add(atom_id)
        return mapping

    def visit_molecule(node: dict[str, Any], expected_molecule: Any | None = None) -> None:
        if expected_molecule is not None:
            set_molecule_node_mapped_smiles(node, expected_molecule)

        for child in node.get("children", []) or []:
            if not isinstance(child, dict) or child.get("type") != "reaction":
                continue
            try:
                reaction = parse_route_reaction(reaction_smiles_from_node(child))
            except Exception:
                visit_reaction_children(child)
                continue

            product_index, product = find_route_side_molecule(
                reaction.products,
                expected_molecule,
                fallback_smiles=molecule_node_smiles(node),
            )
            alignment: dict[int, int] = {}
            if product is not None:
                if expected_molecule is None:
                    alignment = {int(atom_id): int(atom_id) for atom_id in product}
                else:
                    mappings = list(product.get_mapping(expected_molecule))
                    if mappings:
                        alignment = {
                            int(source): int(target)
                            for source, target in mappings[0].items()
                        }
                mapping = complete_mapping(reaction, alignment)
                normalized_reaction = remap_route_reaction(reaction, mapping)
                child["smiles"] = format(normalized_reaction, "m")
                if product_index is not None:
                    normalized_products = list(normalized_reaction.products)
                    if product_index < len(normalized_products):
                        set_molecule_node_mapped_smiles(
                            node,
                            normalized_products[product_index],
                        )
                visit_reaction_children(child, normalized_reaction)
            else:
                mapping = complete_mapping(reaction, {})
                normalized_reaction = remap_route_reaction(reaction, mapping)
                child["smiles"] = format(normalized_reaction, "m")
                visit_reaction_children(child, normalized_reaction)

    def visit_reaction_children(
        reaction_node: dict[str, Any],
        reaction: Any | None = None,
    ) -> None:
        if reaction is None:
            try:
                reaction = parse_route_reaction(reaction_smiles_from_node(reaction_node))
            except Exception:
                for child in reaction_node.get("children", []) or []:
                    if isinstance(child, dict) and child.get("type") == "mol":
                        visit_molecule(child)
                return

        used_reactant_indexes: set[int] = set()
        reactants = list(reaction.reactants)
        for child in reaction_node.get("children", []) or []:
            if not isinstance(child, dict) or child.get("type") != "mol":
                continue
            reactant_index, reactant = find_route_side_molecule(
                reactants,
                None,
                fallback_smiles=molecule_node_smiles(child),
                excluded_indexes=used_reactant_indexes,
            )
            if reactant_index is not None:
                used_reactant_indexes.add(reactant_index)
            visit_molecule(child, reactant)

    visit_molecule(route)
    return route


def route_items(routes_json: Any) -> Iterable[tuple[Any, dict[str, Any]]]:
    if isinstance(routes_json, list):
        for route_id, route in enumerate(routes_json):
            yield route_id, route
        return

    if isinstance(routes_json, dict):
        for route_id, route in routes_json.items():
            yield route_id, route
        return

    raise TypeError(f"unsupported routes JSON root: {type(routes_json)!r}")


def child_reaction_nodes(reaction_node: dict[str, Any]) -> list[dict[str, Any]]:
    children = []
    for mol_node in reaction_node.get("children", []) or []:
        if mol_node.get("type") != "mol":
            continue
        for child in mol_node.get("children", []) or []:
            if child.get("type") == "reaction":
                children.append(child)
    return children


def root_reaction_nodes(route: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        child
        for child in route.get("children", []) or []
        if isinstance(child, dict) and child.get("type") == "reaction"
    ]


def route_target_smiles(route: dict[str, Any]) -> str:
    metadata = route.get("metadata") or {}
    return metadata.get("original_smiles") or route.get("smiles") or metadata.get("smiles") or ""


def reaction_paths_from_node(
    reaction_node: dict[str, Any],
    step_by_reaction_smiles: dict[str, ReactionRuleStep],
) -> list[list[ReactionRuleStep]]:
    step = step_by_reaction_smiles[reaction_smiles_from_node(reaction_node)]
    children = child_reaction_nodes(reaction_node)
    if not children:
        return [[step]]

    paths: list[list[ReactionRuleStep]] = []
    for child in children:
        for suffix in reaction_paths_from_node(child, step_by_reaction_smiles):
            paths.append([step] + suffix)
    return paths


@lru_cache(maxsize=8192)
def parse_route_molecule(molecule_smiles: str) -> Any:
    from chython import smiles as parse_smiles

    return parse_smiles(molecule_smiles)


def side_center_molecules(
    molecules: Iterable[Any],
    center_atoms: frozenset[int],
) -> tuple[MoleculeCenterProjection, ...]:
    projections = []
    for molecule in molecules:
        molecule_center_atoms = center_atoms & set(molecule)
        projections.append(
            MoleculeCenterProjection(
                molecule=molecule,
                center_atoms=frozenset(molecule_center_atoms),
            )
        )
    return tuple(projections)


def project_side_centers_to_route_molecule(
    side_molecules: tuple[MoleculeCenterProjection, ...],
    route_molecule_smiles: str,
) -> tuple[frozenset[int], bool]:
    route_molecule = parse_route_molecule(route_molecule_smiles)
    projected_center_atoms: set[int] = set()
    matched_route_molecule = False

    for side_molecule in side_molecules:
        if side_molecule.molecule.atoms_count != route_molecule.atoms_count:
            continue
        for mapping in side_molecule.molecule.get_mapping(route_molecule):
            matched_route_molecule = True
            projected_center_atoms.update(
                mapping[atom_id]
                for atom_id in side_molecule.center_atoms
                if atom_id in mapping
            )

    return frozenset(projected_center_atoms), matched_route_molecule


def projected_center_atoms_touch(
    route_molecule_smiles: str,
    left_centers: frozenset[int],
    right_centers: frozenset[int],
) -> bool:
    if left_centers & right_centers:
        return True

    route_molecule = parse_route_molecule(route_molecule_smiles)
    for atom_1, atom_2, _bond in route_molecule.bonds():
        if atom_1 in left_centers and atom_2 in right_centers and center_contact_allowed(
            route_molecule,
            atom_1,
            atom_2,
        ):
            return True
        if atom_2 in left_centers and atom_1 in right_centers and center_contact_allowed(
            route_molecule,
            atom_2,
            atom_1,
        ):
            return True
    return False


def projected_center_components(
    route_molecule_smiles: str,
    center_atoms: frozenset[int],
) -> list[frozenset[int]]:
    route_molecule = parse_route_molecule(route_molecule_smiles)
    remaining = set(center_atoms)
    components: list[frozenset[int]] = []
    adjacency: dict[int, set[int]] = {atom: set() for atom in center_atoms}
    for atom_1, atom_2, _bond in route_molecule.bonds():
        if atom_1 in center_atoms and atom_2 in center_atoms:
            adjacency[atom_1].add(atom_2)
            adjacency[atom_2].add(atom_1)

    while remaining:
        stack = [remaining.pop()]
        component = set(stack)
        while stack:
            atom = stack.pop()
            for neighbor in adjacency[atom]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(frozenset(component))
    return components


def touches_all_center_components(
    route_molecule_smiles: str,
    left_centers: frozenset[int],
    right_centers: frozenset[int],
) -> bool:
    components = projected_center_components(route_molecule_smiles, right_centers)
    if len(components) <= 1:
        return True
    return all(
        projected_center_atoms_touch(route_molecule_smiles, left_centers, component)
        for component in components
    )


def center_contact_allowed(route_molecule: Any, atom_1: int, atom_2: int) -> bool:
    atomic_numbers = {
        atom_number: atom.atomic_number for atom_number, atom in route_molecule.atoms()
    }
    atom_1_number = atomic_numbers.get(atom_1)
    atom_2_number = atomic_numbers.get(atom_2)
    if atom_1_number is None or atom_2_number is None:
        return False

    if atom_1_number != 6 or atom_2_number != 6:
        return True

    return is_carbonyl_carbon(route_molecule, atom_1) or is_carbonyl_carbon(
        route_molecule,
        atom_2,
    )


def is_carbonyl_carbon(route_molecule: Any, atom_number: int) -> bool:
    atom = route_molecule.atom(atom_number)
    if atom.atomic_number != 6:
        return False
    for neighbor_id, bond in route_molecule._bonds[atom_number].items():
        neighbor = route_molecule.atom(neighbor_id)
        if neighbor.atomic_number == 8 and int(bond) == 2:
            return True
    return False


def adjacent_centers_overlap(left: ReactionRuleStep, right: ReactionRuleStep) -> bool:
    if (
        right.target_smiles
        and left.reactant_center_molecules
        and right.product_center_molecules
    ):
        left_centers, left_matched = project_side_centers_to_route_molecule(
            left.reactant_center_molecules,
            right.target_smiles,
        )
        right_centers, right_matched = project_side_centers_to_route_molecule(
            right.product_center_molecules,
            right.target_smiles,
        )
        if not (
            left_matched
            and right_matched
            and touches_all_center_components(
                right.target_smiles,
                left_centers,
                right_centers,
            )
            and projected_center_atoms_touch(
                right.target_smiles,
                left_centers,
                right_centers,
            )
        ):
            return False
        return not is_excluded_adjacent_pair(left, right)

    return bool(left.center_atoms & right.center_atoms)


def is_excluded_adjacent_pair(
    left: ReactionRuleStep,
    right: ReactionRuleStep,
) -> bool:
    return (
        is_sulfonyl_ester_activation_rule(left.rule_smarts)
        and is_alcohol_ester_deprotection_rule(right.rule_smarts)
    )


def is_sulfonyl_ester_activation_rule(rule_smarts: str) -> bool:
    left, _, right = rule_smarts.partition(">>")
    return (
        "-[O;D2" in left
        and "-[S;D4" in left
        and "=[O;D1" in left
        and "[O;D1" in right
    )


def is_alcohol_ester_deprotection_rule(rule_smarts: str) -> bool:
    left, _, right = rule_smarts.partition(">>")
    return (
        "-[O;D1" in left
        and "-[O;D2" in right
        and "=[O;D1" in right
    )


def valid_composite_sequences(
    path: list[ReactionRuleStep],
    *,
    min_length: int,
    max_length: int | None,
) -> Iterable[tuple[str, ...]]:
    for sequence, _target_smiles in valid_composite_sequence_occurrences(
        path,
        min_length=min_length,
        max_length=max_length,
    ):
        yield sequence


def valid_composite_sequence_occurrences(
    path: list[ReactionRuleStep],
    *,
    min_length: int,
    max_length: int | None,
) -> Iterable[tuple[tuple[str, ...], str]]:
    if len(path) < min_length:
        return

    segment: list[ReactionRuleStep] = [path[0]]

    def emit_segment(
        steps: list[ReactionRuleStep],
    ) -> Iterable[tuple[tuple[str, ...], str]]:
        if len(steps) < min_length:
            return
        upper = len(steps) if max_length is None else min(len(steps), max_length)
        for start in range(len(steps)):
            for end in range(start + min_length, min(len(steps), start + upper) + 1):
                yield (
                    tuple(step.rule_smarts for step in steps[start:end]),
                    steps[start].target_smiles,
                )

    for step in path[1:]:
        if adjacent_centers_overlap(segment[-1], step):
            segment.append(step)
            continue
        yield from emit_segment(segment)
        segment = [step]

    yield from emit_segment(segment)


class SynPlannerRuleExtractor:
    def __init__(self, config: Any):
        from synplan.chem.data.standardizing import RemoveReagentsStandardizer

        self.config = config
        self.standardizer = RemoveReagentsStandardizer()
        self.cache: dict[str, ReactionRuleStep | None] = {}

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SynPlannerRuleExtractor":
        from synplan.utils.config import RuleExtractionConfig

        if args.config:
            config = RuleExtractionConfig.from_yaml(str(resolve_existing_path(args.config)))
        else:
            config = RuleExtractionConfig(
                min_popularity=1,
                single_product_only=True,
                environment_atom_count=args.environment_atom_count,
                multicenter_rules=True,
                include_rings=args.include_rings,
                include_func_groups=False,
                keep_leaving_groups=args.keep_leaving_groups,
                keep_incoming_groups=args.keep_incoming_groups,
                keep_reagents=False,
                reactor_validation=args.reactor_validation,
            )
        return cls(config)

    def extract(self, reaction_smiles: str) -> tuple[ReactionRuleStep | None, bool]:
        """Return `(step, cache_hit)` for one mapped reaction SMILES."""

        if reaction_smiles in self.cache:
            return self.cache[reaction_smiles], True

        from chython import smiles as parse_smiles
        from synplan.chem.reaction_rules.extraction import (
            _rule_to_reactor_smarts,
            extract_rules,
        )

        reaction = parse_smiles(reaction_smiles)
        standardized = self.standardizer(reaction)
        center_atoms = frozenset((~standardized).center_atoms)
        reactant_center_molecules = side_center_molecules(
            standardized.reactants,
            center_atoms,
        )
        product_center_molecules = side_center_molecules(
            standardized.products,
            center_atoms,
        )
        rules, skipped = extract_rules(self.config, reaction)
        if skipped or not rules:
            self.cache[reaction_smiles] = None
            return None, False
        if len(rules) != 1:
            raise RuleExtractionError(
                "composite extraction expects one multicenter rule per reaction; "
                f"got {len(rules)} rules"
            )

        rule_smarts = _rule_to_reactor_smarts(rules[0])
        step = ReactionRuleStep(
            rule_smarts=rule_smarts,
            center_atoms=center_atoms,
            reaction_smiles=reaction_smiles,
            reactant_center_molecules=reactant_center_molecules,
            product_center_molecules=product_center_molecules,
        )
        self.cache[reaction_smiles] = step
        return step, False


def collect_reaction_contexts(route: dict[str, Any]) -> list[tuple[str, str]]:
    contexts: list[tuple[str, str]] = []

    def visit(node: dict[str, Any]) -> None:
        if node.get("type") == "mol":
            target_smiles = node.get("smiles", "")
            for child in node.get("children", []) or []:
                if isinstance(child, dict) and child.get("type") == "reaction":
                    contexts.append((reaction_smiles_from_node(child), target_smiles))
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                visit(child)

    visit(route)
    return contexts


def collect_reaction_smiles(route: dict[str, Any]) -> list[str]:
    return [reaction_smiles for reaction_smiles, _ in collect_reaction_contexts(route)]


def extract_route_composites(
    route: dict[str, Any],
    rule_extractor: SynPlannerRuleExtractor,
    *,
    min_length: int,
    max_length: int | None,
    stats: RouteProcessingStats,
) -> dict[tuple[str, ...], set[str]]:
    route = normalize_route_tree(route)
    step_by_reaction_smiles: dict[str, ReactionRuleStep] = {}

    for reaction_smiles, target_smiles in collect_reaction_contexts(route):
        stats.reactions_seen += 1
        step, cache_hit = rule_extractor.extract(reaction_smiles)
        if cache_hit:
            stats.reaction_rule_cache_hits += 1
        else:
            stats.reaction_rule_cache_misses += 1
        if step is None:
            stats.skipped_reactions += 1
            continue
        step_by_reaction_smiles[reaction_smiles] = ReactionRuleStep(
            rule_smarts=step.rule_smarts,
            center_atoms=step.center_atoms,
            reaction_smiles=step.reaction_smiles,
            target_smiles=target_smiles,
            reactant_center_molecules=step.reactant_center_molecules,
            product_center_molecules=step.product_center_molecules,
        )

    sequences: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for root in root_reaction_nodes(route):
        try:
            paths = reaction_paths_from_node(root, step_by_reaction_smiles)
        except KeyError:
            continue
        for path in paths:
            for sequence, target_smiles in valid_composite_sequence_occurrences(
                path,
                min_length=min_length,
                max_length=max_length,
            ):
                sequences[sequence].add(target_smiles)
    return sequences


def run(args: argparse.Namespace) -> int:
    setup_runtime_cache_dirs()
    if args.min_length < 2:
        raise ValueError("--min-length must be at least 2")
    if args.max_length is not None and args.max_length <= 0:
        args.max_length = None
    if args.max_length is not None and args.max_length < args.min_length:
        raise ValueError("--max-length must be greater than or equal to --min-length")

    rule_extractor = SynPlannerRuleExtractor.from_args(args)

    routes_json = read_json(args.routes_json)

    references_by_sequence: dict[tuple[str, ...], set[Any]] = defaultdict(set)
    target_molecules_by_sequence: dict[tuple[str, ...], dict[Any, set[str]]] = (
        defaultdict(lambda: defaultdict(set))
    )
    errors: list[dict[str, Any]] = []
    stats = RouteProcessingStats()

    for index, (route_id, route) in enumerate(route_items(routes_json), start=1):
        if args.limit is not None and index > args.limit:
            break
        stats.routes_seen += 1
        try:
            route_sequences = extract_route_composites(
                route,
                rule_extractor,
                min_length=args.min_length,
                max_length=args.max_length,
                stats=stats,
            )
            if route_sequences:
                stats.routes_with_composites += 1
            for sequence, target_molecules in route_sequences.items():
                references_by_sequence[sequence].add(route_id)
                target_molecules_by_sequence[sequence][route_id].update(
                    target_molecules
                )
        except Exception as exc:
            stats.errors += 1
            if not args.ignore_errors:
                raise
            errors.append(
                {
                    "route_id": route_id,
                    "stage": "extract_route_composites",
                    "error_type": type(exc).__qualname__,
                    "message": str(exc) or traceback.format_exc(limit=1).strip(),
                }
            )

        if args.progress_interval and index % args.progress_interval == 0:
            print(
                f"processed routes={index} composite_rules={len(references_by_sequence)} "
                f"errors={stats.errors}",
                flush=True,
            )

    output_summary = write_composite_rules(
        args.output,
        references_by_sequence,
        target_molecules_by_sequence=target_molecules_by_sequence,
    )
    write_errors(args.output, errors)

    summary = {
        "routes_json": str(args.routes_json),
        "routes_seen": stats.routes_seen,
        "routes_with_composite_rules": stats.routes_with_composites,
        "reactions_seen": stats.reactions_seen,
        "reaction_rule_cache_hits": stats.reaction_rule_cache_hits,
        "reaction_rule_cache_misses": stats.reaction_rule_cache_misses,
        "skipped_reactions": stats.skipped_reactions,
        "errors": stats.errors,
        "unique_composite_rules": len(references_by_sequence),
        "target_molecule_occurrences": sum(
            len(targets)
            for route_targets in target_molecules_by_sequence.values()
            for targets in route_targets.values()
        ),
        "min_length": args.min_length,
        "max_length": args.max_length,
        "output_prefix": str(args.output.with_suffix("")),
        **output_summary,
    }
    summary_path = write_summary(args.output, summary)
    summary["summary_file"] = str(summary_path)
    write_summary(args.output, summary)

    print(json.dumps(summary, indent=2), flush=True)
    return 0
