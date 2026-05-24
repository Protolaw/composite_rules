from __future__ import annotations

import argparse
import copy
import json
import sys
import traceback
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __package__ in (None, "") and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from route_inspector.io import (
    dataset_prefix_from_path,
    normalize_n_cpu,
    open_text,
    resolve_existing_path,
    resolve_output_path,
    setup_runtime_cache_dirs,
    write_standard_sidecars,
    write_composite_errors as write_errors,
    write_composite_routes_without_rules,
    write_composite_rules,
    write_composite_summary as write_summary,
)


_COMPOSITE_WORKER_EXTRACTOR: Any | None = None
_COMPOSITE_WORKER_MIN_LENGTH = 2
_COMPOSITE_WORKER_MAX_LENGTH: int | None = 5
_COMPOSITE_WORKER_STORE_ROUTES_WITHOUT_COMPOSITES = True


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
    """Read reaction SMILES stored on a route node.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
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
    """Parse route reaction into chython/SynPlanner objects.

    It operates on PaRoutes route trees after atom-map normalization, which keeps
    reaction centers comparable across convergent route branches.
    """
    from chython import smiles as parse_smiles

    return parse_smiles(reaction_smiles)


def route_reaction_atom_ids(reaction: Any) -> set[int]:
    """Return route reaction atom IDs from a normalized route tree.

    It operates on PaRoutes route trees after atom-map normalization, which keeps
    reaction centers comparable across convergent route branches.
    """
    atom_ids: set[int] = set()
    for molecule in reaction.reactants + reaction.products + reaction.reagents:
        atom_ids.update(int(atom_id) for atom_id in molecule)
    return atom_ids


def same_route_molecule(left: Any, right: Any) -> bool:
    """Return whether two inputs represent the same route molecule.

    It operates on PaRoutes route trees after atom-map normalization, which keeps
    reaction centers comparable across convergent route branches.
    """
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
    """Remap route molecule with globally consistent atom IDs.

    It operates on PaRoutes route trees after atom-map normalization, which keeps
    reaction centers comparable across convergent route branches.
    """
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
    """Remap route reaction with globally consistent atom IDs.

    It operates on PaRoutes route trees after atom-map normalization, which keeps
    reaction centers comparable across convergent route branches.
    """
    from chython.containers import ReactionContainer

    return ReactionContainer(
        tuple(remap_route_molecule(molecule, mapping) for molecule in reaction.reactants),
        tuple(remap_route_molecule(molecule, mapping) for molecule in reaction.products),
        tuple(remap_route_molecule(molecule, mapping) for molecule in reaction.reagents),
        meta=dict(getattr(reaction, "meta", {}) or {}),
        name=getattr(reaction, "name", None),
    )


def molecule_node_smiles(node: dict[str, Any]) -> str:
    """Return molecule node SMILES from a molecule or molecule node.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    metadata = node.get("metadata") or {}
    return node.get("smiles") or metadata.get("smiles") or ""


def set_molecule_node_mapped_smiles(node: dict[str, Any], molecule: Any) -> None:
    """Set molecule node mapped SMILES on a route-tree node.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
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
    """Find route side molecule if a valid match exists.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
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
        """Collect original atom IDs from route-analysis input data.

        It operates on PaRoutes route trees after atom-map normalization, which keeps
        reaction centers comparable across convergent route branches.
        """
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
        """Return the next unused atom-map number for a normalized route branch.

        It operates on PaRoutes route trees after atom-map normalization, which keeps
        reaction centers comparable across convergent route branches.
        """
        nonlocal next_atom_id
        while next_atom_id in used_atom_ids:
            next_atom_id += 1
        atom_id = next_atom_id
        used_atom_ids.add(atom_id)
        next_atom_id += 1
        return atom_id

    def complete_mapping(reaction: Any, alignment: dict[int, int]) -> dict[int, int]:
        """Complete an atom-map alignment with fresh IDs for newly introduced atoms.

        The helper feeds rule extraction, center-overlap filtering, or final TSV
        aggregation without changing the original route JSON in place.
        """
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
        """Visit one molecule during recursive route traversal.

        The helper feeds rule extraction, center-overlap filtering, or final TSV
        aggregation without changing the original route JSON in place.
        """
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
        """Visit one reaction children during recursive route traversal.

        The helper feeds rule extraction, center-overlap filtering, or final TSV
        aggregation without changing the original route JSON in place.
        """
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
    """Return route items from a normalized route tree.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
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
    """Return reaction children directly below a route reaction node.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    children = []
    for mol_node in reaction_node.get("children", []) or []:
        if mol_node.get("type") != "mol":
            continue
        for child in mol_node.get("children", []) or []:
            if child.get("type") == "reaction":
                children.append(child)
    return children


def root_reaction_nodes(route: dict[str, Any]) -> list[dict[str, Any]]:
    """Return top-level reaction nodes below the target molecule.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    return [
        child
        for child in route.get("children", []) or []
        if isinstance(child, dict) and child.get("type") == "reaction"
    ]


def route_target_smiles(route: dict[str, Any]) -> str:
    """Return route target SMILES from a normalized route tree.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    metadata = route.get("metadata") or {}
    return metadata.get("original_smiles") or route.get("smiles") or metadata.get("smiles") or ""


def reaction_paths_from_node(
    reaction_node: dict[str, Any],
    step_by_reaction_smiles: dict[str, ReactionRuleStep],
) -> list[list[ReactionRuleStep]]:
    """Read reaction paths stored on a route node.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
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
    """Parse route molecule into chython/SynPlanner objects.

    It operates on PaRoutes route trees after atom-map normalization, which keeps
    reaction centers comparable across convergent route branches.
    """
    from chython import smiles as parse_smiles

    return parse_smiles(molecule_smiles)


def side_center_molecules(
    molecules: Iterable[Any],
    center_atoms: frozenset[int],
) -> tuple[MoleculeCenterProjection, ...]:
    """Find reactant or product molecules that contain reaction-center atoms.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
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
    """Project side centers to route molecule onto the route molecule being compared.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
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
    """Project center atoms touch onto the route molecule being compared.

    It operates on PaRoutes route trees after atom-map normalization, which keeps
    reaction centers comparable across convergent route branches.
    """
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
    """Project center components onto the route molecule being compared.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
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
    """Return whether all center components touches every required center.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    components = projected_center_components(route_molecule_smiles, right_centers)
    if len(components) <= 1:
        return True
    return all(
        projected_center_atoms_touch(route_molecule_smiles, left_centers, component)
        for component in components
    )


def center_contact_allowed(route_molecule: Any, atom_1: int, atom_2: int) -> bool:
    """Return whether two projected center atoms can link a composite sequence.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
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
    """Return whether carbonyl carbon matches the expected condition.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    atom = route_molecule.atom(atom_number)
    if atom.atomic_number != 6:
        return False
    for neighbor_id, bond in route_molecule._bonds[atom_number].items():
        neighbor = route_molecule.atom(neighbor_id)
        if neighbor.atomic_number == 8 and int(bond) == 2:
            return True
    return False


def adjacent_centers_overlap(left: ReactionRuleStep, right: ReactionRuleStep) -> bool:
    """Return whether adjacent route steps share a compatible reaction center.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
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
    """Return whether excluded adjacent pair matches the expected condition.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    return (
        is_sulfonyl_ester_activation_rule(left.rule_smarts)
        and is_alcohol_ester_deprotection_rule(right.rule_smarts)
    )


def is_sulfonyl_ester_activation_rule(rule_smarts: str) -> bool:
    """Return whether sulfonyl ester activation rule matches the expected condition.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    left, _, right = rule_smarts.partition(">>")
    return (
        "-[O;D2" in left
        and "-[S;D4" in left
        and "=[O;D1" in left
        and "[O;D1" in right
    )


def is_alcohol_ester_deprotection_rule(rule_smarts: str) -> bool:
    """Return whether alcohol ester deprotection rule matches the expected condition.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
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
    """Return ordered rule sequences that satisfy the composite-rule definition.

    The result preserves reaction order and the shared-center constraint used to decide
    whether adjacent route steps form a valid composite rule.
    """
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
    """Return valid composite-rule occurrences with their source step indices.

    The result preserves reaction order and the shared-center constraint used to decide
    whether adjacent route steps form a valid composite rule.
    """
    if len(path) < min_length:
        return

    segment: list[ReactionRuleStep] = [path[0]]

    def emit_segment(
        steps: list[ReactionRuleStep],
    ) -> Iterable[tuple[tuple[str, ...], str]]:
        """Emit one contiguous ordered segment from a candidate route path.

        The helper feeds rule extraction, center-overlap filtering, or final TSV
        aggregation without changing the original route JSON in place.
        """
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


@lru_cache(maxsize=32768)
def rule_querycgr_identity(rule_smarts: str) -> str:
    """Return rule QueryCGR identity used for rule extraction or comparison.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    try:
        from route_inspector.alchemical_rules.alchemical import (
            query_cgr_coarse_signature,
            rule_query_cgr,
        )

        signature = query_cgr_coarse_signature(rule_query_cgr(rule_smarts))
        return json.dumps(signature, sort_keys=True, default=repr)
    except Exception:
        return rule_smarts


def composite_sequence_identity(sequence: tuple[str, ...]) -> tuple[str, ...]:
    """Return the structural identity tuple for a composite-rule sequence.

    The result preserves reaction order and the shared-center constraint used to decide
    whether adjacent route steps form a valid composite rule.
    """
    return tuple(rule_querycgr_identity(rule_smarts) for rule_smarts in sequence)


def merge_composite_sequences_by_querycgr(
    references_by_sequence: dict[tuple[str, ...], set[Any]],
    target_molecules_by_sequence: dict[tuple[str, ...], dict[Any, set[str]]],
) -> tuple[
    dict[tuple[str, ...], set[Any]],
    dict[tuple[str, ...], dict[Any, set[str]]],
]:
    """Merge composite sequences by QueryCGR into aggregate results.

    The result preserves reaction order and the shared-center constraint used to decide
    whether adjacent route steps form a valid composite rule.
    """
    representative_by_identity: dict[tuple[str, ...], tuple[str, ...]] = {}
    references_out: dict[tuple[str, ...], set[Any]] = defaultdict(set)
    targets_out: dict[tuple[str, ...], dict[Any, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for sequence in sorted(references_by_sequence, key=lambda item: "$".join(item)):
        identity = composite_sequence_identity(sequence)
        representative = representative_by_identity.setdefault(identity, sequence)
        references_out[representative].update(references_by_sequence[sequence])
        for route_id, target_molecules in target_molecules_by_sequence.get(
            sequence,
            {},
        ).items():
            targets_out[representative][route_id].update(target_molecules)

    return references_out, targets_out


class SynPlannerRuleExtractor:
    def __init__(self, config: Any):
        """Initialize this object with its resolved configuration.

        The helper feeds rule extraction, center-overlap filtering, or final TSV
        aggregation without changing the original route JSON in place.
        """

        self.config = config
        self.cache: dict[str, ReactionRuleStep | None] = {}

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SynPlannerRuleExtractor":
        """Build an extractor instance from parsed CLI arguments.

        The helper feeds rule extraction, center-overlap filtering, or final TSV
        aggregation without changing the original route JSON in place.
        """
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
        """Extract one rule from a mapped reaction SMILES string.

        The helper feeds rule extraction, center-overlap filtering, or final TSV
        aggregation without changing the original route JSON in place.
        """

        if reaction_smiles in self.cache:
            return self.cache[reaction_smiles], True

        from chython import smiles as parse_smiles
        from synplan.chem.reaction_rules.extraction import (
            _rule_to_reactor_smarts,
            extract_rules,
        )

        reaction = parse_smiles(reaction_smiles)
        # standardized = self.standardizer(reaction)
        center_atoms = frozenset((~reaction).center_atoms)
        reactant_center_molecules = side_center_molecules(
            reaction.reactants,
            center_atoms,
        )
        product_center_molecules = side_center_molecules(
            reaction.products,
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


SerializedReactionRuleStep = dict[str, Any]


def serialize_center_projection(
    projection: MoleculeCenterProjection,
) -> tuple[str, tuple[int, ...]]:
    """Serialize center projection for worker-safe transport.

    Serialization keeps rule-step metadata small enough to pass through worker processes
    and stable enough for later TSV aggregation.
    """
    return (
        format(projection.molecule, "m"),
        tuple(sorted(projection.center_atoms)),
    )


def deserialize_center_projection(
    serialized: tuple[str, tuple[int, ...]],
) -> MoleculeCenterProjection:
    """Deserialize center projection from worker output.

    Serialization keeps rule-step metadata small enough to pass through worker processes
    and stable enough for later TSV aggregation.
    """
    molecule_smiles, center_atoms = serialized
    return MoleculeCenterProjection(
        molecule=parse_route_molecule(molecule_smiles),
        center_atoms=frozenset(center_atoms),
    )


def serialize_reaction_rule_step(
    step: ReactionRuleStep,
) -> SerializedReactionRuleStep:
    """Serialize reaction rule step for worker-safe transport.

    Serialization keeps rule-step metadata small enough to pass through worker processes
    and stable enough for later TSV aggregation.
    """
    return {
        "rule_smarts": step.rule_smarts,
        "center_atoms": tuple(sorted(step.center_atoms)),
        "reaction_smiles": step.reaction_smiles,
        "reactant_center_molecules": tuple(
            serialize_center_projection(projection)
            for projection in step.reactant_center_molecules
        ),
        "product_center_molecules": tuple(
            serialize_center_projection(projection)
            for projection in step.product_center_molecules
        ),
    }


def deserialize_reaction_rule_step(
    serialized: SerializedReactionRuleStep,
) -> ReactionRuleStep:
    """Deserialize reaction rule step from worker output.

    Serialization keeps rule-step metadata small enough to pass through worker processes
    and stable enough for later TSV aggregation.
    """
    return ReactionRuleStep(
        rule_smarts=serialized["rule_smarts"],
        center_atoms=frozenset(serialized["center_atoms"]),
        reaction_smiles=serialized["reaction_smiles"],
        reactant_center_molecules=tuple(
            deserialize_center_projection(projection)
            for projection in serialized["reactant_center_molecules"]
        ),
        product_center_molecules=tuple(
            deserialize_center_projection(projection)
            for projection in serialized["product_center_molecules"]
        ),
    )


class PrecomputedRuleExtractor:
    def __init__(
        self,
        serialized_steps_by_reaction: dict[str, SerializedReactionRuleStep | None],
        errors_by_reaction: dict[str, dict[str, str]] | None = None,
    ):
        """Initialize this object with its resolved configuration.

        The helper feeds rule extraction, center-overlap filtering, or final TSV
        aggregation without changing the original route JSON in place.
        """
        self.serialized_steps_by_reaction = serialized_steps_by_reaction
        self.errors_by_reaction = errors_by_reaction or {}
        self.cache: dict[str, ReactionRuleStep | None] = {}

    def extract(self, reaction_smiles: str) -> tuple[ReactionRuleStep | None, bool]:
        """Extract one rule from a mapped reaction SMILES string.

        The helper feeds rule extraction, center-overlap filtering, or final TSV
        aggregation without changing the original route JSON in place.
        """
        if reaction_smiles in self.errors_by_reaction:
            error = self.errors_by_reaction[reaction_smiles]
            raise RuleExtractionError(
                f"precomputed unique reaction extraction failed: "
                f"{error['error_type']}: {error['message']}"
            )
        if reaction_smiles in self.cache:
            return self.cache[reaction_smiles], True
        if reaction_smiles not in self.serialized_steps_by_reaction:
            raise KeyError("reaction was not present in the precomputed rule cache")

        serialized = self.serialized_steps_by_reaction[reaction_smiles]
        step = deserialize_reaction_rule_step(serialized) if serialized else None
        self.cache[reaction_smiles] = step
        return step, True


def collect_reaction_contexts(route: dict[str, Any]) -> list[tuple[str, str]]:
    """Collect reaction contexts from route-analysis input data.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    contexts: list[tuple[str, str]] = []

    def visit(node: dict[str, Any]) -> None:
        """Visit one route-tree node during recursive route traversal.

        The helper feeds rule extraction, center-overlap filtering, or final TSV
        aggregation without changing the original route JSON in place.
        """
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
    """Collect reaction SMILES from route-analysis input data.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    return [reaction_smiles for reaction_smiles, _ in collect_reaction_contexts(route)]


def extract_route_rule_sequences(
    route: dict[str, Any],
    rule_extractor: SynPlannerRuleExtractor,
    *,
    min_length: int,
    max_length: int | None,
    stats: RouteProcessingStats,
) -> tuple[dict[tuple[str, ...], set[str]], dict[tuple[str, ...], set[str]]]:
    """Extract route rule sequences from mapped route data.

    The result preserves reaction order and the shared-center constraint used to decide
    whether adjacent route steps form a valid composite rule.
    """
    route = normalize_route_tree(route)
    step_by_reaction_smiles: dict[str, ReactionRuleStep] = {}
    single_rules: dict[tuple[str, ...], set[str]] = defaultdict(set)

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
        single_rules[(step.rule_smarts,)].add(target_smiles)

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
    return sequences, single_rules


def extract_route_composites(
    route: dict[str, Any],
    rule_extractor: SynPlannerRuleExtractor,
    *,
    min_length: int,
    max_length: int | None,
    stats: RouteProcessingStats,
) -> dict[tuple[str, ...], set[str]]:
    """Extract route composites from mapped route data.

    The result preserves reaction order and the shared-center constraint used to decide
    whether adjacent route steps form a valid composite rule.
    """
    sequences, _single_rules = extract_route_rule_sequences(
        route,
        rule_extractor,
        min_length=min_length,
        max_length=max_length,
        stats=stats,
    )
    return sequences


def no_composite_reason(
    *,
    reactions_seen: int,
    extracted_reaction_rules: int,
    skipped_reactions: int,
    min_length: int,
) -> str:
    """Explain why a processed route produced no composite rules.

    The result preserves reaction order and the shared-center constraint used to decide
    whether adjacent route steps form a valid composite rule.
    """
    if reactions_seen == 0:
        return "no_reactions"
    if extracted_reaction_rules == 0 and skipped_reactions:
        return "all_reactions_skipped"
    if extracted_reaction_rules < min_length:
        return "fewer_than_min_length_extracted_reactions"
    return "no_reaction_center_sharing_sequence"


def rule_extractor_args_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Return rule extractor args dict used for rule extraction or comparison.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    return {
        "config": str(args.config) if getattr(args, "config", None) else None,
        "environment_atom_count": getattr(args, "environment_atom_count", 1),
        "include_rings": getattr(args, "include_rings", False),
        "keep_leaving_groups": getattr(args, "keep_leaving_groups", True),
        "keep_incoming_groups": getattr(args, "keep_incoming_groups", False),
        "reactor_validation": getattr(args, "reactor_validation", False),
    }


def limited_route_items(
    routes_json: Any,
    limit: int | None,
) -> list[tuple[Any, dict[str, Any]]]:
    """Yield route items after applying the optional route limit.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    items = []
    for index, item in enumerate(route_items(routes_json), start=1):
        if limit is not None and index > limit:
            break
        items.append(item)
    return items


def iter_route_items_from_json_file(
    path: Path,
    limit: int | None,
    *,
    chunk_size: int = 1024 * 1024,
) -> Iterable[tuple[Any, dict[str, Any]]]:
    """Stream top-level route entries from a list or dict JSON file.

    The PaRoutes/all-routes files can be larger than 1 GB. Loading the entire
    JSON before honoring ``--limit`` adds a large fixed cost and puts pressure on
    multiprocessing. This parser keeps only a small text buffer and the current
    route object in memory.
    """

    decoder = json.JSONDecoder()
    path = resolve_existing_path(path)

    with open_text(path) as file:
        buffer = ""
        position = 0
        eof = False

        def compact_buffer() -> None:
            """Discard consumed text from the streaming JSON parser buffer.

            The helper feeds rule extraction, center-overlap filtering, or final TSV
            aggregation without changing the original route JSON in place.
            """
            nonlocal buffer, position
            if position:
                buffer = buffer[position:]
                position = 0

        def read_more() -> None:
            """Read the next chunk for streaming JSON parsing.

            The helper feeds rule extraction, center-overlap filtering, or final TSV
            aggregation without changing the original route JSON in place.
            """
            nonlocal buffer, eof
            compact_buffer()
            chunk = file.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        def ensure_buffer() -> None:
            """Fill the streaming JSON parser buffer before decoding continues.

            The helper feeds rule extraction, center-overlap filtering, or final TSV
            aggregation without changing the original route JSON in place.
            """
            if position >= len(buffer) and not eof:
                read_more()

        def skip_whitespace() -> None:
            """Advance the streaming JSON parser past insignificant whitespace.

            The helper feeds rule extraction, center-overlap filtering, or final TSV
            aggregation without changing the original route JSON in place.
            """
            nonlocal position
            while True:
                ensure_buffer()
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return

        def expect_character(expected: str) -> None:
            """Consume the expected character from the streaming JSON parser.

            The helper feeds rule extraction, center-overlap filtering, or final TSV
            aggregation without changing the original route JSON in place.
            """
            nonlocal position
            skip_whitespace()
            ensure_buffer()
            if position >= len(buffer):
                raise ValueError(f"expected {expected!r}, reached end of JSON")
            actual = buffer[position]
            if actual != expected:
                raise ValueError(f"expected {expected!r}, got {actual!r}")
            position += 1

        def decode_next() -> Any:
            """Decode the next route object from the streaming JSON parser.

            The helper feeds rule extraction, center-overlap filtering, or final TSV
            aggregation without changing the original route JSON in place.
            """
            nonlocal position
            skip_whitespace()
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    return value
                except json.JSONDecodeError:
                    if eof:
                        raise
                    read_more()

        read_more()
        skip_whitespace()
        ensure_buffer()
        if position >= len(buffer):
            return

        root_start = buffer[position]
        position += 1
        seen = 0

        if root_start == "[":
            route_id = 0
            skip_whitespace()
            ensure_buffer()
            if position < len(buffer) and buffer[position] == "]":
                return
            while True:
                route = decode_next()
                yield route_id, route
                seen += 1
                if limit is not None and seen >= limit:
                    return
                route_id += 1

                skip_whitespace()
                ensure_buffer()
                if position >= len(buffer):
                    raise ValueError("unexpected end of JSON array")
                separator = buffer[position]
                position += 1
                if separator == "]":
                    return
                if separator != ",":
                    raise ValueError(f"expected ',' or ']', got {separator!r}")

        elif root_start == "{":
            skip_whitespace()
            ensure_buffer()
            if position < len(buffer) and buffer[position] == "}":
                return
            while True:
                route_id = decode_next()
                if not isinstance(route_id, str):
                    raise ValueError("top-level route JSON object keys must be strings")
                expect_character(":")
                route = decode_next()
                yield route_id, route
                seen += 1
                if limit is not None and seen >= limit:
                    return

                skip_whitespace()
                ensure_buffer()
                if position >= len(buffer):
                    raise ValueError("unexpected end of JSON object")
                separator = buffer[position]
                position += 1
                if separator == "}":
                    return
                if separator != ",":
                    raise ValueError(f"expected ',' or '}}', got {separator!r}")
        else:
            raise TypeError(f"unsupported routes JSON root: {root_start!r}")


def chunked_route_items(
    items: Iterable[tuple[Any, dict[str, Any]]],
    chunk_size: int,
) -> Iterable[list[tuple[Any, dict[str, Any]]]]:
    """Yield route items in chunks suitable for worker processes.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    chunk: list[tuple[Any, dict[str, Any]]] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def merge_route_processing_stats(
    target: RouteProcessingStats,
    source: RouteProcessingStats,
) -> None:
    """Merge route processing stats into aggregate results.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    target.routes_seen += source.routes_seen
    target.routes_with_composites += source.routes_with_composites
    target.reactions_seen += source.reactions_seen
    target.reaction_rule_cache_hits += source.reaction_rule_cache_hits
    target.reaction_rule_cache_misses += source.reaction_rule_cache_misses
    target.skipped_reactions += source.skipped_reactions
    target.errors += source.errors


def process_route_for_composites(
    route_id: Any,
    route: dict[str, Any],
    rule_extractor: SynPlannerRuleExtractor,
    *,
    min_length: int,
    max_length: int | None,
    store_route_without_composites: bool = True,
) -> dict[str, Any]:
    """Process route for composites and return serializable results.

    The result preserves reaction order and the shared-center constraint used to decide
    whether adjacent route steps form a valid composite rule.
    """
    stats = RouteProcessingStats(routes_seen=1)
    route_sequences, single_rules = extract_route_rule_sequences(
        route,
        rule_extractor,
        min_length=min_length,
        max_length=max_length,
        stats=stats,
    )
    route_without_composites = None
    reason = ""
    if route_sequences:
        stats.routes_with_composites = 1
    else:
        extracted_reaction_rules = stats.reactions_seen - stats.skipped_reactions
        reason = no_composite_reason(
            reactions_seen=stats.reactions_seen,
            extracted_reaction_rules=extracted_reaction_rules,
            skipped_reactions=stats.skipped_reactions,
            min_length=min_length,
        )
        if store_route_without_composites:
            route_without_composites = copy.deepcopy(route)
            metadata = route_without_composites.setdefault("metadata", {})
            metadata["composite_rule_extraction"] = {
                "route_id": route_id,
                "target_smiles": route_target_smiles(route),
                "reactions_seen": stats.reactions_seen,
                "extracted_reaction_rules": extracted_reaction_rules,
                "skipped_reactions": stats.skipped_reactions,
                "reason": reason,
            }
    return {
        "route_id": route_id,
        "route_sequences": route_sequences,
        "single_rules": single_rules,
        "route_without_composites": route_without_composites,
        "no_composite_reason": reason,
        "stats": stats,
        "error": None,
    }


def _init_composite_worker(
    extractor_args: dict[str, Any],
    min_length: int,
    max_length: int | None,
    store_routes_without_composites: bool,
) -> None:
    """Run the worker entry point for init composite.

    The result preserves reaction order and the shared-center constraint used to decide
    whether adjacent route steps form a valid composite rule.
    """
    global _COMPOSITE_WORKER_EXTRACTOR
    global _COMPOSITE_WORKER_MIN_LENGTH
    global _COMPOSITE_WORKER_MAX_LENGTH
    global _COMPOSITE_WORKER_STORE_ROUTES_WITHOUT_COMPOSITES
    setup_runtime_cache_dirs()
    _COMPOSITE_WORKER_EXTRACTOR = SynPlannerRuleExtractor.from_args(
        argparse.Namespace(**extractor_args)
    )
    _COMPOSITE_WORKER_MIN_LENGTH = min_length
    _COMPOSITE_WORKER_MAX_LENGTH = max_length
    _COMPOSITE_WORKER_STORE_ROUTES_WITHOUT_COMPOSITES = (
        store_routes_without_composites
    )


def _composite_route_worker(item: tuple[Any, dict[str, Any]]) -> dict[str, Any]:
    """Run the worker entry point for composite route.

    The result preserves reaction order and the shared-center constraint used to decide
    whether adjacent route steps form a valid composite rule.
    """
    route_id, route = item
    try:
        if _COMPOSITE_WORKER_EXTRACTOR is None:
            raise RuntimeError("composite worker was not initialized")
        return process_route_for_composites(
            route_id,
            route,
            _COMPOSITE_WORKER_EXTRACTOR,
            min_length=_COMPOSITE_WORKER_MIN_LENGTH,
            max_length=_COMPOSITE_WORKER_MAX_LENGTH,
            store_route_without_composites=(
                _COMPOSITE_WORKER_STORE_ROUTES_WITHOUT_COMPOSITES
            ),
        )
    except Exception as exc:
        return {
            "route_id": route_id,
            "route_sequences": {},
            "single_rules": {},
            "route_without_composites": None,
            "no_composite_reason": "",
            "stats": RouteProcessingStats(routes_seen=1, errors=1),
            "error": {
                "route_id": route_id,
                "stage": "extract_route_composites",
                "error_type": type(exc).__qualname__,
                "message": str(exc) or traceback.format_exc(limit=1).strip(),
            },
        }


def _composite_route_chunk_worker(
    items: list[tuple[Any, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Run the worker entry point for composite route chunk.

    The result preserves reaction order and the shared-center constraint used to decide
    whether adjacent route steps form a valid composite rule.
    """
    return [_composite_route_worker(item) for item in items]


def _init_precomputed_composite_worker(
    serialized_steps_by_reaction: dict[str, SerializedReactionRuleStep | None],
    errors_by_reaction: dict[str, dict[str, str]],
    min_length: int,
    max_length: int | None,
    store_routes_without_composites: bool,
) -> None:
    """Run the worker entry point for init precomputed composite.

    The result preserves reaction order and the shared-center constraint used to decide
    whether adjacent route steps form a valid composite rule.
    """
    global _COMPOSITE_WORKER_EXTRACTOR
    global _COMPOSITE_WORKER_MIN_LENGTH
    global _COMPOSITE_WORKER_MAX_LENGTH
    global _COMPOSITE_WORKER_STORE_ROUTES_WITHOUT_COMPOSITES
    setup_runtime_cache_dirs()
    _COMPOSITE_WORKER_EXTRACTOR = PrecomputedRuleExtractor(
        serialized_steps_by_reaction,
        errors_by_reaction,
    )
    _COMPOSITE_WORKER_MIN_LENGTH = min_length
    _COMPOSITE_WORKER_MAX_LENGTH = max_length
    _COMPOSITE_WORKER_STORE_ROUTES_WITHOUT_COMPOSITES = (
        store_routes_without_composites
    )


def _unique_reaction_worker(reaction_smiles: str) -> dict[str, Any]:
    """Run the worker entry point for unique reaction.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    try:
        if _COMPOSITE_WORKER_EXTRACTOR is None:
            raise RuntimeError("unique reaction worker was not initialized")
        step, _cache_hit = _COMPOSITE_WORKER_EXTRACTOR.extract(reaction_smiles)
        return {
            "reaction_smiles": reaction_smiles,
            "step": serialize_reaction_rule_step(step) if step is not None else None,
            "skipped": step is None,
            "error": None,
        }
    except Exception as exc:
        return {
            "reaction_smiles": reaction_smiles,
            "step": None,
            "skipped": False,
            "error": {
                "error_type": type(exc).__qualname__,
                "message": str(exc) or traceback.format_exc(limit=1).strip(),
            },
        }


def route_result_chunks_from_pool(
    executor: ProcessPoolExecutor,
    route_items_iter: Iterable[tuple[Any, dict[str, Any]]],
    *,
    worker_chunksize: int,
    max_pending_chunks: int,
) -> Iterable[list[dict[str, Any]]]:
    """Return route result chunks from pool from a normalized route tree.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    chunks = iter(chunked_route_items(route_items_iter, worker_chunksize))
    pending = set()

    def submit_until_full() -> None:
        """Submit route-processing jobs until the worker queue reaches capacity.

        The helper feeds rule extraction, center-overlap filtering, or final TSV
        aggregation without changing the original route JSON in place.
        """
        while len(pending) < max_pending_chunks:
            try:
                chunk = next(chunks)
            except StopIteration:
                return
            pending.add(executor.submit(_composite_route_chunk_worker, chunk))

    submit_until_full()
    while pending:
        done, pending_remaining = wait(pending, return_when=FIRST_COMPLETED)
        pending = pending_remaining
        for future in done:
            yield future.result()
        submit_until_full()


def collect_unique_normalized_reactions(
    args: argparse.Namespace,
    *,
    stats: RouteProcessingStats,
    errors: list[dict[str, Any]],
) -> tuple[set[str], set[Any]]:
    """Collect unique normalized reactions from route-analysis input data.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    unique_reactions: set[str] = set()
    failed_route_ids: set[Any] = set()

    for index, (route_id, route) in enumerate(
        iter_route_items_from_json_file(args.routes_json, args.limit),
        start=1,
    ):
        try:
            normalized_route = normalize_route_tree(route)
            for reaction_smiles, _target_smiles in collect_reaction_contexts(
                normalized_route
            ):
                unique_reactions.add(reaction_smiles)
        except Exception as exc:
            failed_route_ids.add(route_id)
            stats.routes_seen += 1
            stats.errors += 1
            error = {
                "route_id": route_id,
                "stage": "collect_unique_reactions",
                "error_type": type(exc).__qualname__,
                "message": str(exc) or traceback.format_exc(limit=1).strip(),
            }
            if not args.ignore_errors:
                raise RuntimeError(
                    f"route {route_id} failed during collect_unique_reactions: "
                    f"{error['error_type']}: {error['message']}"
                ) from exc
            errors.append(error)

        if args.progress_interval and index % args.progress_interval == 0:
            print(
                f"scanned routes={index} unique_reactions={len(unique_reactions)} "
                f"errors={stats.errors}",
                flush=True,
            )

    if args.progress_interval and (stats.routes_seen + len(unique_reactions)) == 0:
        print("scanned routes=0 unique_reactions=0 errors=0", flush=True)

    return unique_reactions, failed_route_ids


def extract_unique_reaction_rules(
    reaction_smiles_values: Iterable[str],
    args: argparse.Namespace,
    *,
    n_cpu: int,
    worker_chunksize: int,
) -> tuple[
    dict[str, SerializedReactionRuleStep | None],
    dict[str, dict[str, str]],
    dict[str, int],
]:
    """Extract unique reaction rules from mapped route data.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    reaction_smiles_list = sorted(reaction_smiles_values)
    serialized_steps_by_reaction: dict[str, SerializedReactionRuleStep | None] = {}
    errors_by_reaction: dict[str, dict[str, str]] = {}
    stats = {
        "unique_reactions_seen": len(reaction_smiles_list),
        "unique_reaction_rules_extracted": 0,
        "unique_reaction_rules_skipped": 0,
        "unique_reaction_rule_errors": 0,
    }

    def consume_result(result: dict[str, Any], index: int) -> None:
        """Merge one worker result into the aggregate state.

        The helper feeds rule extraction, center-overlap filtering, or final TSV
        aggregation without changing the original route JSON in place.
        """
        reaction_smiles = result["reaction_smiles"]
        error = result.get("error")
        if error:
            stats["unique_reaction_rule_errors"] += 1
            errors_by_reaction[reaction_smiles] = error
            if not args.ignore_errors:
                raise RuntimeError(
                    "unique reaction rule extraction failed: "
                    f"{error['error_type']}: {error['message']}"
                )
        else:
            serialized_steps_by_reaction[reaction_smiles] = result["step"]
            if result["skipped"]:
                stats["unique_reaction_rules_skipped"] += 1
            else:
                stats["unique_reaction_rules_extracted"] += 1

        if args.progress_interval and index % args.progress_interval == 0:
            print(
                f"extracted unique reactions={index}/{len(reaction_smiles_list)} "
                f"rules={stats['unique_reaction_rules_extracted']} "
                f"skipped={stats['unique_reaction_rules_skipped']} "
                f"errors={stats['unique_reaction_rule_errors']}",
                flush=True,
            )

    if n_cpu > 1 and reaction_smiles_list:
        with ProcessPoolExecutor(
            max_workers=n_cpu,
            initializer=_init_composite_worker,
            initargs=(
                rule_extractor_args_dict(args),
                args.min_length,
                args.max_length,
                False,
            ),
        ) as executor:
            for index, result in enumerate(
                executor.map(
                    _unique_reaction_worker,
                    reaction_smiles_list,
                    chunksize=worker_chunksize,
                ),
                start=1,
            ):
                consume_result(result, index)
    else:
        rule_extractor = SynPlannerRuleExtractor.from_args(args)
        for index, reaction_smiles in enumerate(reaction_smiles_list, start=1):
            try:
                step, _cache_hit = rule_extractor.extract(reaction_smiles)
                result = {
                    "reaction_smiles": reaction_smiles,
                    "step": (
                        serialize_reaction_rule_step(step)
                        if step is not None
                        else None
                    ),
                    "skipped": step is None,
                    "error": None,
                }
            except Exception as exc:
                result = {
                    "reaction_smiles": reaction_smiles,
                    "step": None,
                    "skipped": False,
                    "error": {
                        "error_type": type(exc).__qualname__,
                        "message": str(exc)
                        or traceback.format_exc(limit=1).strip(),
                    },
                }
            consume_result(result, index)

    return serialized_steps_by_reaction, errors_by_reaction, stats


def filtered_route_items_from_json_file(
    path: Path,
    limit: int | None,
    failed_route_ids: set[Any],
) -> Iterable[tuple[Any, dict[str, Any]]]:
    """Stream route items from JSON while applying optional route-ID filters.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    for route_id, route in iter_route_items_from_json_file(path, limit):
        if route_id in failed_route_ids:
            continue
        yield route_id, route


def consume_composite_route_result(
    result: dict[str, Any],
    index: int,
    *,
    args: argparse.Namespace,
    stats: RouteProcessingStats,
    references_by_sequence: dict[tuple[str, ...], set[Any]],
    target_molecules_by_sequence: dict[tuple[str, ...], dict[Any, set[str]]],
    references_by_single_rule: dict[tuple[str, ...], set[Any]],
    target_molecules_by_single_rule: dict[tuple[str, ...], dict[Any, set[str]]],
    errors: list[dict[str, Any]],
    routes_without_composites: dict[Any, dict[str, Any]],
    routes_without_composites_by_reason: dict[str, int],
) -> None:
    """Explain the consume composite route result helper used by route analysis.

    The result preserves reaction order and the shared-center constraint used to decide
    whether adjacent route steps form a valid composite rule.
    """
    merge_route_processing_stats(stats, result["stats"])
    error = result.get("error")
    if error:
        if not args.ignore_errors:
            raise RuntimeError(
                f"route {error['route_id']} failed during {error['stage']}: "
                f"{error['error_type']}: {error['message']}"
            )
        errors.append(error)
        return

    route_id = result["route_id"]
    route_sequences = result["route_sequences"]
    single_rules = result.get("single_rules", {})
    if result["no_composite_reason"]:
        reason = result["no_composite_reason"]
        routes_without_composites_by_reason[reason] += 1
    if result["route_without_composites"] is not None:
        routes_without_composites[route_id] = result["route_without_composites"]

    for sequence, target_molecules in single_rules.items():
        references_by_single_rule[sequence].add(route_id)
        target_molecules_by_single_rule[sequence][route_id].update(target_molecules)

    for sequence, target_molecules in route_sequences.items():
        references_by_sequence[sequence].add(route_id)
        target_molecules_by_sequence[sequence][route_id].update(target_molecules)

    if args.progress_interval and index % args.progress_interval == 0:
        print(
            f"processed routes={index} composite_rules={len(references_by_sequence)} "
            f"routes_without_composites={sum(routes_without_composites_by_reason.values())} "
            f"errors={stats.errors}",
            flush=True,
        )


def write_composite_extraction_outputs(
    args: argparse.Namespace,
    *,
    stats: RouteProcessingStats,
    references_by_sequence: dict[tuple[str, ...], set[Any]],
    target_molecules_by_sequence: dict[tuple[str, ...], dict[Any, set[str]]],
    references_by_single_rule: dict[tuple[str, ...], set[Any]],
    target_molecules_by_single_rule: dict[tuple[str, ...], dict[Any, set[str]]],
    errors: list[dict[str, Any]],
    routes_without_composites: dict[Any, dict[str, Any]],
    routes_without_composites_by_reason: dict[str, int],
    n_cpu: int,
    worker_chunksize: int,
    max_pending_chunks: int,
    store_routes_without_composites: bool,
    extraction_mode: str,
    extra_summary: dict[str, Any] | None = None,
) -> int:
    """Write composite extraction outputs to disk.

    The result preserves reaction order and the shared-center constraint used to decide
    whether adjacent route steps form a valid composite rule.
    """
    references_for_output, target_molecules_for_output = (
        merge_composite_sequences_by_querycgr(
            references_by_sequence,
            target_molecules_by_sequence,
        )
    )
    single_references_for_output, single_target_molecules_for_output = (
        merge_composite_sequences_by_querycgr(
            references_by_single_rule,
            target_molecules_by_single_rule,
        )
    )
    all_references_for_output = {
        **single_references_for_output,
        **references_for_output,
    }
    all_target_molecules_for_output = {
        **single_target_molecules_for_output,
        **target_molecules_for_output,
    }

    output_summary = write_composite_rules(
        args.output,
        all_references_for_output,
        target_molecules_by_sequence=all_target_molecules_for_output,
    )
    write_errors(args.output, errors)
    routes_without_composites_path = None
    if store_routes_without_composites:
        routes_without_composites_path = write_composite_routes_without_rules(
            args.output,
            routes_without_composites,
            getattr(args, "routes_without_composites_output", None),
        )
    routes_without_composites_count = sum(
        routes_without_composites_by_reason.values()
    )

    summary = {
        "routes_json": str(args.routes_json),
        "extraction_mode": extraction_mode,
        "routes_seen": stats.routes_seen,
        "routes_with_composite_rules": stats.routes_with_composites,
        "routes_without_composite_rules": routes_without_composites_count,
        "routes_without_composite_rules_file": (
            str(routes_without_composites_path)
            if routes_without_composites_path is not None
            else None
        ),
        "routes_without_composite_rules_output_skipped": (
            not store_routes_without_composites
        ),
        "routes_without_composite_rules_by_reason": dict(
            sorted(routes_without_composites_by_reason.items())
        ),
        "reactions_seen": stats.reactions_seen,
        "reaction_rule_cache_hits": stats.reaction_rule_cache_hits,
        "reaction_rule_cache_misses": stats.reaction_rule_cache_misses,
        "skipped_reactions": stats.skipped_reactions,
        "errors": stats.errors,
        "unique_composite_rules": len(references_for_output),
        "unique_single_step_rules": len(single_references_for_output),
        "raw_composite_rule_sequences": len(references_by_sequence),
        "raw_single_step_rules": len(references_by_single_rule),
        "target_molecule_occurrences": sum(
            len(targets)
            for route_targets in all_target_molecules_for_output.values()
            for targets in route_targets.values()
        ),
        "min_length": args.min_length,
        "max_length": args.max_length,
        "n_cpu": n_cpu,
        "worker_chunksize": worker_chunksize if n_cpu > 1 else 1,
        "max_pending_chunks": max_pending_chunks if n_cpu > 1 else 1,
        "output_prefix": str(args.output.with_suffix("")),
        **(extra_summary or {}),
        **output_summary,
    }
    summary_path = write_summary(args.output, summary)
    summary["summary_file"] = str(summary_path)
    write_summary(args.output, summary)
    sidecars = write_standard_sidecars(
        args.output.parent,
        command_name="extract-composite-rules",
        summary=summary,
        errors=errors,
        input_files=[args.routes_json],
        output_files={
            **output_summary.get("output_files", {}),
            "summary": summary_path,
            "routes_without_composite_rules": routes_without_composites_path or "",
        },
        config_path=getattr(args, "config", None),
        cli_args=args,
    )
    summary["sidecar_files"] = sidecars
    write_summary(args.output, summary)

    print(json.dumps(summary, indent=2), flush=True)
    return 0


def resolve_composite_output_args(args: argparse.Namespace) -> None:
    """Resolve --output or --output-dir into the existing output-prefix argument."""
    if getattr(args, "output", None) is None:
        dataset = dataset_prefix_from_path(args.routes_json)
        args.output = resolve_output_path(
            output=None,
            output_dir=getattr(args, "output_dir", None),
            default_filename=f"{dataset}.tsv",
        )
    elif getattr(args, "output_dir", None) is not None:
        raise ValueError("--output and --output-dir cannot be used together")


def empty_collection_state() -> dict[str, Any]:
    """Create empty aggregate containers for composite-rule collection.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    return {
        "references_by_sequence": defaultdict(set),
        "target_molecules_by_sequence": defaultdict(lambda: defaultdict(set)),
        "references_by_single_rule": defaultdict(set),
        "target_molecules_by_single_rule": defaultdict(lambda: defaultdict(set)),
        "errors": [],
        "routes_without_composites": {},
        "routes_without_composites_by_reason": defaultdict(int),
        "stats": RouteProcessingStats(),
    }


def run_unique_reactions_first(
    args: argparse.Namespace,
    *,
    n_cpu: int,
    worker_chunksize: int,
    max_pending_chunks: int,
    store_routes_without_composites: bool,
) -> int:
    """Run unique reactions first using configured inputs.

    Worker-safe inputs and outputs are used here so large route collections can be
    processed either serially or with `--n_cpu` parallelism.
    """
    state = empty_collection_state()
    unique_reactions, failed_route_ids = collect_unique_normalized_reactions(
        args,
        stats=state["stats"],
        errors=state["errors"],
    )
    serialized_steps_by_reaction, errors_by_reaction, unique_stats = (
        extract_unique_reaction_rules(
            unique_reactions,
            args,
            n_cpu=n_cpu,
            worker_chunksize=worker_chunksize,
        )
    )

    route_items_iter = filtered_route_items_from_json_file(
        args.routes_json,
        args.limit,
        failed_route_ids,
    )
    rule_extractor = PrecomputedRuleExtractor(
        serialized_steps_by_reaction,
        errors_by_reaction,
    )

    if n_cpu > 1:
        with ProcessPoolExecutor(
            max_workers=n_cpu,
            initializer=_init_precomputed_composite_worker,
            initargs=(
                serialized_steps_by_reaction,
                errors_by_reaction,
                args.min_length,
                args.max_length,
                store_routes_without_composites,
            ),
        ) as executor:
            index = 0
            for result_chunk in route_result_chunks_from_pool(
                executor,
                route_items_iter,
                worker_chunksize=worker_chunksize,
                max_pending_chunks=max_pending_chunks,
            ):
                for result in result_chunk:
                    index += 1
                    consume_composite_route_result(
                        result,
                        index,
                        args=args,
                        stats=state["stats"],
                        references_by_sequence=state["references_by_sequence"],
                        target_molecules_by_sequence=state[
                            "target_molecules_by_sequence"
                        ],
                        references_by_single_rule=state[
                            "references_by_single_rule"
                        ],
                        target_molecules_by_single_rule=state[
                            "target_molecules_by_single_rule"
                        ],
                        errors=state["errors"],
                        routes_without_composites=state[
                            "routes_without_composites"
                        ],
                        routes_without_composites_by_reason=state[
                            "routes_without_composites_by_reason"
                        ],
                    )
    else:
        for index, (route_id, route) in enumerate(route_items_iter, start=1):
            try:
                result = process_route_for_composites(
                    route_id,
                    route,
                    rule_extractor,
                    min_length=args.min_length,
                    max_length=args.max_length,
                    store_route_without_composites=store_routes_without_composites,
                )
            except Exception as exc:
                result = {
                    "route_id": route_id,
                    "route_sequences": {},
                    "single_rules": {},
                    "route_without_composites": None,
                    "no_composite_reason": "",
                    "stats": RouteProcessingStats(routes_seen=1, errors=1),
                    "error": {
                        "route_id": route_id,
                        "stage": "extract_route_composites",
                        "error_type": type(exc).__qualname__,
                        "message": str(exc) or traceback.format_exc(limit=1).strip(),
                    },
                }
            consume_composite_route_result(
                result,
                index,
                args=args,
                stats=state["stats"],
                references_by_sequence=state["references_by_sequence"],
                target_molecules_by_sequence=state["target_molecules_by_sequence"],
                references_by_single_rule=state["references_by_single_rule"],
                target_molecules_by_single_rule=state[
                    "target_molecules_by_single_rule"
                ],
                errors=state["errors"],
                routes_without_composites=state["routes_without_composites"],
                routes_without_composites_by_reason=state[
                    "routes_without_composites_by_reason"
                ],
            )

    if args.progress_interval and state["stats"].routes_seen % args.progress_interval:
        print(
            f"processed routes={state['stats'].routes_seen} "
            f"composite_rules={len(state['references_by_sequence'])} "
            f"routes_without_composites="
            f"{sum(state['routes_without_composites_by_reason'].values())} "
            f"errors={state['stats'].errors}",
            flush=True,
        )

    return write_composite_extraction_outputs(
        args,
        stats=state["stats"],
        references_by_sequence=state["references_by_sequence"],
        target_molecules_by_sequence=state["target_molecules_by_sequence"],
        references_by_single_rule=state["references_by_single_rule"],
        target_molecules_by_single_rule=state["target_molecules_by_single_rule"],
        errors=state["errors"],
        routes_without_composites=state["routes_without_composites"],
        routes_without_composites_by_reason=state["routes_without_composites_by_reason"],
        n_cpu=n_cpu,
        worker_chunksize=worker_chunksize,
        max_pending_chunks=max_pending_chunks,
        store_routes_without_composites=store_routes_without_composites,
        extraction_mode="unique_reactions_first",
        extra_summary={
            **unique_stats,
            "routes_failed_during_unique_reaction_scan": len(failed_route_ids),
        },
    )


def run(args: argparse.Namespace) -> int:
    """Run this module command with parsed CLI arguments.

    The helper feeds rule extraction, center-overlap filtering, or final TSV aggregation
    without changing the original route JSON in place.
    """
    setup_runtime_cache_dirs()
    resolve_composite_output_args(args)
    if args.min_length < 2:
        raise ValueError("--min-length must be at least 2")
    if args.max_length is not None and args.max_length <= 0:
        args.max_length = None
    if args.max_length is not None and args.max_length < args.min_length:
        raise ValueError("--max-length must be greater than or equal to --min-length")

    route_items_iter = iter_route_items_from_json_file(args.routes_json, args.limit)
    n_cpu = normalize_n_cpu(getattr(args, "n_cpu", 1))
    worker_chunksize = max(1, int(getattr(args, "worker_chunksize", 16) or 16))
    max_pending_chunks = max(
        n_cpu,
        int(getattr(args, "max_pending_chunks", 0) or n_cpu * 2),
    )
    store_routes_without_composites = not getattr(
        args,
        "skip_routes_without_composites_output",
        False,
    )
    if getattr(args, "unique_reactions_first", False):
        return run_unique_reactions_first(
            args,
            n_cpu=n_cpu,
            worker_chunksize=worker_chunksize,
            max_pending_chunks=max_pending_chunks,
            store_routes_without_composites=store_routes_without_composites,
        )

    references_by_sequence: dict[tuple[str, ...], set[Any]] = defaultdict(set)
    target_molecules_by_sequence: dict[tuple[str, ...], dict[Any, set[str]]] = (
        defaultdict(lambda: defaultdict(set))
    )
    references_by_single_rule: dict[tuple[str, ...], set[Any]] = defaultdict(set)
    target_molecules_by_single_rule: dict[tuple[str, ...], dict[Any, set[str]]] = (
        defaultdict(lambda: defaultdict(set))
    )
    errors: list[dict[str, Any]] = []
    routes_without_composites: dict[Any, dict[str, Any]] = {}
    routes_without_composites_by_reason: dict[str, int] = defaultdict(int)
    stats = RouteProcessingStats()

    def consume_result(result: dict[str, Any], index: int) -> None:
        """Merge one worker result into the aggregate state.

        The helper feeds rule extraction, center-overlap filtering, or final TSV
        aggregation without changing the original route JSON in place.
        """
        merge_route_processing_stats(stats, result["stats"])
        error = result.get("error")
        if error:
            if not args.ignore_errors:
                raise RuntimeError(
                    f"route {error['route_id']} failed during {error['stage']}: "
                    f"{error['error_type']}: {error['message']}"
                )
            errors.append(error)
            return

        route_id = result["route_id"]
        route_sequences = result["route_sequences"]
        single_rules = result.get("single_rules", {})
        if result["no_composite_reason"]:
            reason = result["no_composite_reason"]
            routes_without_composites_by_reason[reason] += 1
        if result["route_without_composites"] is not None:
            routes_without_composites[route_id] = result["route_without_composites"]

        for sequence, target_molecules in single_rules.items():
            references_by_single_rule[sequence].add(route_id)
            target_molecules_by_single_rule[sequence][route_id].update(
                target_molecules
            )

        for sequence, target_molecules in route_sequences.items():
            references_by_sequence[sequence].add(route_id)
            target_molecules_by_sequence[sequence][route_id].update(target_molecules)

        if args.progress_interval and index % args.progress_interval == 0:
            print(
                f"processed routes={index} composite_rules={len(references_by_sequence)} "
                f"routes_without_composites={sum(routes_without_composites_by_reason.values())} "
                f"errors={stats.errors}",
                flush=True,
            )

    if n_cpu > 1:
        with ProcessPoolExecutor(
            max_workers=n_cpu,
            initializer=_init_composite_worker,
            initargs=(
                rule_extractor_args_dict(args),
                args.min_length,
                args.max_length,
                store_routes_without_composites,
            ),
        ) as executor:
            index = 0
            for result_chunk in route_result_chunks_from_pool(
                executor,
                route_items_iter,
                worker_chunksize=worker_chunksize,
                max_pending_chunks=max_pending_chunks,
            ):
                for result in result_chunk:
                    index += 1
                    consume_result(result, index)
    else:
        rule_extractor = SynPlannerRuleExtractor.from_args(args)
        for index, (route_id, route) in enumerate(route_items_iter, start=1):
            try:
                result = process_route_for_composites(
                    route_id,
                    route,
                    rule_extractor,
                    min_length=args.min_length,
                    max_length=args.max_length,
                    store_route_without_composites=(
                        store_routes_without_composites
                    ),
                )
            except Exception as exc:
                result = {
                    "route_id": route_id,
                    "route_sequences": {},
                    "single_rules": {},
                    "route_without_composites": None,
                    "no_composite_reason": "",
                    "stats": RouteProcessingStats(routes_seen=1, errors=1),
                    "error": {
                        "route_id": route_id,
                        "stage": "extract_route_composites",
                        "error_type": type(exc).__qualname__,
                        "message": str(exc) or traceback.format_exc(limit=1).strip(),
                    },
                }
            consume_result(result, index)

    if args.progress_interval and stats.routes_seen % args.progress_interval:
        print(
            f"processed routes={stats.routes_seen} "
            f"composite_rules={len(references_by_sequence)} "
            f"routes_without_composites={sum(routes_without_composites_by_reason.values())} "
            f"errors={stats.errors}",
            flush=True,
        )

    return write_composite_extraction_outputs(
        args,
        stats=stats,
        references_by_sequence=references_by_sequence,
        target_molecules_by_sequence=target_molecules_by_sequence,
        references_by_single_rule=references_by_single_rule,
        target_molecules_by_single_rule=target_molecules_by_single_rule,
        errors=errors,
        routes_without_composites=routes_without_composites,
        routes_without_composites_by_reason=routes_without_composites_by_reason,
        n_cpu=n_cpu,
        worker_chunksize=worker_chunksize,
        max_pending_chunks=max_pending_chunks,
        store_routes_without_composites=store_routes_without_composites,
        extraction_mode="route_streaming",
    )
