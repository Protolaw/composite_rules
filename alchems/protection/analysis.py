from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from alchems.alchemical_rules.alchemical import (
    query_cgr_coarse_signature,
    query_cgr_isomorphic,
    rule_query_cgr,
)
from alchems.composite_rules.extract import (
    ReactionRuleStep,
    SynPlannerRuleExtractor,
    normalize_route_tree,
    reaction_smiles_from_node,
    route_items,
    route_target_smiles,
    valid_composite_sequence_occurrences,
)
from alchems.composite_rules.unwrap import split_composite_rule
from alchems.io import (
    expand_composite_rule_tsv_paths,
    read_tsv_rows,
    reference_sort_key,
    split_cell,
)
from alchems.protection.chython_rules import ProtectionRule


@dataclass
class ProtectionAnalysisConfig:
    min_composite_size: int = 2
    max_composite_size: int = 6
    similarity_threshold: float = 0.70
    include_multicenter: bool = True
    deprotection_first: bool = True
    querycgr_compare: bool = True
    keep_ambiguous_traces: bool = True
    collect_interval_rules: bool = True
    ignore_errors: bool = False
    write_debug_json: bool = False
    write_debug_svg: bool = False
    max_trace_depth: int = 50
    raw_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | None) -> "ProtectionAnalysisConfig":
        if path is None:
            return cls()
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        composite = data.get("composite_rules") or {}
        multicenter = data.get("multicenter") or {}
        tracing = data.get("tracing") or {}
        outputs = data.get("outputs") or {}
        errors = data.get("error_handling") or {}
        return cls(
            min_composite_size=int(composite.get("min_composite_size", 2)),
            max_composite_size=int(composite.get("max_composite_size", 6)),
            similarity_threshold=float(composite.get("similarity_threshold", 0.70)),
            include_multicenter=bool(multicenter.get("include_multicenter", True)),
            deprotection_first=bool(multicenter.get("deprotection_first", True)),
            querycgr_compare=bool(composite.get("compare_by_querycgr", True)),
            keep_ambiguous_traces=bool(tracing.get("keep_ambiguous_traces", True)),
            collect_interval_rules=bool(
                composite.get("collect_interval_composite_rules", True)
            ),
            ignore_errors=bool(errors.get("ignore_errors", False)),
            write_debug_json=bool(outputs.get("write_debug_json", False)),
            write_debug_svg=bool(outputs.get("write_debug_svg", False)),
            max_trace_depth=int(tracing.get("max_trace_depth", 50)),
            raw_config=data,
        )

    def with_cli_overrides(self, args: argparse.Namespace) -> "ProtectionAnalysisConfig":
        if getattr(args, "min_composite_size", None) is not None:
            self.min_composite_size = args.min_composite_size
        if getattr(args, "max_composite_size", None) is not None:
            self.max_composite_size = args.max_composite_size
        if getattr(args, "similarity_threshold", None) is not None:
            self.similarity_threshold = args.similarity_threshold
        if getattr(args, "include_multicenter", False):
            self.include_multicenter = True
        if getattr(args, "deprotection_first", False):
            self.deprotection_first = True
        if getattr(args, "querycgr_compare", False):
            self.querycgr_compare = True
        if getattr(args, "write_debug_json", False):
            self.write_debug_json = True
        if getattr(args, "write_debug_svg", False):
            self.write_debug_svg = True
        if getattr(args, "ignore_errors", False):
            self.ignore_errors = True
        return self


@dataclass
class MoleculeRecord:
    node_id: str
    node: dict[str, Any]
    smiles: str
    mapped_smiles: str
    depth: int
    parent_reaction_id: str | None = None


@dataclass
class ReactionRecord:
    node_id: str
    node: dict[str, Any]
    reaction_smiles: str
    depth: int
    parent_mol_id: str
    child_mol_ids: tuple[str, ...] = ()


@dataclass
class RouteIndex:
    route: dict[str, Any]
    target_smiles: str
    molecule_records: dict[str, MoleculeRecord]
    reaction_records: dict[str, ReactionRecord]
    reaction_order: tuple[str, ...]
    children_by_mol: dict[str, tuple[str, ...]]
    parent_mol_by_reaction: dict[str, str]
    child_mols_by_reaction: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class DeprotectionMatch:
    pg_type: str
    rule_id: str
    rule: ProtectionRule
    deprotection_node_id: str
    protected_precursor_node_id: str | None
    protected_precursor_smiles: str
    deprotected_product_smiles: str
    protected_atom_ids: tuple[int, ...]
    protected_query_atom_ids: tuple[int, ...]
    raw_mapping: tuple[tuple[int, int], ...]
    confidence: str
    multicenter_status: str
    n_other_reaction_centers: int
    n_matching_pg_sites_before_deprotection: int
    n_sites_deprotected: int
    selective_deprotection: bool
    ambiguous_rule_match: bool = False
    multiple_deprotection_rules: bool = False
    matched_rule_ids: tuple[str, ...] = ()
    matched_rule_names: tuple[str, ...] = ()


@dataclass
class TraceResult:
    trace_status: str
    source_type: str
    source_node_id: str | None = None
    protection_node_id: str | None = None
    stock_node_id: str | None = None
    stock_smiles: str = ""
    visited_reaction_ids: tuple[str, ...] = ()
    protected_molecule_ids: tuple[str, ...] = ()
    failure_reason: str = ""
    last_successful_node_id: str = ""
    last_successful_smiles: str = ""
    candidate_next_nodes: tuple[str, ...] = ()
    n_candidate_traces: int = 0
    debug_message: str = ""


@dataclass
class DeprotectionEvent:
    event_id: str
    route_id: str
    target_smiles: str
    pg_type: str
    protected_functional_group: str
    source_type: str
    trace_status: str
    confidence: str
    deprotection_node_id: str
    deprotection_reaction_smiles: str
    protection_node_id: str = ""
    protection_reaction_smiles: str = ""
    stock_node_id: str = ""
    stock_smiles: str = ""
    protected_precursor_smiles: str = ""
    deprotected_product_smiles: str = ""
    protected_atom_ids: tuple[int, ...] = ()
    protected_substructure_smiles: str = ""
    depth_deprotection_from_target: int = 0
    depth_source_from_target: int | None = None
    lifetime_steps: int = 0
    n_intervening_steps: int = 0
    multicenter_status: str = "unknown"
    n_other_reaction_centers: int = 0
    n_matching_pg_sites_before_deprotection: int = 0
    n_sites_deprotected: int = 0
    selective_deprotection: bool = False
    multiple_deprotection_rules: bool = False
    matched_deprotection_rules: tuple[str, ...] = ()
    failure_reason: str = ""
    interval_reaction_ids: tuple[str, ...] = ()
    trace_last_successful_node_id: str = ""
    trace_last_successful_smiles: str = ""
    trace_candidate_next_nodes: tuple[str, ...] = ()
    trace_n_candidate_traces: int = 0
    trace_debug_message: str = ""


@dataclass
class IntervalRuleObservation:
    event_id: str
    route_id: str
    pg_type: str
    trace_status: str
    interval_step_ids: tuple[str, ...]
    interval_reaction_smiles: tuple[str, ...]
    single_step_rules: tuple[str, ...]
    composite_rule_size: int
    composite_rule_smarts: str
    composite_rule_querycgr: str
    composite_rule_querycgr_hash: str
    composite_rule_family_id: str
    composite_rule_support_in_routes: int
    is_full_interval_rule: bool
    is_adjacent_window_rule: bool
    macro_depth_saving: int
    min_distance_pg_to_reaction_center: str = ""
    median_distance_pg_to_reaction_center: str = ""
    first_reaction_after_source: str = ""
    last_reaction_before_deprotection: str = ""


@dataclass
class CompositeRuleFamily:
    family_id: str
    composite_rule: str
    sequence: tuple[str, ...]
    querycgr_hash: str
    querycgr: str
    route_ids: set[str] = field(default_factory=set)
    target_molecules: set[str] = field(default_factory=set)
    source_rows: set[str] = field(default_factory=set)


@dataclass
class ProtectionAnalysisResult:
    route_stats_rows: list[dict[str, Any]]
    event_rows: list[dict[str, Any]]
    interval_rule_rows: list[dict[str, Any]]
    group_summary_rows: list[dict[str, Any]]
    rule_family_rows: list[dict[str, Any]]
    network_edge_rows: list[dict[str, Any]]
    trace_failure_rows: list[dict[str, Any]]
    summary: dict[str, Any]
    debug_routes: dict[str, dict[str, Any]] = field(default_factory=dict)


@lru_cache(maxsize=16384)
def parse_molecule(smiles_text: str) -> Any:
    from chython import smiles

    return smiles(smiles_text)


@lru_cache(maxsize=8192)
def parse_reaction(reaction_smiles: str) -> Any:
    from chython import smiles

    return smiles(reaction_smiles)


def canonical_molecule_string(molecule: Any) -> str:
    molecule_copy = molecule.copy()
    molecule_copy.canonicalize()
    return str(molecule_copy)


def same_molecule(left: Any, right: Any) -> bool:
    if left.atoms_count != right.atoms_count or left.bonds_count != right.bonds_count:
        return False
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


def molecule_node_mapped_smiles(node: dict[str, Any]) -> str:
    metadata = node.get("metadata") or {}
    return (
        node.get("mapped_smiles")
        or metadata.get("mapped_smiles")
        or metadata.get("mapped_molecule_smiles")
        or ""
    )


def molecule_node_smiles(node: dict[str, Any]) -> str:
    metadata = node.get("metadata") or {}
    return node.get("smiles") or metadata.get("smiles") or ""


def is_stock_molecule(node: dict[str, Any]) -> bool:
    return bool(node.get("in_stock") or (node.get("metadata") or {}).get("in_stock"))


def reaction_atom_ids(reaction: Any) -> set[int]:
    atom_ids: set[int] = set()
    for molecule in reaction.reactants + reaction.products + reaction.reagents:
        atom_ids.update(int(atom_id) for atom_id in molecule)
    return atom_ids


def remap_molecule(molecule: Any, mapping: dict[int, int]) -> Any:
    molecule_copy = molecule.copy()
    molecule_mapping = {
        atom_id: mapping[atom_id]
        for atom_id in molecule_copy
        if atom_id in mapping and atom_id != mapping[atom_id]
    }
    if molecule_mapping:
        molecule_copy.remap(molecule_mapping)
    return molecule_copy


def remap_reaction(reaction: Any, mapping: dict[int, int]) -> Any:
    from chython.containers import ReactionContainer

    return ReactionContainer(
        tuple(remap_molecule(molecule, mapping) for molecule in reaction.reactants),
        tuple(remap_molecule(molecule, mapping) for molecule in reaction.products),
        tuple(remap_molecule(molecule, mapping) for molecule in reaction.reagents),
        meta=dict(getattr(reaction, "meta", {}) or {}),
        name=getattr(reaction, "name", None),
    )


def find_matching_molecule(
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
            reference_molecule = parse_molecule(fallback_smiles)
        except Exception:
            reference_molecule = None
    if reference_molecule is None:
        return None, None

    for index, candidate in enumerate(candidates):
        if index in excluded:
            continue
        try:
            if same_molecule(candidate, reference_molecule):
                return index, candidate
        except Exception:
            continue
    return None, None


def format_mapped_molecule(molecule: Any) -> str:
    return format(molecule, "m")


def format_mapped_reaction(reaction: Any) -> str:
    return format(reaction, "m")


def set_node_mapped_smiles(node: dict[str, Any], molecule: Any) -> None:
    metadata = node.setdefault("metadata", {})
    metadata["mapped_smiles"] = format_mapped_molecule(molecule)


def normalize_route_tree_global_atom_maps(route: dict[str, Any]) -> dict[str, Any]:
    """Normalize a PaRoutes tree and make atom maps consistent across steps.

    PaRoutes stores mapped reaction SMILES per step, but those maps are local to
    each step. For protection tracing we need the same atom to keep the same map
    number as we move from a parent reaction reactant into the child reaction
    product. This walks the retrosynthetic tree from target to stock, aligns each
    child reaction product to the mapped molecule expected by its parent, and
    assigns fresh map numbers to atoms that are new to that branch.
    """

    route = normalize_route_tree(route)
    all_original_atom_ids: set[int] = set()

    def collect_original_atom_ids(node: dict[str, Any]) -> None:
        if node.get("type") == "reaction":
            try:
                all_original_atom_ids.update(reaction_atom_ids(parse_reaction(node["smiles"])))
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

        for atom_id in sorted(reaction_atom_ids(reaction)):
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
            set_node_mapped_smiles(node, expected_molecule)

        for child in node.get("children", []) or []:
            if not isinstance(child, dict) or child.get("type") != "reaction":
                continue
            try:
                reaction = parse_reaction(reaction_smiles_from_node(child))
            except Exception:
                visit_reaction_children(child)
                continue

            product_index, product = find_matching_molecule(
                reaction.products,
                expected_molecule,
                fallback_smiles=molecule_node_smiles(node),
            )
            alignment: dict[int, int] = {}
            if product is not None:
                if expected_molecule is None:
                    expected_product = product
                    alignment = {int(atom_id): int(atom_id) for atom_id in product}
                else:
                    mappings = list(product.get_mapping(expected_molecule))
                    if mappings:
                        alignment = {
                            int(source): int(target)
                            for source, target in mappings[0].items()
                        }
                        expected_product = expected_molecule
                    else:
                        expected_product = product
                mapping = complete_mapping(reaction, alignment)
                normalized_reaction = remap_reaction(reaction, mapping)
                child["smiles"] = format_mapped_reaction(normalized_reaction)

                normalized_product = None
                if product_index is not None:
                    try:
                        normalized_product = list(normalized_reaction.products)[product_index]
                    except Exception:
                        normalized_product = expected_product
                if normalized_product is not None:
                    set_node_mapped_smiles(node, normalized_product)
                visit_reaction_children(child, normalized_reaction)
            else:
                mapping = complete_mapping(reaction, {})
                normalized_reaction = remap_reaction(reaction, mapping)
                child["smiles"] = format_mapped_reaction(normalized_reaction)
                visit_reaction_children(child, normalized_reaction)

    def visit_reaction_children(
        reaction_node: dict[str, Any],
        reaction: Any | None = None,
    ) -> None:
        if reaction is None:
            try:
                reaction = parse_reaction(reaction_smiles_from_node(reaction_node))
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
            reactant_index, reactant = find_matching_molecule(
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


def build_route_index(route: dict[str, Any]) -> RouteIndex:
    route = normalize_route_tree(route)
    mol_counter = 0
    rxn_counter = 0
    molecule_records: dict[str, MoleculeRecord] = {}
    reaction_records: dict[str, ReactionRecord] = {}
    reaction_order: list[str] = []
    children_by_mol: dict[str, tuple[str, ...]] = {}
    parent_mol_by_reaction: dict[str, str] = {}
    child_mols_by_reaction: dict[str, tuple[str, ...]] = {}

    def next_mol_id() -> str:
        nonlocal mol_counter
        node_id = f"m{mol_counter}"
        mol_counter += 1
        return node_id

    def next_rxn_id() -> str:
        nonlocal rxn_counter
        node_id = f"r{rxn_counter}"
        rxn_counter += 1
        return node_id

    def visit_molecule(
        node: dict[str, Any],
        *,
        depth: int,
        parent_reaction_id: str | None,
    ) -> str:
        mol_id = str(node.get("node_id") or node.get("id") or next_mol_id())
        molecule_records[mol_id] = MoleculeRecord(
            node_id=mol_id,
            node=node,
            smiles=molecule_node_smiles(node),
            mapped_smiles=molecule_node_mapped_smiles(node),
            depth=depth,
            parent_reaction_id=parent_reaction_id,
        )
        reaction_ids: list[str] = []
        for child in node.get("children", []) or []:
            if not isinstance(child, dict) or child.get("type") != "reaction":
                continue
            rxn_id = str(child.get("node_id") or child.get("id") or next_rxn_id())
            reaction_ids.append(rxn_id)
            parent_mol_by_reaction[rxn_id] = mol_id
            child_mol_ids = visit_reaction(
                child,
                node_id=rxn_id,
                depth=depth + 1,
                parent_mol_id=mol_id,
            )
            child_mols_by_reaction[rxn_id] = tuple(child_mol_ids)
        children_by_mol[mol_id] = tuple(reaction_ids)
        return mol_id

    def visit_reaction(
        node: dict[str, Any],
        *,
        node_id: str,
        depth: int,
        parent_mol_id: str,
    ) -> list[str]:
        child_mol_ids: list[str] = []
        for child in node.get("children", []) or []:
            if isinstance(child, dict) and child.get("type") == "mol":
                child_mol_ids.append(
                    visit_molecule(
                        child,
                        depth=depth,
                        parent_reaction_id=node_id,
                    )
                )
        reaction_records[node_id] = ReactionRecord(
            node_id=node_id,
            node=node,
            reaction_smiles=reaction_smiles_from_node(node),
            depth=depth,
            parent_mol_id=parent_mol_id,
            child_mol_ids=tuple(child_mol_ids),
        )
        reaction_order.append(node_id)
        return child_mol_ids

    visit_molecule(route, depth=0, parent_reaction_id=None)
    return RouteIndex(
        route=route,
        target_smiles=route_target_smiles(route),
        molecule_records=molecule_records,
        reaction_records=reaction_records,
        reaction_order=tuple(reaction_order),
        children_by_mol=children_by_mol,
        parent_mol_by_reaction=parent_mol_by_reaction,
        child_mols_by_reaction=child_mols_by_reaction,
    )


def reaction_center_atoms(reaction_smiles: str) -> frozenset[int]:
    try:
        from synplan.chem.data.standardizing import RemoveReagentsStandardizer

        reaction = parse_reaction(reaction_smiles)
        standardized = RemoveReagentsStandardizer()(reaction)
        return frozenset((~standardized).center_atoms)
    except Exception:
        return frozenset()


def has_protected_pattern_for_atoms(
    molecule: Any,
    rule: ProtectionRule,
    protected_atom_ids: Iterable[int],
) -> bool:
    protected_atom_set = set(protected_atom_ids)
    for mapping in rule.query.get_mapping(molecule):
        kept_atoms = {
            mapping[query_atom]
            for query_atom in rule.atoms_to_keep
            if query_atom in mapping
        }
        if kept_atoms and kept_atoms <= protected_atom_set:
            return True
    return False


def has_protected_pattern(molecule: Any, rule: ProtectionRule) -> bool:
    return bool(matching_sites(molecule, rule))


def molecule_node_has_protected_pattern(
    molecule_record: MoleculeRecord,
    rule: ProtectionRule,
) -> bool:
    molecule_smiles = molecule_record.mapped_smiles or molecule_record.smiles
    if not molecule_smiles:
        return False
    try:
        return has_protected_pattern(parse_molecule(molecule_smiles), rule)
    except Exception:
        return False


def molecule_record_has_protected_atoms(
    molecule_record: MoleculeRecord,
    rule: ProtectionRule,
    protected_atom_ids: Iterable[int],
) -> bool:
    molecule_smiles = molecule_record.mapped_smiles or molecule_record.smiles
    if not molecule_smiles:
        return False
    try:
        return has_protected_pattern_for_atoms(
            parse_molecule(molecule_smiles),
            rule,
            protected_atom_ids,
        )
    except Exception:
        return False


def matching_site_mappings(
    molecule: Any,
    rule: ProtectionRule,
) -> list[tuple[tuple[int, ...], dict[int, int]]]:
    sites: list[tuple[tuple[int, ...], dict[int, int]]] = []
    for mapping in rule.query.get_mapping(molecule):
        atoms = tuple(
            mapping[query_atom]
            for query_atom in rule.atoms_to_keep
            if query_atom in mapping
        )
        if len(atoms) == len(rule.atoms_to_keep):
            protected_atom_ids = tuple(int(atom_id) for atom_id in atoms)
            if skip_rule_match(rule, molecule, protected_atom_ids):
                continue
            sites.append(
                (
                    protected_atom_ids,
                    {int(source): int(target) for source, target in mapping.items()},
                )
            )
    return sites


def matching_sites(molecule: Any, rule: ProtectionRule) -> list[tuple[int, ...]]:
    return [atoms for atoms, _mapping in matching_site_mappings(molecule, rule)]


def oxygen_attached_to_carbonyl_carbon(molecule: Any, atom_id: int) -> bool:
    if atom_id not in set(molecule):
        return False
    atom = molecule.atom(atom_id)
    if atom.atomic_number != 8:
        return False
    try:
        neighbors = molecule._bonds[atom_id]
    except Exception:
        return False
    for neighbor_id in neighbors:
        neighbor = molecule.atom(neighbor_id)
        if neighbor.atomic_number != 6:
            continue
        for second_neighbor_id, bond in molecule._bonds[neighbor_id].items():
            if second_neighbor_id == atom_id:
                continue
            second_neighbor = molecule.atom(second_neighbor_id)
            if second_neighbor.atomic_number == 8 and int(bond) == 2:
                return True
    return False


def skip_rule_match(
    rule: ProtectionRule,
    reactant: Any,
    protected_atom_ids: tuple[int, ...],
) -> bool:
    if rule.name == "hydroxyl_methyl" and protected_atom_ids:
        return oxygen_attached_to_carbonyl_carbon(reactant, protected_atom_ids[0])
    return False


def carboxyl_deprotection_rule(rule: ProtectionRule) -> bool:
    return rule.name.startswith("carboxyl_")


def atom_is_carboxylic_acid_carbon(molecule: Any, atom_id: int) -> bool:
    if atom_id not in set(molecule):
        return False
    atom = molecule.atom(atom_id)
    if atom.atomic_number != 6:
        return False
    try:
        neighbors = molecule._bonds[atom_id]
    except Exception:
        return False

    has_double_oxygen = False
    has_single_oxygen = False
    for neighbor_id, bond in neighbors.items():
        neighbor = molecule.atom(neighbor_id)
        if neighbor.atomic_number != 8:
            continue
        if int(bond) == 2:
            has_double_oxygen = True
        elif int(bond) == 1:
            has_single_oxygen = True
    return has_double_oxygen and has_single_oxygen


def molecule_contains_atom_ids(molecule: Any, atom_ids: Iterable[int]) -> bool:
    molecule_atom_ids = set(molecule)
    return set(atom_ids) <= molecule_atom_ids


def find_child_molecule_node(
    index: RouteIndex,
    rxn_record: ReactionRecord,
    reaction_side_molecule: Any,
) -> str | None:
    for child_mol_id in rxn_record.child_mol_ids:
        smiles_text = index.molecule_records[child_mol_id].smiles
        if not smiles_text:
            continue
        try:
            if same_molecule(parse_molecule(smiles_text), reaction_side_molecule):
                return child_mol_id
        except Exception:
            continue
    if len(rxn_record.child_mol_ids) == 1:
        return rxn_record.child_mol_ids[0]
    return None


def protected_functional_group(pg_type: str, molecule: Any, atom_ids: tuple[int, ...]) -> str:
    if pg_type.startswith("amine"):
        return "amine"
    if pg_type.startswith("carbonyl"):
        return "aldehyde/ketone"
    if pg_type.startswith("carboxyl"):
        return "carboxylic_acid"
    if pg_type.startswith("diol"):
        return "diol"
    if pg_type.startswith("thiol"):
        return "thiol"
    if pg_type.startswith("hydroxyl"):
        atom_id = atom_ids[0] if atom_ids else None
        if atom_id is not None and atom_id in set(molecule):
            try:
                for neighbor_id, _bond in molecule._bonds[atom_id].items():
                    neighbor = molecule.atom(neighbor_id)
                    if getattr(neighbor, "is_aromatic", False):
                        return "phenol"
            except Exception:
                pass
        return "alcohol"
    return pg_type.split("_", 1)[0] if "_" in pg_type else "unknown"


def classify_multicenter_status(
    reaction_smiles: str,
    deprotection_query_atoms: set[int],
) -> tuple[str, int]:
    centers = reaction_center_atoms(reaction_smiles)
    if not centers:
        return "unknown", 0
    other_centers = centers - deprotection_query_atoms
    if other_centers:
        try:
            reaction = parse_reaction(reaction_smiles)
            for molecule in reaction.reactants + reaction.products:
                atom_ids = set(molecule)
                if not (atom_ids & deprotection_query_atoms and atom_ids & other_centers):
                    continue
                for atom_1, atom_2, _bond in molecule.bonds():
                    if (
                        atom_1 in deprotection_query_atoms
                        and atom_2 in other_centers
                    ) or (
                        atom_2 in deprotection_query_atoms
                        and atom_1 in other_centers
                    ):
                        return "deprotective_combo", len(other_centers)
        except Exception:
            pass
        return "deprotection_plus_other", len(other_centers)
    return "single_center_deprotection", 0


def transformed_products(rule: ProtectionRule, molecule: Any) -> list[Any]:
    from chython import Transformer

    try:
        return list(Transformer(rule.query, rule.product_query)(molecule))
    except Exception:
        return []


def detect_deprotections(
    index: RouteIndex,
    rxn_record: ReactionRecord,
    protection_rules: dict[str, ProtectionRule],
    config: ProtectionAnalysisConfig,
) -> list[DeprotectionMatch]:
    reaction = parse_reaction(rxn_record.reaction_smiles)
    products = list(reaction.products)
    matches: list[DeprotectionMatch] = []

    for reactant in reaction.reactants:
        child_mol_id = find_child_molecule_node(index, rxn_record, reactant)
        reactant_smiles = canonical_molecule_string(reactant)
        for rule_id, rule in protection_rules.items():
            site_mappings = matching_site_mappings(reactant, rule)
            if not site_mappings:
                continue
            sites = [atoms for atoms, _mapping in site_mappings]
            transformed = transformed_products(rule, reactant)
            for protected_atom_ids, mapping in site_mappings:
                exact_product = None
                exact = False
                for candidate in transformed:
                    for product in products:
                        if same_molecule(candidate, product):
                            exact_product = product
                            exact = True
                            break
                    if exact_product is not None:
                        break

                product_with_mark = None
                product_still_protected = False
                for product in products:
                    if molecule_contains_atom_ids(product, protected_atom_ids):
                        product_with_mark = product
                        if has_protected_pattern_for_atoms(
                            product,
                            rule,
                            protected_atom_ids,
                        ):
                            product_still_protected = True
                        break

                if exact_product is None and (
                    not config.include_multicenter
                    or product_with_mark is None
                    or product_still_protected
                ):
                    continue

                product = exact_product or product_with_mark
                if product is not None and has_protected_pattern_for_atoms(
                    product,
                    rule,
                    protected_atom_ids,
                ):
                    continue
                if carboxyl_deprotection_rule(rule) and (
                    product is None
                    or not protected_atom_ids
                    or not atom_is_carboxylic_acid_carbon(product, protected_atom_ids[0])
                ):
                    continue
                confidence = "exact" if exact else "high"
                multicenter_status, n_other_centers = classify_multicenter_status(
                    rxn_record.reaction_smiles,
                    set(mapping.values()),
                )
                matches.append(
                    DeprotectionMatch(
                        pg_type=rule.name,
                        rule_id=rule_id,
                        rule=rule,
                        deprotection_node_id=rxn_record.node_id,
                        protected_precursor_node_id=child_mol_id,
                        protected_precursor_smiles=reactant_smiles,
                        deprotected_product_smiles=canonical_molecule_string(product),
                        protected_atom_ids=protected_atom_ids,
                        protected_query_atom_ids=tuple(rule.atoms_to_keep),
                        raw_mapping=tuple(sorted(mapping.items())),
                        confidence=confidence,
                        multicenter_status=multicenter_status,
                        n_other_reaction_centers=n_other_centers,
                        n_matching_pg_sites_before_deprotection=len(set(sites)),
                        n_sites_deprotected=1,
                        selective_deprotection=len(set(sites)) > 1,
                    )
                )

    suppressed_match_indexes: set[int] = set()
    for index, match in enumerate(matches):
        if match.pg_type != "hydroxyl_methyl":
            continue
        match_atoms = set(match.protected_atom_ids)
        for other_index, other in enumerate(matches):
            if index == other_index or other.pg_type == match.pg_type:
                continue
            if (
                other.deprotection_node_id == match.deprotection_node_id
                and other.protected_precursor_node_id
                == match.protected_precursor_node_id
                and match_atoms
                and match_atoms <= {target for _source, target in other.raw_mapping}
            ):
                suppressed_match_indexes.add(index)
                break
    if suppressed_match_indexes:
        matches = [
            match
            for index, match in enumerate(matches)
            if index not in suppressed_match_indexes
        ]

    grouped: dict[tuple[str, str | None, tuple[int, ...], str], list[DeprotectionMatch]] = (
        defaultdict(list)
    )
    for match in matches:
        grouped[
            (
                match.deprotection_node_id,
                match.protected_precursor_node_id,
                match.protected_atom_ids,
                match.pg_type,
            )
        ].append(match)

    deduplicated: list[DeprotectionMatch] = []
    for group in grouped.values():
        rule_ids = tuple(sorted({match.rule_id for match in group}))
        rule_names = tuple(sorted({match.pg_type for match in group}))
        deduplicated.append(
            replace(
                group[0],
                multiple_deprotection_rules=len(rule_ids) > 1,
                matched_rule_ids=rule_ids,
                matched_rule_names=rule_names,
            )
        )

    atom_groups: dict[tuple[str, str | None, tuple[int, ...]], list[DeprotectionMatch]] = (
        defaultdict(list)
    )
    for match in deduplicated:
        atom_groups[
            (
                match.deprotection_node_id,
                match.protected_precursor_node_id,
                match.protected_atom_ids,
            )
        ].append(match)

    final_matches: list[DeprotectionMatch] = []
    for group in atom_groups.values():
        rule_ids = tuple(
            sorted({rule_id for match in group for rule_id in match.matched_rule_ids})
        )
        rule_names = tuple(
            sorted({rule_name for match in group for rule_name in match.matched_rule_names})
        )
        for match in group:
            final_matches.append(
                replace(
                    match,
                    multiple_deprotection_rules=(
                        match.multiple_deprotection_rules or len(rule_names) > 1
                    ),
                    matched_rule_ids=rule_ids,
                    matched_rule_names=rule_names,
                )
            )
    return final_matches


def trace_protected_group_backward(
    index: RouteIndex,
    match: DeprotectionMatch,
    config: ProtectionAnalysisConfig,
) -> TraceResult:
    if match.protected_precursor_node_id is None:
        return TraceResult(
            trace_status="failed",
            source_type="unresolved",
            failure_reason="route_structure_unexpected",
            last_successful_node_id=match.deprotection_node_id,
            debug_message="could not map protected precursor to a route molecule node",
        )
    if match.ambiguous_rule_match:
        return TraceResult(
            trace_status="ambiguous",
            source_type="ambiguous",
            source_node_id=match.protected_precursor_node_id,
            failure_reason="multiple_deprotection_rules_match",
            last_successful_node_id=match.protected_precursor_node_id,
            n_candidate_traces=2,
            debug_message="multiple chython deprotection rules matched the same atom set",
        )

    current_mol_id = match.protected_precursor_node_id
    protected_atom_ids = match.protected_atom_ids
    visited_reaction_ids: list[str] = []
    protected_molecule_ids: list[str] = []

    for _depth in range(config.max_trace_depth):
        current_record = index.molecule_records[current_mol_id]
        if current_mol_id not in protected_molecule_ids:
            protected_molecule_ids.append(current_mol_id)
        if is_stock_molecule(current_record.node):
            return TraceResult(
                trace_status="stock",
                source_type="stock",
                source_node_id=current_mol_id,
                stock_node_id=current_mol_id,
                stock_smiles=current_record.smiles,
                visited_reaction_ids=tuple(visited_reaction_ids),
                protected_molecule_ids=tuple(protected_molecule_ids),
                last_successful_node_id=current_mol_id,
                last_successful_smiles=current_record.smiles,
            )

        next_reactions = index.children_by_mol.get(current_mol_id, ())
        if not next_reactions:
            return TraceResult(
                trace_status="failed",
                source_type="unresolved",
                visited_reaction_ids=tuple(visited_reaction_ids),
                protected_molecule_ids=tuple(protected_molecule_ids),
                failure_reason="no_parent_reaction",
                last_successful_node_id=current_mol_id,
                last_successful_smiles=current_record.smiles,
            )

        protected_candidates: list[tuple[str, str]] = []
        unprotected_candidates: list[str] = []
        candidate_reaction_ids: list[str] = []
        for next_rxn_id in next_reactions:
            rxn_record = index.reaction_records[next_rxn_id]
            candidate_reaction_ids.append(next_rxn_id)
            try:
                parse_reaction(rxn_record.reaction_smiles)
            except Exception:
                return TraceResult(
                    trace_status="failed",
                    source_type="unresolved",
                    visited_reaction_ids=tuple(visited_reaction_ids),
                    protected_molecule_ids=tuple(protected_molecule_ids),
                    failure_reason="reaction_parse_error",
                    last_successful_node_id=current_mol_id,
                    last_successful_smiles=current_record.smiles,
                    candidate_next_nodes=tuple(candidate_reaction_ids),
                )

            for child_mol_id in rxn_record.child_mol_ids:
                child_record = index.molecule_records[child_mol_id]
                if molecule_record_has_protected_atoms(
                    child_record,
                    match.rule,
                    protected_atom_ids,
                ):
                    protected_candidates.append((next_rxn_id, child_mol_id))
                elif child_record.smiles:
                    unprotected_candidates.append(child_mol_id)

        unique_protected = sorted(set(protected_candidates))
        if len(unique_protected) == 1:
            next_rxn_id, next_mol_id = unique_protected[0]
            visited_reaction_ids.append(next_rxn_id)
            current_mol_id = next_mol_id
            continue

        if len(unique_protected) > 1:
            return TraceResult(
                trace_status="ambiguous",
                source_type="ambiguous",
                visited_reaction_ids=tuple(visited_reaction_ids),
                protected_molecule_ids=tuple(protected_molecule_ids),
                failure_reason="multiple_candidate_ancestors",
                last_successful_node_id=current_mol_id,
                last_successful_smiles=current_record.smiles,
                candidate_next_nodes=tuple(
                    f"{rxn_id}:{mol_id}" for rxn_id, mol_id in unique_protected
                ),
                n_candidate_traces=len(unique_protected),
            )

        unique_unprotected_candidates = sorted(set(unprotected_candidates))
        if unique_unprotected_candidates:
            if len(next_reactions) == 1:
                protection_node_id = next_reactions[0]
                return TraceResult(
                    trace_status="introduced",
                    source_type="introduced",
                    source_node_id=protection_node_id,
                    protection_node_id=protection_node_id,
                    visited_reaction_ids=tuple(visited_reaction_ids),
                    protected_molecule_ids=tuple(protected_molecule_ids),
                    last_successful_node_id=current_mol_id,
                    last_successful_smiles=current_record.smiles,
                    candidate_next_nodes=tuple(unique_unprotected_candidates),
                    n_candidate_traces=len(unique_unprotected_candidates),
                )
            return TraceResult(
                trace_status="ambiguous",
                source_type="ambiguous",
                visited_reaction_ids=tuple(visited_reaction_ids),
                protected_molecule_ids=tuple(protected_molecule_ids),
                failure_reason="multiple_candidate_ancestors",
                last_successful_node_id=current_mol_id,
                last_successful_smiles=current_record.smiles,
                candidate_next_nodes=tuple(unique_unprotected_candidates),
                n_candidate_traces=len(unique_unprotected_candidates),
            )

        return TraceResult(
            trace_status="failed",
            source_type="unresolved",
            visited_reaction_ids=tuple(visited_reaction_ids),
            protected_molecule_ids=tuple(protected_molecule_ids),
            failure_reason="mapping_lost",
            last_successful_node_id=current_mol_id,
            last_successful_smiles=current_record.smiles,
            candidate_next_nodes=tuple(candidate_reaction_ids),
        )

    return TraceResult(
        trace_status="failed",
        source_type="unresolved",
        visited_reaction_ids=tuple(visited_reaction_ids),
        protected_molecule_ids=tuple(protected_molecule_ids),
        failure_reason="max_trace_depth_exceeded",
        last_successful_node_id=current_mol_id,
        last_successful_smiles=index.molecule_records[current_mol_id].smiles,
    )


def event_from_match_and_trace(
    route_id: str,
    event_index: int,
    index: RouteIndex,
    match: DeprotectionMatch,
    trace: TraceResult,
) -> DeprotectionEvent:
    rxn_record = index.reaction_records[match.deprotection_node_id]
    protection_reaction_smiles = ""
    if trace.protection_node_id:
        protection_reaction_smiles = index.reaction_records[
            trace.protection_node_id
        ].reaction_smiles
    depth_source = None
    if trace.stock_node_id:
        depth_source = index.molecule_records[trace.stock_node_id].depth
    elif trace.protection_node_id:
        depth_source = index.reaction_records[trace.protection_node_id].depth

    precursor_molecule = parse_molecule(match.protected_precursor_smiles)
    if trace.source_type == "stock":
        lifetime_steps = max(0, len(trace.protected_molecule_ids) - 2)
    elif trace.source_type == "introduced":
        lifetime_steps = max(0, len(trace.protected_molecule_ids) - 1)
    else:
        lifetime_steps = len(trace.protected_molecule_ids)
    return DeprotectionEvent(
        event_id=f"{route_id}:pg{event_index}",
        route_id=str(route_id),
        target_smiles=index.target_smiles,
        pg_type=match.pg_type,
        protected_functional_group=protected_functional_group(
            match.pg_type,
            precursor_molecule,
            match.protected_atom_ids,
        ),
        source_type=trace.source_type,
        trace_status=trace.trace_status,
        confidence=match.confidence,
        deprotection_node_id=match.deprotection_node_id,
        deprotection_reaction_smiles=rxn_record.reaction_smiles,
        protection_node_id=trace.protection_node_id or "",
        protection_reaction_smiles=protection_reaction_smiles,
        stock_node_id=trace.stock_node_id or "",
        stock_smiles=trace.stock_smiles,
        protected_precursor_smiles=match.protected_precursor_smiles,
        deprotected_product_smiles=match.deprotected_product_smiles,
        protected_atom_ids=match.protected_atom_ids,
        protected_substructure_smiles=match.protected_precursor_smiles,
        depth_deprotection_from_target=rxn_record.depth,
        depth_source_from_target=depth_source,
        lifetime_steps=lifetime_steps,
        n_intervening_steps=lifetime_steps,
        multicenter_status=match.multicenter_status,
        n_other_reaction_centers=match.n_other_reaction_centers,
        n_matching_pg_sites_before_deprotection=(
            match.n_matching_pg_sites_before_deprotection
        ),
        n_sites_deprotected=match.n_sites_deprotected,
        selective_deprotection=match.selective_deprotection,
        multiple_deprotection_rules=match.multiple_deprotection_rules,
        matched_deprotection_rules=match.matched_rule_names or (match.pg_type,),
        failure_reason=trace.failure_reason,
        interval_reaction_ids=trace.visited_reaction_ids,
        trace_last_successful_node_id=trace.last_successful_node_id,
        trace_last_successful_smiles=trace.last_successful_smiles,
        trace_candidate_next_nodes=trace.candidate_next_nodes,
        trace_n_candidate_traces=trace.n_candidate_traces,
        trace_debug_message=trace.debug_message,
    )


def sequence_querycgr_parts(sequence: tuple[str, ...]) -> tuple[Any, ...]:
    parts = []
    for rule_smarts in sequence:
        query_cgr = rule_query_cgr(rule_smarts)
        parts.append(query_cgr_coarse_signature(query_cgr))
    return tuple(parts)


def sequence_querycgr_hash(sequence: tuple[str, ...]) -> tuple[str, str]:
    try:
        parts = sequence_querycgr_parts(sequence)
        querycgr_text = json.dumps(parts, sort_keys=True, default=repr)
        digest = hashlib.sha1(querycgr_text.encode("utf-8")).hexdigest()[:16]
        return querycgr_text, digest
    except Exception:
        fallback = "$".join(sequence)
        digest = hashlib.sha1(fallback.encode("utf-8")).hexdigest()[:16]
        return fallback, digest


def sequences_querycgr_isomorphic(
    left: tuple[str, ...],
    right: tuple[str, ...],
) -> bool:
    if len(left) != len(right):
        return False
    try:
        return all(
            query_cgr_isomorphic(rule_query_cgr(left_rule), rule_query_cgr(right_rule))
            for left_rule, right_rule in zip(left, right)
        )
    except Exception:
        return False


def load_composite_rule_index(paths: Iterable[Path] | None) -> dict[str, CompositeRuleFamily]:
    if not paths:
        return {}
    families_by_hash: dict[str, CompositeRuleFamily] = {}
    expanded = expand_composite_rule_tsv_paths(paths)
    for tsv_path in expanded:
        fieldnames, rows = read_tsv_rows(tsv_path)
        if "Composite_rule" not in fieldnames:
            continue
        for row_index, row in enumerate(rows):
            composite_rule = (row.get("Composite_rule") or "").strip()
            if not composite_rule:
                continue
            sequence = tuple(split_composite_rule(composite_rule))
            querycgr, query_hash = sequence_querycgr_hash(sequence)
            family = families_by_hash.get(query_hash)
            if family is None:
                family = CompositeRuleFamily(
                    family_id=f"crf_{len(families_by_hash) + 1:06d}",
                    composite_rule=composite_rule,
                    sequence=sequence,
                    querycgr_hash=query_hash,
                    querycgr=querycgr,
                )
                families_by_hash[query_hash] = family
            family.route_ids.update(split_cell(row.get("Reference")))
            family.target_molecules.update(split_cell(row.get("Target_molecules")))
            family.source_rows.add(f"{tsv_path}:{row_index}")
    return families_by_hash


def default_rule_extractor() -> SynPlannerRuleExtractor:
    args = argparse.Namespace(
        config=None,
        environment_atom_count=1,
        include_rings=False,
        keep_leaving_groups=True,
        keep_incoming_groups=False,
        reactor_validation=False,
    )
    return SynPlannerRuleExtractor.from_args(args)


def match_family_for_sequence(
    sequence: tuple[str, ...],
    families_by_hash: dict[str, CompositeRuleFamily],
) -> CompositeRuleFamily | None:
    _querycgr, query_hash = sequence_querycgr_hash(sequence)
    return families_by_hash.get(query_hash)


def collect_interval_rule_observations(
    event: DeprotectionEvent,
    index: RouteIndex,
    families_by_hash: dict[str, CompositeRuleFamily],
    config: ProtectionAnalysisConfig,
    rule_extractor: SynPlannerRuleExtractor | None,
) -> list[IntervalRuleObservation]:
    if not config.collect_interval_rules or not event.interval_reaction_ids:
        return []
    if rule_extractor is None:
        return []

    steps: list[ReactionRuleStep] = []
    for rxn_id in event.interval_reaction_ids:
        rxn_record = index.reaction_records[rxn_id]
        step, _cache_hit = rule_extractor.extract(rxn_record.reaction_smiles)
        if step is None:
            continue
        parent_mol = index.molecule_records[rxn_record.parent_mol_id]
        steps.append(
            ReactionRuleStep(
                rule_smarts=step.rule_smarts,
                center_atoms=step.center_atoms,
                reaction_smiles=step.reaction_smiles,
                target_smiles=parent_mol.smiles,
                reactant_center_molecules=step.reactant_center_molecules,
                product_center_molecules=step.product_center_molecules,
            )
        )

    observations: list[IntervalRuleObservation] = []
    if not steps:
        return observations

    for sequence, _target_smiles in valid_composite_sequence_occurrences(
        steps,
        min_length=config.min_composite_size,
        max_length=config.max_composite_size,
    ):
        family = match_family_for_sequence(sequence, families_by_hash)
        querycgr, query_hash = sequence_querycgr_hash(sequence)
        family_id = family.family_id if family else f"interval_{query_hash}"
        support = len(family.route_ids) if family else 0
        composite_rule = "$".join(sequence)
        observations.append(
            IntervalRuleObservation(
                event_id=event.event_id,
                route_id=event.route_id,
                pg_type=event.pg_type,
                trace_status=event.trace_status,
                interval_step_ids=event.interval_reaction_ids,
                interval_reaction_smiles=tuple(
                    index.reaction_records[rxn_id].reaction_smiles
                    for rxn_id in event.interval_reaction_ids
                ),
                single_step_rules=tuple(step.rule_smarts for step in steps),
                composite_rule_size=len(sequence),
                composite_rule_smarts=composite_rule,
                composite_rule_querycgr=querycgr,
                composite_rule_querycgr_hash=query_hash,
                composite_rule_family_id=family_id,
                composite_rule_support_in_routes=support,
                is_full_interval_rule=len(sequence) == len(steps),
                is_adjacent_window_rule=True,
                macro_depth_saving=len(sequence) - 1,
                first_reaction_after_source=(
                    steps[-1].reaction_smiles if steps else ""
                ),
                last_reaction_before_deprotection=(
                    steps[0].reaction_smiles if steps else ""
                ),
            )
        )
    return observations


def analyze_route_protection(
    route: dict[str, Any],
    route_id: Any,
    protection_rules: dict[str, ProtectionRule],
    composite_rule_index: dict[str, CompositeRuleFamily] | None = None,
    config: ProtectionAnalysisConfig | None = None,
    rule_extractor: SynPlannerRuleExtractor | None = None,
) -> tuple[list[DeprotectionEvent], list[IntervalRuleObservation], RouteIndex]:
    config = config or ProtectionAnalysisConfig()
    families = composite_rule_index or {}
    index = build_route_index(route)
    events: list[DeprotectionEvent] = []
    interval_rules: list[IntervalRuleObservation] = []

    for rxn_id in index.reaction_order:
        rxn_record = index.reaction_records[rxn_id]
        matches = detect_deprotections(index, rxn_record, protection_rules, config)
        for match in matches:
            trace = trace_protected_group_backward(index, match, config)
            event = event_from_match_and_trace(
                str(route_id),
                len(events) + 1,
                index,
                match,
                trace,
            )
            events.append(event)
            interval_rules.extend(
                collect_interval_rule_observations(
                    event,
                    index,
                    families,
                    config,
                    rule_extractor,
                )
            )
    return events, interval_rules, index


def event_to_row(event: DeprotectionEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "route_id": event.route_id,
        "target_smiles": event.target_smiles,
        "pg_type": event.pg_type,
        "protected_functional_group": event.protected_functional_group,
        "source_type": event.source_type,
        "trace_status": event.trace_status,
        "confidence": event.confidence,
        "deprotection_node_id": event.deprotection_node_id,
        "deprotection_reaction_smiles": event.deprotection_reaction_smiles,
        "protection_node_id": event.protection_node_id,
        "protection_reaction_smiles": event.protection_reaction_smiles,
        "stock_node_id": event.stock_node_id,
        "stock_smiles": event.stock_smiles,
        "protected_precursor_smiles": event.protected_precursor_smiles,
        "deprotected_product_smiles": event.deprotected_product_smiles,
        "protected_atom_ids": ",".join(map(str, event.protected_atom_ids)),
        "protected_substructure_smiles": event.protected_substructure_smiles,
        "depth_deprotection_from_target": event.depth_deprotection_from_target,
        "depth_source_from_target": (
            "" if event.depth_source_from_target is None else event.depth_source_from_target
        ),
        "lifetime_steps": event.lifetime_steps,
        "n_intervening_steps": event.n_intervening_steps,
        "multicenter_status": event.multicenter_status,
        "n_other_reaction_centers": event.n_other_reaction_centers,
        "n_matching_pg_sites_before_deprotection": (
            event.n_matching_pg_sites_before_deprotection
        ),
        "n_sites_deprotected": event.n_sites_deprotected,
        "selective_deprotection": event.selective_deprotection,
        "multiple_deprotection_rules": event.multiple_deprotection_rules,
        "matched_deprotection_rules": ",".join(event.matched_deprotection_rules),
        "failure_reason": event.failure_reason,
    }


def interval_rule_to_row(observation: IntervalRuleObservation) -> dict[str, Any]:
    return {
        "event_id": observation.event_id,
        "route_id": observation.route_id,
        "pg_type": observation.pg_type,
        "trace_status": observation.trace_status,
        "interval_step_ids": ",".join(observation.interval_step_ids),
        "interval_reaction_smiles": " || ".join(observation.interval_reaction_smiles),
        "single_step_rules": " || ".join(observation.single_step_rules),
        "composite_rule_size": observation.composite_rule_size,
        "composite_rule_smarts": observation.composite_rule_smarts,
        "composite_rule_querycgr": observation.composite_rule_querycgr,
        "composite_rule_querycgr_hash": observation.composite_rule_querycgr_hash,
        "composite_rule_family_id": observation.composite_rule_family_id,
        "composite_rule_support_in_routes": observation.composite_rule_support_in_routes,
        "is_full_interval_rule": observation.is_full_interval_rule,
        "is_adjacent_window_rule": observation.is_adjacent_window_rule,
        "macro_depth_saving": observation.macro_depth_saving,
        "min_distance_pg_to_reaction_center": (
            observation.min_distance_pg_to_reaction_center
        ),
        "median_distance_pg_to_reaction_center": (
            observation.median_distance_pg_to_reaction_center
        ),
        "first_reaction_after_source": observation.first_reaction_after_source,
        "last_reaction_before_deprotection": observation.last_reaction_before_deprotection,
    }


def trace_failure_row(
    event: DeprotectionEvent,
    trace: TraceResult | None = None,
) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "route_id": event.route_id,
        "pg_type": event.pg_type,
        "deprotection_node_id": event.deprotection_node_id,
        "protected_precursor_smiles": event.protected_precursor_smiles,
        "protected_atom_ids": ",".join(map(str, event.protected_atom_ids)),
        "trace_status": event.trace_status,
        "failure_reason": event.failure_reason,
        "last_successful_node_id": event.trace_last_successful_node_id,
        "last_successful_smiles": event.trace_last_successful_smiles,
        "candidate_next_nodes": ",".join(event.trace_candidate_next_nodes),
        "n_candidate_traces": event.trace_n_candidate_traces,
        "debug_message": event.trace_debug_message,
    }


def route_stats_row(
    route_id: str,
    index: RouteIndex,
    events: list[DeprotectionEvent],
    interval_rules: list[IntervalRuleObservation],
) -> dict[str, Any]:
    lifetimes = [event.lifetime_steps for event in events]
    pg_types = sorted({event.pg_type for event in events})
    family_ids = {obs.composite_rule_family_id for obs in interval_rules}
    max_simultaneous = max_simultaneous_pg(events)
    return {
        "route_id": route_id,
        "target_smiles": index.target_smiles,
        "n_steps": len(index.reaction_records),
        "n_molecule_nodes": len(index.molecule_records),
        "n_reaction_nodes": len(index.reaction_records),
        "n_in_stock_leaves": sum(
            1 for record in index.molecule_records.values() if is_stock_molecule(record.node)
        ),
        "n_protecting_groups_total": len(events),
        "n_pg_types": len(pg_types),
        "pg_types": ",".join(pg_types),
        "n_pg_introduced": sum(1 for event in events if event.source_type == "introduced"),
        "n_pg_stock": sum(1 for event in events if event.source_type == "stock"),
        "n_pg_ambiguous": sum(1 for event in events if event.trace_status == "ambiguous"),
        "n_pg_failed": sum(1 for event in events if event.trace_status == "failed"),
        "n_deprotection_events": len(events),
        "n_multicenter_deprotection_events": sum(
            1
            for event in events
            if event.multicenter_status
            in {"deprotection_plus_other", "deprotective_combo"}
        ),
        "n_deprotective_combo_events": sum(
            1 for event in events if event.multicenter_status == "deprotective_combo"
        ),
        "max_simultaneous_pg": max_simultaneous,
        "mean_pg_lifetime_steps": (
            f"{statistics.mean(lifetimes):.3f}" if lifetimes else ""
        ),
        "median_pg_lifetime_steps": (
            f"{statistics.median(lifetimes):.3f}" if lifetimes else ""
        ),
        "n_interval_composite_rules": len(interval_rules),
        "n_unique_interval_composite_rule_families": len(family_ids),
    }


def max_simultaneous_pg(events: list[DeprotectionEvent]) -> int:
    if not events:
        return 0
    points = sorted(
        {
            event.depth_deprotection_from_target
            for event in events
            if event.depth_source_from_target is not None
        }
        | {
            int(event.depth_source_from_target)
            for event in events
            if event.depth_source_from_target is not None
        }
    )
    maximum = 1 if events else 0
    for point in points:
        active = 0
        for event in events:
            if event.depth_source_from_target is None:
                continue
            low = min(event.depth_deprotection_from_target, event.depth_source_from_target)
            high = max(event.depth_deprotection_from_target, event.depth_source_from_target)
            if low <= point <= high:
                active += 1
        maximum = max(maximum, active)
    return maximum


def summarize_groups(
    events: list[DeprotectionEvent],
    interval_rules: list[IntervalRuleObservation],
) -> list[dict[str, Any]]:
    events_by_pg: dict[str, list[DeprotectionEvent]] = defaultdict(list)
    rules_by_pg: dict[str, list[IntervalRuleObservation]] = defaultdict(list)
    for event in events:
        events_by_pg[event.pg_type].append(event)
    for observation in interval_rules:
        rules_by_pg[observation.pg_type].append(observation)

    rows = []
    for pg_type, pg_events in events_by_pg.items():
        lifetimes = [event.lifetime_steps for event in pg_events]
        route_ids = {event.route_id for event in pg_events}
        targets = {event.target_smiles for event in pg_events}
        pg_rules = rules_by_pg.get(pg_type, [])
        family_counts = Counter(obs.composite_rule_family_id for obs in pg_rules)
        first_reactions = Counter(
            obs.first_reaction_after_source
            for obs in pg_rules
            if obs.first_reaction_after_source
        )
        last_reactions = Counter(
            obs.last_reaction_before_deprotection
            for obs in pg_rules
            if obs.last_reaction_before_deprotection
        )
        functional_groups = Counter(event.protected_functional_group for event in pg_events)
        introduced = sum(1 for event in pg_events if event.source_type == "introduced")
        stock = sum(1 for event in pg_events if event.source_type == "stock")
        failed = sum(1 for event in pg_events if event.trace_status == "failed")
        ambiguous = sum(1 for event in pg_events if event.trace_status == "ambiguous")
        selective = sum(1 for event in pg_events if event.selective_deprotection)
        popularity = len(pg_events)
        rows.append(
            {
                "pg_type": pg_type,
                "popularity": popularity,
                "route_count": len(route_ids),
                "target_count": len(targets),
                "introduced_count": introduced,
                "stock_count": stock,
                "ambiguous_count": ambiguous,
                "failed_count": failed,
                "introduction_fraction": f"{introduced / popularity:.6f}",
                "stock_fraction": f"{stock / popularity:.6f}",
                "failure_fraction": f"{failed / popularity:.6f}",
                "median_lifetime_steps": (
                    f"{statistics.median(lifetimes):.3f}" if lifetimes else ""
                ),
                "mean_lifetime_steps": f"{statistics.mean(lifetimes):.3f}" if lifetimes else "",
                "max_lifetime_steps": max(lifetimes) if lifetimes else "",
                "n_unique_composite_rules": len(
                    {obs.composite_rule_smarts for obs in pg_rules}
                ),
                "n_unique_composite_rule_families": len(family_counts),
                "top_composite_rule_families": ",".join(
                    family for family, _count in family_counts.most_common(5)
                ),
                "top_first_reactions_after_source": " || ".join(
                    reaction for reaction, _count in first_reactions.most_common(3)
                ),
                "top_last_reactions_before_deprotection": " || ".join(
                    reaction for reaction, _count in last_reactions.most_common(3)
                ),
                "top_protected_functional_groups": ",".join(
                    name for name, _count in functional_groups.most_common(5)
                ),
                "n_multicenter_deprotection_events": sum(
                    1
                    for event in pg_events
                    if event.multicenter_status
                    in {"deprotection_plus_other", "deprotective_combo"}
                ),
                "n_deprotective_combo_events": sum(
                    1
                    for event in pg_events
                    if event.multicenter_status == "deprotective_combo"
                ),
                "n_selective_deprotections": selective,
                "selective_deprotection_fraction": f"{selective / popularity:.6f}",
            }
        )
    return sorted(rows, key=lambda row: (-int(row["popularity"]), row["pg_type"]))


def summarize_rule_families(
    events: list[DeprotectionEvent],
    interval_rules: list[IntervalRuleObservation],
) -> list[dict[str, Any]]:
    event_by_id = {event.event_id: event for event in events}
    grouped: dict[tuple[str, str], list[IntervalRuleObservation]] = defaultdict(list)
    for observation in interval_rules:
        grouped[(observation.pg_type, observation.composite_rule_family_id)].append(
            observation
        )

    rows = []
    for (pg_type, family_id), observations in grouped.items():
        route_ids = {obs.route_id for obs in observations}
        event_ids = {obs.event_id for obs in observations}
        targets = {
            event_by_id[event_id].target_smiles
            for event_id in event_ids
            if event_id in event_by_id
        }
        representative = observations[0]
        rows.append(
            {
                "pg_type": pg_type,
                "composite_rule_family_id": family_id,
                "family_popularity": len(observations),
                "route_count": len(route_ids),
                "target_count": len(targets),
                "representative_composite_rule_smarts": (
                    representative.composite_rule_smarts
                ),
                "family_size_rules": len(
                    {obs.composite_rule_smarts for obs in observations}
                ),
                "median_pairwise_similarity": "",
                "min_pairwise_similarity": "",
                "max_pairwise_similarity": "",
                "example_route_ids": ",".join(
                    sorted(route_ids, key=reference_sort_key)[:10]
                ),
                "example_event_ids": ",".join(sorted(event_ids)[:10]),
                "example_target_smiles": ",".join(sorted(targets)[:5]),
                "interpretation_label": f"{pg_type}->{family_id}",
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -int(row["family_popularity"]),
            row["pg_type"],
            row["composite_rule_family_id"],
        ),
    )


def network_edges(
    rule_family_rows: list[dict[str, Any]],
    events: list[DeprotectionEvent],
) -> list[dict[str, Any]]:
    lifetime_by_pg = defaultdict(list)
    for event in events:
        lifetime_by_pg[event.pg_type].append(event.lifetime_steps)
    rows = []
    for row in rule_family_rows:
        lifetimes = lifetime_by_pg.get(row["pg_type"], [])
        rows.append(
            {
                "source_pg_type": row["pg_type"],
                "target_rule_family_id": row["composite_rule_family_id"],
                "edge_weight": row["family_popularity"],
                "route_count": row["route_count"],
                "target_count": row["target_count"],
                "median_lifetime_steps": (
                    f"{statistics.median(lifetimes):.3f}" if lifetimes else ""
                ),
                "representative_rule": row["representative_composite_rule_smarts"],
                "interpretation_label": row["interpretation_label"],
            }
        )
    return rows


def analyze_protection_in_routes(
    routes_json: Any,
    composite_rule_index: dict[str, CompositeRuleFamily] | None = None,
    config: ProtectionAnalysisConfig | None = None,
    *,
    protection_rules: dict[str, ProtectionRule] | None = None,
    limit: int | None = None,
    route_ids: set[str] | None = None,
    collect_interval_rules: bool | None = None,
    progress_interval: int = 0,
) -> ProtectionAnalysisResult:
    from alchems.protection.chython_rules import load_chython_protection_rules

    config = config or ProtectionAnalysisConfig()
    protection_rules = protection_rules or load_chython_protection_rules()
    if collect_interval_rules is not None:
        config.collect_interval_rules = collect_interval_rules
    rule_extractor = None
    if config.collect_interval_rules:
        try:
            rule_extractor = default_rule_extractor()
        except Exception:
            rule_extractor = None

    route_stats_rows: list[dict[str, Any]] = []
    events: list[DeprotectionEvent] = []
    interval_rules: list[IntervalRuleObservation] = []
    errors: list[dict[str, Any]] = []
    debug_routes: dict[str, dict[str, Any]] = {}
    routes_seen = 0

    for _index_in_file, (route_id_raw, route) in enumerate(
        route_items(routes_json),
        start=1,
    ):
        route_id = str(route_id_raw)
        if route_ids is not None and route_id not in route_ids:
            continue
        if limit is not None and routes_seen >= limit:
            break
        routes_seen += 1
        try:
            route_events, route_interval_rules, route_index = analyze_route_protection(
                route,
                route_id,
                protection_rules,
                composite_rule_index=composite_rule_index,
                config=config,
                rule_extractor=rule_extractor,
            )
            events.extend(route_events)
            interval_rules.extend(route_interval_rules)
            route_stats_rows.append(
                route_stats_row(
                    route_id,
                    route_index,
                    route_events,
                    route_interval_rules,
                )
            )
            if config.write_debug_json and route_events:
                debug_routes[route_id] = route_index.route
        except Exception as exc:
            if not config.ignore_errors:
                raise
            errors.append(
                {
                    "route_id": route_id,
                    "stage": "analyze_route_protection",
                    "error_type": type(exc).__qualname__,
                    "message": str(exc) or traceback.format_exc(limit=1).strip(),
                }
            )
            try:
                route_index = build_route_index(route)
                route_stats_rows.append(
                    route_stats_row(route_id, route_index, [], [])
                )
            except Exception:
                pass

        if progress_interval and routes_seen % progress_interval == 0:
            print(
                "[analyze-protection] processed "
                f"{routes_seen} routes; events={len(events)}; "
                f"failures={sum(1 for event in events if event.trace_status == 'failed')}; "
                f"ambiguous={sum(1 for event in events if event.trace_status == 'ambiguous')}",
                file=sys.stderr,
                flush=True,
            )

    event_rows = [event_to_row(event) for event in events]
    interval_rule_rows = [interval_rule_to_row(obs) for obs in interval_rules]
    trace_failure_rows = [
        trace_failure_row(event)
        for event in events
        if event.trace_status in {"failed", "ambiguous"}
    ]
    group_summary_rows = summarize_groups(events, interval_rules)
    rule_family_rows = summarize_rule_families(events, interval_rules)
    network_edge_rows = network_edges(rule_family_rows, events)

    top_pg_types = [
        {"pg_type": row["pg_type"], "popularity": row["popularity"]}
        for row in group_summary_rows[:20]
    ]
    summary = {
        "dataset": "",
        "n_routes": routes_seen,
        "n_routes_with_pg": len({event.route_id for event in events}),
        "n_deprotection_events": len(events),
        "n_resolved_introduced": sum(
            1 for event in events if event.trace_status == "introduced"
        ),
        "n_resolved_stock": sum(1 for event in events if event.trace_status == "stock"),
        "n_ambiguous": sum(1 for event in events if event.trace_status == "ambiguous"),
        "n_failed": sum(1 for event in events if event.trace_status == "failed"),
        "n_multicenter_deprotections": sum(
            1
            for event in events
            if event.multicenter_status
            in {"deprotection_plus_other", "deprotective_combo"}
        ),
        "n_deprotective_combo_events": sum(
            1 for event in events if event.multicenter_status == "deprotective_combo"
        ),
        "top_pg_types": top_pg_types,
        "top_pg_rule_families": rule_family_rows[:20],
        "config": config.raw_config
        or {
            "min_composite_size": config.min_composite_size,
            "max_composite_size": config.max_composite_size,
            "similarity_threshold": config.similarity_threshold,
            "include_multicenter": config.include_multicenter,
            "querycgr_compare": config.querycgr_compare,
        },
        "software": {
            "composite_rules_commit": "",
            "chython_version": "",
            "synplanner_version": "",
        },
        "n_errors": len(errors),
        "errors": errors[:100],
        "n_chython_protection_rules": len(protection_rules),
    }
    return ProtectionAnalysisResult(
        route_stats_rows=route_stats_rows,
        event_rows=event_rows,
        interval_rule_rows=interval_rule_rows,
        group_summary_rows=group_summary_rows,
        rule_family_rows=rule_family_rows,
        network_edge_rows=network_edge_rows,
        trace_failure_rows=trace_failure_rows,
        summary=summary,
        debug_routes=debug_routes,
    )
