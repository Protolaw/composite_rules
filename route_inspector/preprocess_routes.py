from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from dataclasses import replace
from itertools import islice, permutations
from pathlib import Path
from typing import Any

from route_inspector.composite_rules.extract import (
    reaction_smiles_from_node,
    route_items,
)
from route_inspector.composite_rules.extract import (
    normalize_route_tree,
)
from route_inspector.composite_rules.unwrap import unwrap_rule_sequence
from route_inspector.io import (
    dataset_prefix_from_path,
    normalize_n_cpu,
    read_json,
    resolve_existing_path,
    setup_runtime_cache_dirs,
    stage_output_dir,
    write_json,
    write_standard_sidecars,
)
from route_inspector.protection.analysis import (
    ProtectionAnalysisConfig,
    ReactionRecord,
    RouteIndex,
    analyze_route_protection,
    build_route_index,
    detect_deprotections,
    molecule_node_smiles,
    parse_molecule,
    same_molecule,
)
from route_inspector.protection.chython_rules import (
    ProtectionRule,
    load_chython_protection_rules,
)


@dataclass(frozen=True)
class SingleCenterRule:
    rule_smarts: str
    center_atoms: frozenset[int]
    forward_change_kind: str = "unknown"
    forward_bonds_formed: tuple[tuple[int, int], ...] = ()
    forward_bonds_broken: tuple[tuple[int, int], ...] = ()
    forward_bonds_changed: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class ReactionGranularity:
    reaction_smiles: str
    multicenter_rule_smarts: str
    single_center_rules: tuple[SingleCenterRule, ...]
    center_components: tuple[frozenset[int], ...]
    skipped: bool = False

    @property
    def is_multicenter(self) -> bool:
        return len(self.single_center_rules) > 1


@dataclass(frozen=True)
class SplitPlan:
    route_id: Any
    reaction_id: str
    parent_mol_id: str
    reaction_node: dict[str, Any]
    parent_mol_node: dict[str, Any]
    original_children: tuple[dict[str, Any], ...]
    original_reaction_smiles: str
    parent_smiles: str
    extraction: ReactionGranularity
    protection_matches: tuple[Any, ...]
    protection_atom_ids: frozenset[int]


@dataclass
class RoutePreprocessResult:
    route_id: Any
    route: dict[str, Any]
    normalized: bool
    modified: bool
    multicenter_reactions: int
    protection_multicenter_reactions: int
    split_reactions: int
    protection_split_reactions: int
    changes: list[dict[str, Any]]
    unresolved_reactions: list[dict[str, Any]]
    errors: list[dict[str, Any]]


class RouteSplitError(ValueError):
    """Raised when a multicenter reaction cannot be split."""


REACTION_SMILES_CORRECTIONS: dict[str, str] = {
    (
        "[CH2:32]([CH3:31])[N:33]([CH2:34][CH3:35])[CH2:36][CH3:24]."
        "[O:20]=[S:18](=[O:21])([CH3:19])[Cl:37].[cH:5]1[cH:4]"
        "[c:3]([cH:30][cH:29][c:6]1[CH2:7][n:8]2[c:28]3[c:11]"
        "([cH:12][cH:13][c:14]([cH:27]3)[C@@H:15]([CH2:16]"
        "[OH:17])[OH:22])[cH:10][n:9]2)[O:2][CH3:1]>>[cH:4]1"
        "[cH:5][c:6]([cH:29][cH:30][c:3]1[O:2][CH3:1])[CH2:7]"
        "[n:8]2[n:9][cH:10][c:11]3[c:28]2[cH:27][c:14]([cH:13]"
        "[cH:12]3)[C@@H:15]([CH2:16][O:17][S:18](=[O:20])(=[O:21])"
        "[CH3:19])[O:22][S:23](=[O:25])(=[O:26])[CH3:24]"
    ): (
        "[O:1]=[S:2](=[O:3])([CH3:4])[Cl:5].[O:6]=[S:7](=[O:8])"
        "([CH3:9])[Cl:10].[cH:11]1[cH:12][c:13]([cH:14][cH:15]"
        "[c:16]1[CH2:17][n:18]2[c:19]3[c:20]([cH:21][cH:22]"
        "[c:23]([cH:24]3)[C@@H:25]([CH2:26][OH:27])[OH:28])"
        "[cH:29][n:30]2)[O:31][CH3:32]>>[cH:12]1[cH:11][c:16]"
        "([cH:15][cH:14][c:13]1[O:31][CH3:32])[CH2:17][n:18]2"
        "[n:30][cH:29][c:20]3[c:19]2[cH:24][c:23]([cH:22][cH:21]3)"
        "[C@@H:25]([CH2:26][O:27][S:2](=[O:3])(=[O:1])[CH3:4])"
        "[O:28][S:7](=[O:8])(=[O:6])[CH3:9]"
    ),
    (
        "[CH3:19][I:28].[CH3:25][C:26]([CH3:20])=[O:27].[Cl:11]"
        "[c:10]1[cH:9][cH:18][c:16]([c:14]([OH:15])[c:12]1"
        "[Cl:13])[OH:17]>>[Cl:13][c:12]1[c:10]([Cl:11])[cH:9]"
        "[cH:18][c:16]([c:14]1[O:15][CH3:19])[O:17][CH3:20]"
    ): (
        "[CH3:19][I:28].[CH3:20][I:21].[Cl:11][c:10]1[cH:9]"
        "[cH:18][c:16]([c:14]([OH:15])[c:12]1[Cl:13])[OH:17]>>"
        "[Cl:13][c:12]1[c:10]([Cl:11])[cH:9][cH:18][c:16]"
        "([c:14]1[O:15][CH3:19])[O:17][CH3:20]"
    ),
    (
        "[CH3:1][OH:2].[CH3:51][Si:52]([CH3:53])([CH3:54])[O:6]"
        "[CH:5]([c:11]1[n:26][c:24]([CH3:25])[s:23][cH:12]1)"
        "[C:3]#[N:55].[O:4]=[C:49]([CH3:48])[Cl:50]>>[OH:6]"
        "[CH:5]([C:3](=[O:4])[O:2][CH3:1])[c:11]1[cH:12][s:23]"
        "[c:24]([CH3:25])[n:26]1"
    ): (
        "[CH3:1][OH:2].[CH3:51][Si:52]([CH3:53])([CH3:54])[O:6]"
        "[CH:5]([c:11]1[n:26][c:24]([CH3:25])[s:23][cH:12]1)"
        "[C:3]#[N:55].[OH2:4]>>[OH:6][CH:5]([C:3](=[O:4])"
        "[O:2][CH3:1])[c:11]1[cH:12][s:23][c:24]([CH3:25])[n:26]1"
    ),
    (
        "[CH3:1][O:2][c:3]1[cH:4][c:5]2[O:29][CH2:28][C:9]"
        "([c:6]2[cH:7][cH:8]1)=[O:48].[cH:31]1[cH:60][c:59]"
        "([cH:34][cH:33][cH:32]1)[P:58]([c:57]2[cH:56][cH:55]"
        "[cH:54][cH:42][cH:41]2)([c:35]3[cH:40][cH:39][cH:38]"
        "[cH:37][cH:36]3)=[CH:10][C:11](=[O:44])[OH:47].[cH:50]1"
        "[cH:51][c:46]([cH:52][cH:53][cH:49]1)[CH3:45]>>[C:11]"
        "([CH2:10][c:9]1[c:6]2[cH:7][cH:8][c:3]([cH:4][c:5]2"
        "[o:29][cH:28]1)[O:2][CH3:1])([O:47][CH2:46][CH3:45])=[O:44]"
    ): (
        "[CH3:1][O:2][c:3]1[cH:4][c:5]2[O:29][CH2:28][C:9]"
        "([c:6]2[cH:7][cH:8]1)=[O:48].[cH:31]1[cH:60][c:59]"
        "([cH:34][cH:33][cH:32]1)[P:58]([c:57]2[cH:56][cH:55]"
        "[cH:54][cH:42][cH:41]2)([c:35]3[cH:40][cH:39][cH:38]"
        "[cH:37][cH:36]3)=[CH:10][C:11](=[O:44])[OH:47].[CH3:45]"
        "[CH2:46][OH:47]>>[C:11]([CH2:10][c:9]1[c:6]2[cH:7]"
        "[cH:8][c:3]([cH:4][c:5]2[o:29][cH:28]1)[O:2][CH3:1])"
        "([O:47][CH2:46][CH3:45])=[O:44]"
    ),
    (
        "[CH2:29]([CH2:28][n:27]1[c:31](=[O:32])[n:4]([c:5]([c:7]2"
        "[c:8]1[n:9][c:10](/[CH:11]=[CH:12]/[c:13]3[cH:24][cH:23]"
        "[c:19]([c:15]([OH:16])[cH:14]3)[OH:20])[n:25]2[CH3:26])"
        "=[O:6])[CH2:3][CH2:2][CH3:1])[CH3:30].[CH3:18][CH2:17]"
        "[I:37].[CH3:33][N:34]([CH3:22])[CH:35]=[O:36].[OH:40]"
        "[C:21]([OH:39])=[O:38]>>[cH:23]1[cH:24][c:13](/[CH:12]"
        "=[CH:11]/[c:10]2[n:9][c:8]3[c:7]([c:5](=[O:6])[n:4]"
        "([c:31](=[O:32])[n:27]3[CH2:28][CH2:29][CH3:30])[CH2:3]"
        "[CH2:2][CH3:1])[n:25]2[CH3:26])[cH:14][c:15]([c:19]1"
        "[O:20][CH2:21][CH3:22])[O:16][CH2:17][CH3:18]"
    ): (
        "[CH2:29]([CH2:28][n:27]1[c:31](=[O:32])[n:4]([c:5]([c:7]2"
        "[c:8]1[n:9][c:10](/[CH:11]=[CH:12]/[c:13]3[cH:24][cH:23]"
        "[c:19]([c:15]([OH:16])[cH:14]3)[OH:20])[n:25]2[CH3:26])"
        "=[O:6])[CH2:3][CH2:2][CH3:1])[CH3:30].[CH3:18][CH2:17]"
        "[I:37].[CH3:22][CH2:21][I:38]>>[cH:23]1[cH:24][c:13]"
        "(/[CH:12]=[CH:11]/[c:10]2[n:9][c:8]3[c:7]([c:5](=[O:6])"
        "[n:4]([c:31](=[O:32])[n:27]3[CH2:28][CH2:29][CH3:30])"
        "[CH2:3][CH2:2][CH3:1])[n:25]2[CH3:26])[cH:14][c:15]"
        "([c:19]1[O:20][CH2:21][CH3:22])[O:16][CH2:17][CH3:18]"
    ),
    (
        "[CH3:34][Si:33]([CH3:35])([CH3:36])[CH2:32][CH2:31][O:30]"
        "[C:28](=[O:29])[CH:6]([CH2:7][c:8]1[cH:9][cH:10][c:11]"
        "([NH2:12])[c:22]2[cH:23][cH:24][cH:25][cH:26][c:27]12)"
        "[NH:5][C:3]([O:2][CH3:1])=[O:4].[cH:39]1[cH:38][cH:37]"
        "[cH:43][cH:42][c:40]1[I+:41][c:13]2[c:14]([C:15](=[O:16])"
        "[O-:17])[cH:18][cH:19][cH:20][cH:21]2>>[CH3:34][Si:33]"
        "([CH3:35])([CH3:36])[CH2:32][CH2:31][O:30][C:28](=[O:29])"
        "[CH:6]([NH:5][C:3]([O:2][CH3:1])=[O:4])[CH2:7][c:8]1"
        "[c:27]2[cH:26][cH:25][cH:24][cH:23][c:22]2[c:11]([NH:12]"
        "[c:13]3[c:14]([C:15]([OH:17])=[O:16])[cH:18][cH:19][cH:20]"
        "[cH:21]3)[cH:10][cH:9]1"
    ): (
        "[CH3:34][Si:33]([CH3:35])([CH3:36])[CH2:32][CH2:31][O:30]"
        "[C:28](=[O:29])[CH:6]([CH2:7][c:8]1[cH:9][cH:10][c:11]"
        "([NH2:12])[c:22]2[cH:23][cH:24][cH:25][cH:26][c:27]12)"
        "[NH:5][C:3]([O:2][CH3:1])=[O:4].[I:41][c:13]2[c:14]"
        "([C:15](=[O:16])[OH:17])[cH:18][cH:19][cH:20][cH:21]2>>"
        "[CH3:34][Si:33]([CH3:35])([CH3:36])[CH2:32][CH2:31][O:30]"
        "[C:28](=[O:29])[CH:6]([NH:5][C:3]([O:2][CH3:1])=[O:4])"
        "[CH2:7][c:8]1[c:27]2[cH:26][cH:25][cH:24][cH:23][c:22]2"
        "[c:11]([NH:12][c:13]3[c:14]([C:15]([OH:17])=[O:16])"
        "[cH:18][cH:19][cH:20][cH:21]3)[cH:10][cH:9]1"
    ),
    (
        "[CH3:25][C:24]([c:26]1[cH:31][cH:30][c:29]2[nH:32]"
        "[c:34](=[O:35])[nH:36][c:28]2[cH:27]1)=[O:39].[CH3:33]"
        "[N:41]([CH3:40])[CH:42]=[O:43].[CH3:37][I:44]>>[O:35]"
        "=[c:34]1[n:32]([c:29]2[c:28]([cH:27][c:26]([C:24]([CH3:25])"
        "=[O:39])[cH:31][cH:30]2)[n:36]1[CH3:37])[CH3:33]"
    ): (
        "[CH3:25][C:24]([c:26]1[cH:31][cH:30][c:29]2[nH:32]"
        "[c:34](=[O:35])[nH:36][c:28]2[cH:27]1)=[O:39].[CH3:33]"
        "[I:40].[CH3:37][I:44]>>[O:35]=[c:34]1[n:32]([c:29]2"
        "[c:28]([cH:27][c:26]([C:24]([CH3:25])=[O:39])[cH:31]"
        "[cH:30]2)[n:36]1[CH3:37])[CH3:33]"
    ),
}


REACTION_SOURCE_ERROR_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "[cH:41]1[cH:42][c:43]([cH:53][cH:54][c:40]1[N:39]2[CH2:70]"
        "[CH2:69][N:68]([CH2:56][CH2:55]2)[CH3:67])[NH:44][c:5]3"
        "[n:6][cH:7][c:8](-[c:45]4[cH:46][s:47][c:48]([C:49]"
        "([NH2:50])=[O:51])[cH:52]4)[n:14]5[n:15][cH:16][n:17]"
        "[c:18]35",
        "irrelevant_building_block_in_source_reaction",
    ),
)


def source_reaction_error_reason(reaction_smiles: str) -> str | None:
    for pattern, reason in REACTION_SOURCE_ERROR_PATTERNS:
        if pattern in reaction_smiles:
            return reason
    return None


def reaction_side_bonds(
    molecules: Iterable[Any],
    center_atoms: frozenset[int],
) -> dict[tuple[int, int], str]:
    bonds: dict[tuple[int, int], str] = {}
    for molecule in molecules:
        for atom_1, atom_2, bond in molecule.bonds():
            atom_1 = int(atom_1)
            atom_2 = int(atom_2)
            if atom_1 not in center_atoms and atom_2 not in center_atoms:
                continue
            key = (min(atom_1, atom_2), max(atom_1, atom_2))
            try:
                value = str(int(bond))
            except Exception:
                value = str(bond)
            bonds[key] = value
    return bonds


def component_forward_change(
    reaction: Any,
    center_atoms: frozenset[int],
) -> dict[str, Any]:
    reactant_bonds = reaction_side_bonds(reaction.reactants, center_atoms)
    product_bonds = reaction_side_bonds(reaction.products, center_atoms)
    reactant_keys = set(reactant_bonds)
    product_keys = set(product_bonds)

    formed = tuple(sorted(product_keys - reactant_keys))
    broken = tuple(sorted(reactant_keys - product_keys))
    changed = tuple(
        sorted(
            key
            for key in reactant_keys & product_keys
            if reactant_bonds[key] != product_bonds[key]
        )
    )

    if formed and not broken:
        kind = "bond_forming"
    elif broken and not formed:
        kind = "bond_breaking"
    elif formed and broken:
        kind = "bond_forming_and_breaking"
    elif changed:
        kind = "bond_order_change"
    else:
        kind = "other"

    return {
        "forward_change_kind": kind,
        "forward_bonds_formed": formed,
        "forward_bonds_broken": broken,
        "forward_bonds_changed": changed,
    }


def reaction_molecules(reaction: Any) -> tuple[Any, ...]:
    return tuple(reaction.reactants) + tuple(reaction.products)


def atom_symbol_in_reaction(reaction: Any, atom_id: int) -> str:
    for molecule in reaction_molecules(reaction):
        if molecule.has_atom(atom_id):
            return str(molecule.atom(atom_id).atomic_symbol)
    return ""


def bond_symbols_in_reaction(
    reaction: Any,
    atom_1: int,
    atom_2: int,
) -> frozenset[str]:
    return frozenset(
        (
            atom_symbol_in_reaction(reaction, atom_1),
            atom_symbol_in_reaction(reaction, atom_2),
        )
    )


def reaction_adjacency(reaction: Any) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    for molecule in reaction_molecules(reaction):
        for atom_1, atom_2, _bond in molecule.bonds():
            atom_1 = int(atom_1)
            atom_2 = int(atom_2)
            adjacency.setdefault(atom_1, set()).add(atom_2)
            adjacency.setdefault(atom_2, set()).add(atom_1)
    return adjacency


def min_component_distance(
    reaction: Any,
    left: frozenset[int],
    right: frozenset[int],
) -> int | None:
    from collections import deque

    adjacency = reaction_adjacency(reaction)
    queue = deque((atom_id, 0) for atom_id in left)
    seen = set(left)
    while queue:
        atom_id, distance = queue.popleft()
        if atom_id in right and distance > 0:
            return distance
        for neighbor in adjacency.get(atom_id, ()):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


def rule_has_formed_bond_symbols(
    reaction: Any,
    rule: SingleCenterRule,
    symbols: frozenset[str],
) -> bool:
    return any(
        bond_symbols_in_reaction(reaction, atom_1, atom_2) == symbols
        for atom_1, atom_2 in rule.forward_bonds_formed
    )


def pure_bond_breaking_rule(rule: SingleCenterRule) -> bool:
    return (
        rule.forward_change_kind == "bond_breaking"
        and bool(rule.forward_bonds_broken)
        and not rule.forward_bonds_formed
    )


def boronate_workup_single_center(
    reaction: Any,
    rules: tuple[SingleCenterRule, ...],
) -> bool:
    """Return whether SynPlanner split a local borylation/boronate workup."""
    mixed_rules = [
        rule
        for rule in rules
        if rule.forward_bonds_formed
        and rule_has_formed_bond_symbols(reaction, rule, frozenset({"B", "C"}))
    ]
    if not mixed_rules:
        return False

    product_boron_atoms = {
        atom_id
        for atom_id in {atom for rule in mixed_rules for bond in rule.forward_bonds_formed for atom in bond}
        if atom_symbol_in_reaction(reaction, atom_id) == "B"
    }
    if not product_boron_atoms:
        return False

    for rule in rules:
        if not pure_bond_breaking_rule(rule):
            continue
        oxygen_atoms = {
            atom_id
            for atom_id in rule.center_atoms
            if atom_symbol_in_reaction(reaction, atom_id) == "O"
        }
        for product in reaction.products:
            for boron_atom in product_boron_atoms:
                if not product.has_atom(boron_atom):
                    continue
                if any(
                    product.has_atom(oxygen_atom)
                    and molecule_has_bond(product, boron_atom, oxygen_atom)
                    for oxygen_atom in oxygen_atoms
                ):
                    return True
    return False


def acetonide_c_n_cascade_single_center(
    reaction: Any,
    rules: tuple[SingleCenterRule, ...],
) -> bool:
    """Detect local acetonide/enolate C-N cascades that should not be split."""
    if not any(
        rule.forward_bonds_formed
        and rule_has_formed_bond_symbols(reaction, rule, frozenset({"C", "N"}))
        for rule in rules
    ):
        return False

    for rule in rules:
        if not pure_bond_breaking_rule(rule):
            continue
        broken_by_atom: dict[int, list[tuple[int, int]]] = {}
        for atom_1, atom_2 in rule.forward_bonds_broken:
            broken_by_atom.setdefault(atom_1, []).append((atom_1, atom_2))
            broken_by_atom.setdefault(atom_2, []).append((atom_1, atom_2))
        for atom_id, bonds in broken_by_atom.items():
            if atom_symbol_in_reaction(reaction, atom_id) != "C":
                continue
            broken_symbols = [
                bond_symbols_in_reaction(reaction, atom_1, atom_2)
                for atom_1, atom_2 in bonds
            ]
            carbon_oxygen_breaks = broken_symbols.count(frozenset({"C", "O"}))
            carbon_carbon_breaks = broken_symbols.count(frozenset({"C"}))
            if carbon_oxygen_breaks >= 2 and carbon_carbon_breaks >= 1:
                return True
    return False


def local_cascade_single_center(
    reaction: Any,
    rules: tuple[SingleCenterRule, ...],
) -> bool:
    if any(pure_bond_breaking_rule(rule) for rule in rules):
        return False
    for left_index, left in enumerate(rules):
        for right in rules[left_index + 1 :]:
            distance = min_component_distance(
                reaction,
                left.center_atoms,
                right.center_atoms,
            )
            if distance is None or distance > 1:
                return False
    return True


def local_pure_breaking_single_center(
    reaction: Any,
    rules: tuple[SingleCenterRule, ...],
) -> bool:
    if not rules or not all(pure_bond_breaking_rule(rule) for rule in rules):
        return False
    for left_index, left in enumerate(rules):
        for right in rules[left_index + 1 :]:
            distance = min_component_distance(
                reaction,
                left.center_atoms,
                right.center_atoms,
            )
            if distance is None or distance > 2:
                return False
    return True


def local_conjugated_dealkylation_single_center(
    reaction: Any,
    rules: tuple[SingleCenterRule, ...],
) -> bool:
    if not any(rule.forward_bonds_changed for rule in rules):
        return False
    has_o_alkyl_cleavage = False
    for rule in rules:
        for atom_1, atom_2 in rule.forward_bonds_broken:
            if bond_symbols_in_reaction(reaction, atom_1, atom_2) == frozenset(
                {"C", "O"}
            ):
                has_o_alkyl_cleavage = True
                break
        if has_o_alkyl_cleavage:
            break
    if not has_o_alkyl_cleavage:
        return False

    for left_index, left in enumerate(rules):
        for right in rules[left_index + 1 :]:
            distance = min_component_distance(
                reaction,
                left.center_atoms,
                right.center_atoms,
            )
            if distance is None or distance > 2:
                return False
    return True


def rule_has_no_bond_change(rule: SingleCenterRule) -> bool:
    return (
        not rule.forward_bonds_formed
        and not rule.forward_bonds_broken
        and not rule.forward_bonds_changed
    )


def spectator_state_change_single_center(
    _reaction: Any,
    rules: tuple[SingleCenterRule, ...],
) -> bool:
    """Ignore isolated protonation/charge cleanup beside one real center."""
    real_rules = [rule for rule in rules if not rule_has_no_bond_change(rule)]
    spectator_rules = [rule for rule in rules if rule_has_no_bond_change(rule)]
    return len(real_rules) == 1 and bool(spectator_rules)


def rule_breaks_carbon_halide(
    reaction: Any,
    rule: SingleCenterRule,
) -> bool:
    halogens = {"F", "Cl", "Br", "I"}
    return any(
        "C" in (symbols := bond_symbols_in_reaction(reaction, atom_1, atom_2))
        and bool(symbols & halogens)
        for atom_1, atom_2 in rule.forward_bonds_broken
    )


def molecule_path_exists(
    molecule: Any,
    start: int,
    target: int,
    *,
    ignored_bond: tuple[int, int],
) -> bool:
    from collections import deque

    ignored = (min(ignored_bond), max(ignored_bond))
    queue = deque([start])
    seen = {start}
    while queue:
        atom_id = queue.popleft()
        if atom_id == target:
            return True
        for atom_1, atom_2, _bond in molecule.bonds():
            atom_1 = int(atom_1)
            atom_2 = int(atom_2)
            if (min(atom_1, atom_2), max(atom_1, atom_2)) == ignored:
                continue
            if atom_1 == atom_id and atom_2 not in seen:
                seen.add(atom_2)
                queue.append(atom_2)
            elif atom_2 == atom_id and atom_1 not in seen:
                seen.add(atom_1)
                queue.append(atom_1)
    return False


def formed_bond_closes_ring(
    reaction: Any,
    atom_1: int,
    atom_2: int,
) -> bool:
    for product in reaction.products:
        if not product.has_atom(atom_1) or not product.has_atom(atom_2):
            continue
        if not molecule_has_bond(product, atom_1, atom_2):
            continue
        return molecule_path_exists(
            product,
            atom_1,
            atom_2,
            ignored_bond=(atom_1, atom_2),
        )
    return False


def ring_condensation_cascade_single_center(
    reaction: Any,
    rules: tuple[SingleCenterRule, ...],
) -> bool:
    """Treat compact heterocycle-forming condensations as one reaction center."""
    if not rules:
        return False
    if any(pure_bond_breaking_rule(rule) for rule in rules):
        return False
    if not all(rule.forward_bonds_formed for rule in rules):
        return False
    if any(rule_breaks_carbon_halide(reaction, rule) for rule in rules):
        return False

    formed_bonds = [
        bond for rule in rules for bond in rule.forward_bonds_formed
    ]
    if not formed_bonds:
        return False
    return all(
        formed_bond_closes_ring(reaction, atom_1, atom_2)
        for atom_1, atom_2 in formed_bonds
    )


def thp_alkene_mapping_single_center(
    reaction: Any,
    rules: tuple[SingleCenterRule, ...],
) -> bool:
    """Recognize THP/DHP ether mappings split into O-C formation plus alkene shift."""
    has_oxygen_carbon_formation = any(
        rule.forward_bonds_formed
        and any(
            bond_symbols_in_reaction(reaction, atom_1, atom_2)
            == frozenset({"C", "O"})
            for atom_1, atom_2 in rule.forward_bonds_formed
        )
        for rule in rules
    )
    has_alkene_order_change = any(
        rule.forward_change_kind == "bond_order_change"
        and rule.forward_bonds_changed
        for rule in rules
    )
    return has_oxygen_carbon_formation and has_alkene_order_change


def semantic_single_center_reason(
    reaction_smiles: str,
    extraction: ReactionGranularity,
) -> str | None:
    from chython import smiles as parse_smiles

    reaction = parse_smiles(reaction_smiles)
    rules = extraction.single_center_rules
    if spectator_state_change_single_center(reaction, rules):
        return "spectator_state_change_single_center"
    if boronate_workup_single_center(reaction, rules):
        return "boronate_workup_single_center"
    if acetonide_c_n_cascade_single_center(reaction, rules):
        return "acetonide_c_n_cascade_single_center"
    if thp_alkene_mapping_single_center(reaction, rules):
        return "thp_alkene_mapping_single_center"
    if ring_condensation_cascade_single_center(reaction, rules):
        return "ring_condensation_cascade_single_center"
    if local_conjugated_dealkylation_single_center(reaction, rules):
        return "local_conjugated_dealkylation_single_center"
    if local_pure_breaking_single_center(reaction, rules):
        return "local_pure_breaking_single_center"
    if local_cascade_single_center(reaction, rules):
        return "local_cascade_single_center"
    return None


class SynPlannerGranularityExtractor:
    """Extract multicenter and single-center SynPlanner rules for one reaction."""

    def __init__(self, config: Any):
        from synplan.chem.data.standardizing import RemoveReagentsStandardizer

        self.config = config
        self.standardizer = RemoveReagentsStandardizer()
        self.cache: dict[str, ReactionGranularity] = {}

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SynPlannerGranularityExtractor":
        from synplan.utils.config import RuleExtractionConfig

        if getattr(args, "config", None):
            config = RuleExtractionConfig.from_yaml(
                str(resolve_existing_path(args.config))
            )
        else:
            config = RuleExtractionConfig(
                min_popularity=1,
                single_product_only=True,
                environment_atom_count=getattr(args, "environment_atom_count", 1),
                multicenter_rules=True,
                include_rings=getattr(args, "include_rings", False),
                include_func_groups=False,
                keep_leaving_groups=getattr(args, "keep_leaving_groups", True),
                keep_incoming_groups=getattr(args, "keep_incoming_groups", False),
                keep_reagents=False,
                reactor_validation=getattr(args, "reactor_validation", False),
            )
        return cls(config)

    def _config_with(self, **updates: Any) -> Any:
        if hasattr(self.config, "model_copy"):
            return self.config.model_copy(update=updates)
        if hasattr(self.config, "copy"):
            return self.config.copy(update=updates)
        values = dict(getattr(self.config, "__dict__", {}))
        values.update(updates)
        return type(self.config)(**values)

    def _refresh_aromaticity(self, reaction: Any) -> Any:
        refreshed = reaction.copy()
        try:
            refreshed.kekule()
            refreshed.thiele()
        except Exception:
            return reaction
        return refreshed

    def extract(self, reaction_smiles: str) -> ReactionGranularity:
        if reaction_smiles in self.cache:
            return self.cache[reaction_smiles]

        from chython import smiles as parse_smiles
        from synplan.chem.reaction_rules.extraction import (
            _rule_to_reactor_smarts,
            create_rule,
        )

        reaction = parse_smiles(reaction_smiles)
        if getattr(self.config, "ignore_stereo", True):
            reaction = reaction.copy()
            reaction.clean_stereo()
        reaction = self._refresh_aromaticity(reaction)
        standardized = self.standardizer(reaction)

        if getattr(self.config, "single_product_only", True) and (
            len(standardized.products) != 1
        ):
            result = ReactionGranularity(
                reaction_smiles=reaction_smiles,
                multicenter_rule_smarts="",
                single_center_rules=(),
                center_components=(),
                skipped=True,
            )
            self.cache[reaction_smiles] = result
            return result

        cgr = ~standardized
        center_components = tuple(
            frozenset(int(atom_id) for atom_id in component)
            for component in islice(cgr.centers_list, 15)
        )
        skip_full_validation = len(center_components) > 1
        multicenter_config = self._config_with(multicenter_rules=True)
        single_center_config = self._config_with(multicenter_rules=False)

        multicenter_rule = create_rule(multicenter_config, standardized)
        multicenter_rule_smarts = _rule_to_reactor_smarts(multicenter_rule)

        seen_cgrs: dict[Any, SingleCenterRule] = {}
        for component in center_components:
            rule = create_rule(
                single_center_config,
                standardized,
                _restrict_center_atoms=set(component),
                _skip_full_reaction_validation=skip_full_validation,
            )
            rule_cgr = ~rule
            if rule_cgr in seen_cgrs:
                continue
            seen_cgrs[rule_cgr] = SingleCenterRule(
                rule_smarts=_rule_to_reactor_smarts(rule),
                center_atoms=component,
                **component_forward_change(standardized, component),
            )

        result = ReactionGranularity(
            reaction_smiles=reaction_smiles,
            multicenter_rule_smarts=multicenter_rule_smarts,
            single_center_rules=tuple(seen_cgrs.values()),
            center_components=center_components,
        )
        self.cache[reaction_smiles] = result
        return result


def same_molecule_smiles(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right:
        return False
    try:
        return same_molecule(parse_molecule(left), parse_molecule(right))
    except Exception:
        return False


def reaction_reactants_smiles(reaction_smiles: str) -> list[str]:
    reactants, _product = reaction_smiles.split(">>", 1)
    return [reactant for reactant in reactants.split(".") if reactant]


def sync_reaction_children_to_reactants(
    reaction_node: dict[str, Any],
    reaction_smiles: str,
) -> None:
    reactants = reaction_reactants_smiles(reaction_smiles)
    matched_reactants: set[int] = set()
    synced_children: list[dict[str, Any]] = []

    for child in reaction_node.get("children", []) or []:
        if not isinstance(child, dict) or child.get("type") != "mol":
            synced_children.append(child)
            continue

        child_smiles = molecule_node_smiles(child)
        match_index = next(
            (
                index
                for index, reactant in enumerate(reactants)
                if index not in matched_reactants
                and same_molecule_smiles(child_smiles, reactant)
            ),
            None,
        )
        if match_index is None:
            if child.get("children"):
                synced_children.append(child)
            continue
        matched_reactants.add(match_index)
        child["smiles"] = reactants[match_index]
        metadata = child.setdefault("metadata", {})
        metadata["mapped_smiles"] = reactants[match_index]
        synced_children.append(child)

    for index, reactant in enumerate(reactants):
        if index in matched_reactants:
            continue
        synced_children.append(
            {
                "type": "mol",
                "smiles": reactant,
                "metadata": {"mapped_smiles": reactant},
                "in_stock": True,
                "children": [],
            }
        )
    reaction_node["children"] = synced_children


def apply_reaction_smiles_corrections(
    route: dict[str, Any],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> None:
        if node.get("type") == "reaction":
            try:
                original_smiles = reaction_smiles_from_node(node)
            except Exception:
                original_smiles = ""
            corrected_smiles = REACTION_SMILES_CORRECTIONS.get(original_smiles)
            if corrected_smiles is not None:
                node["smiles"] = corrected_smiles
                metadata = node.setdefault("metadata", {})
                metadata["smiles"] = corrected_smiles
                sync_reaction_children_to_reactants(node, corrected_smiles)
                changes.append(
                    {
                        "reaction_id": "",
                        "parent_mol_id": "",
                        "original_reaction_smiles": original_smiles,
                        "corrected_reaction_smiles": corrected_smiles,
                        "split_method": "source_reaction_correction",
                    }
                )
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                visit(child)

    visit(route)
    return changes


def protection_atom_ids_from_matches(matches: Iterable[Any]) -> frozenset[int]:
    atom_ids: set[int] = set()
    for match in matches:
        atom_ids.update(
            int(atom_id) for atom_id in getattr(match, "protected_atom_ids", ())
        )
        for _query_atom, route_atom in getattr(match, "raw_mapping", ()):
            atom_ids.add(int(route_atom))
    return frozenset(atom_ids)


NON_PROTECTION_SPLIT_PRIORITY = {
    "bond_breaking": 0,
    "bond_forming_and_breaking": 1,
    "bond_order_change": 1,
    "other": 1,
    "unknown": 1,
    "bond_forming": 2,
}


def non_protection_rule_groups(
    rules: tuple[SingleCenterRule, ...],
) -> tuple[tuple[SingleCenterRule, ...], ...]:
    grouped: dict[int, list[SingleCenterRule]] = {}
    for rule in rules:
        priority = NON_PROTECTION_SPLIT_PRIORITY.get(rule.forward_change_kind, 1)
        grouped.setdefault(priority, []).append(rule)
    return tuple(tuple(grouped[priority]) for priority in sorted(grouped))


def grouped_rule_permutations(
    groups: tuple[tuple[SingleCenterRule, ...], ...],
) -> Iterable[tuple[SingleCenterRule, ...]]:
    if not groups:
        yield ()
        return
    first, *rest = groups
    for left in permutations(first):
        for right in grouped_rule_permutations(tuple(rest)):
            yield tuple(left + right)


def split_candidate_rule_orders(
    rules: tuple[SingleCenterRule, ...],
    protection_atom_ids: frozenset[int],
    *,
    deprotection_first: bool,
) -> Iterable[tuple[SingleCenterRule, ...]]:
    if protection_atom_ids:
        protection_rules = tuple(
            rule for rule in rules if rule.center_atoms & protection_atom_ids
        )
        other_rules = tuple(
            rule for rule in rules if not (rule.center_atoms & protection_atom_ids)
        )
        if protection_rules:
            groups = (
                (protection_rules, other_rules)
                if deprotection_first
                else (other_rules, protection_rules)
            )
        else:
            groups = non_protection_rule_groups(rules)
    else:
        groups = non_protection_rule_groups(rules)
    groups = tuple(group for group in groups if group)

    primary = tuple(rule for group in groups for rule in group)
    if not primary:
        return
    yielded = {tuple(rule.rule_smarts for rule in primary)}
    yield primary

    if len(rules) > 6:
        return

    for candidate in grouped_rule_permutations(groups):
        key = tuple(rule.rule_smarts for rule in candidate)
        if key in yielded:
            continue
        yielded.add(key)
        yield candidate


def molecule_leaf_slots(
    node: dict[str, Any],
) -> list[tuple[list[dict[str, Any]], int, dict[str, Any]]]:
    slots: list[tuple[list[dict[str, Any]], int, dict[str, Any]]] = []

    def visit(current: dict[str, Any]) -> None:
        children = current.get("children", []) or []
        if current.get("type") == "mol":
            reaction_children = [
                child
                for child in children
                if isinstance(child, dict) and child.get("type") == "reaction"
            ]
            if not reaction_children:
                return
        for index, child in enumerate(children):
            if not isinstance(child, dict):
                continue
            if child.get("type") == "mol":
                child_reactions = [
                    grandchild
                    for grandchild in child.get("children", []) or []
                    if isinstance(grandchild, dict)
                    and grandchild.get("type") == "reaction"
                ]
                if not child_reactions:
                    slots.append((children, index, child))
                else:
                    visit(child)
            else:
                visit(child)

    visit(node)
    return slots


def collect_reaction_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    reactions: list[dict[str, Any]] = []

    def visit(current: dict[str, Any]) -> None:
        if current.get("type") == "reaction":
            reactions.append(current)
        for child in current.get("children", []) or []:
            if isinstance(child, dict):
                visit(child)

    visit(node)
    return reactions


def reattach_original_children(
    generated_root: dict[str, Any],
    original_children: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    leaf_slots = molecule_leaf_slots(generated_root)
    used_original_indexes: set[int] = set()

    for parent_children, leaf_index, leaf_node in leaf_slots:
        leaf_smiles = molecule_node_smiles(leaf_node)
        matched_index = None
        for original_index, original_child in enumerate(original_children):
            if original_index in used_original_indexes:
                continue
            if same_molecule_smiles(leaf_smiles, molecule_node_smiles(original_child)):
                matched_index = original_index
                break
        if matched_index is None:
            raise RouteSplitError(
                f"generated leaf {leaf_smiles!r} does not match any original reactant"
            )
        used_original_indexes.add(matched_index)
        parent_children[leaf_index] = original_children[matched_index]

    if len(used_original_indexes) != len(original_children):
        unmatched = [
            molecule_node_smiles(child)
            for index, child in enumerate(original_children)
            if index not in used_original_indexes
        ]
        raise RouteSplitError(
            "split route did not regenerate all original reactants: "
            + ", ".join(unmatched)
        )
    return generated_root


def unwrap_split_route(
    plan: SplitPlan,
    ordered_rules: tuple[SingleCenterRule, ...],
    *,
    unwrapper: Callable[..., Any],
) -> dict[str, Any]:
    rule_smarts = [rule.rule_smarts for rule in ordered_rules]
    raw_result = unwrapper(
        plan.parent_smiles,
        rule_smarts,
        route_id=0,
        rule_key_prefix="preprocess_split",
        mark_leaves_in_stock=False,
    )
    if hasattr(raw_result, "routes_json"):
        generated_root = raw_result.routes_json[0]
    elif isinstance(raw_result, dict) and 0 in raw_result:
        generated_root = raw_result[0]
    elif isinstance(raw_result, dict):
        generated_root = raw_result
    else:
        raise TypeError(f"unsupported unwrap result: {type(raw_result)!r}")
    return reattach_original_children(generated_root, plan.original_children)


def original_reactants_smiles(reaction_smiles: str) -> str:
    if ">>" not in reaction_smiles:
        raise RouteSplitError("reaction SMILES does not contain >>")
    reactants, _products = reaction_smiles.split(">>", 1)
    return reactants


def molecule_has_bond(molecule: Any, atom_1: int, atom_2: int) -> bool:
    try:
        return bool(molecule.has_bond(atom_1, atom_2))
    except Exception:
        try:
            molecule.bond(atom_1, atom_2)
        except Exception:
            return False
        return True


def source_molecule_for_broken_rule(reaction: Any, rule: SingleCenterRule) -> Any:
    for molecule in reaction.reactants:
        if all(
            molecule_has_bond(molecule, atom_1, atom_2)
            for atom_1, atom_2 in rule.forward_bonds_broken
        ):
            return molecule
    raise RouteSplitError(
        f"could not find reactant source for broken center {sorted(rule.center_atoms)}"
    )


def molecule_atom_ids(molecule: Any) -> set[int]:
    return {int(atom_id) for atom_id in molecule}


def atoms_to_restore_from_source(
    molecule: Any,
    source_molecule: Any,
    rule: SingleCenterRule,
) -> tuple[set[int], set[int]]:
    product_atoms = molecule_atom_ids(molecule)
    missing_seeds: set[int] = set()
    existing_bridge_atoms: set[int] = set()

    for atom_1, atom_2 in rule.forward_bonds_broken:
        atom_1_present = molecule.has_atom(atom_1)
        atom_2_present = molecule.has_atom(atom_2)
        if atom_1_present and not atom_2_present:
            existing_bridge_atoms.add(atom_1)
            missing_seeds.add(atom_2)
        elif atom_2_present and not atom_1_present:
            existing_bridge_atoms.add(atom_2)
            missing_seeds.add(atom_1)
        elif atom_1_present and atom_2_present:
            existing_bridge_atoms.update((atom_1, atom_2))
        else:
            missing_seeds.update((atom_1, atom_2))

    atoms_to_restore = set(missing_seeds)
    adjacency: dict[int, set[int]] = {}
    for atom_1, atom_2, _bond in source_molecule.bonds():
        atom_1 = int(atom_1)
        atom_2 = int(atom_2)
        adjacency.setdefault(atom_1, set()).add(atom_2)
        adjacency.setdefault(atom_2, set()).add(atom_1)

    stack = list(missing_seeds)
    while stack:
        atom_id = stack.pop()
        for neighbor in adjacency.get(atom_id, ()):
            if neighbor in product_atoms:
                existing_bridge_atoms.add(neighbor)
                continue
            if neighbor in atoms_to_restore:
                continue
            atoms_to_restore.add(neighbor)
            stack.append(neighbor)

    atoms_to_update = existing_bridge_atoms | (rule.center_atoms & product_atoms)
    return atoms_to_restore, atoms_to_update


ATOM_MAP_PATTERN = re.compile(r"(\[[^\]]*?:)(\d+)([^\]]*\])")


def remap_smiles_atom_ids(smiles_text: str, atom_id_map: dict[int, int]) -> str:
    if not atom_id_map:
        return smiles_text

    def replace(match: re.Match[str]) -> str:
        atom_id = int(match.group(2))
        replacement = atom_id_map.get(atom_id, atom_id)
        return f"{match.group(1)}{replacement}{match.group(3)}"

    return ATOM_MAP_PATTERN.sub(replace, smiles_text)


def remap_route_node_atom_ids(node: dict[str, Any], atom_id_map: dict[int, int]) -> None:
    if not atom_id_map:
        return
    for key in ("smiles", "mapped_smiles"):
        value = node.get(key)
        if isinstance(value, str):
            node[key] = remap_smiles_atom_ids(value, atom_id_map)

    metadata = node.get("metadata")
    if isinstance(metadata, dict):
        for key in ("smiles", "mapped_smiles", "rsmi"):
            value = metadata.get(key)
            if isinstance(value, str):
                metadata[key] = remap_smiles_atom_ids(value, atom_id_map)

    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            remap_route_node_atom_ids(child, atom_id_map)


def molecule_adjacency(molecule: Any) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    for atom_1, atom_2, _bond in molecule.bonds():
        atom_1 = int(atom_1)
        atom_2 = int(atom_2)
        adjacency.setdefault(atom_1, set()).add(atom_2)
        adjacency.setdefault(atom_2, set()).add(atom_1)
    return adjacency


def atom_symbol_in_molecule(molecule: Any, atom_id: int) -> str:
    if not molecule.has_atom(atom_id):
        return ""
    return str(molecule.atom(atom_id).atomic_symbol)


def shortest_atom_path(
    adjacency: dict[int, set[int]],
    start: int,
    target: int,
    *,
    blocked: set[int],
) -> list[int] | None:
    from collections import deque

    queue = deque([(start, [start])])
    seen = {start} | set(blocked)
    while queue:
        atom_id, path = queue.popleft()
        if atom_id == target:
            return path
        for neighbor in adjacency.get(atom_id, ()):
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    return None


def restored_o_cycle_atom_id_reassignment(
    molecule: Any,
    source_molecule: Any,
    rule: SingleCenterRule,
) -> dict[int, int]:
    """Realign fresh map IDs for restored O-cycles with equivalent orientations.

    Some PaRoutes reactions leave protecting-ring atoms unmapped. During route
    normalization those atoms receive arbitrary fresh IDs, and restoring the ring
    from the source reactant can preserve an equivalent but unintuitive mapping.
    For N-attached saturated O-cycles, flip the restored cycle maps to the
    orientation that keeps the heteroatom attached to the O-adjacent carbon.
    """
    if not pure_bond_breaking_rule(rule):
        return {}

    product_atoms = molecule_atom_ids(molecule)
    atoms_to_restore, _atoms_to_update = atoms_to_restore_from_source(
        molecule,
        source_molecule,
        rule,
    )
    adjacency = molecule_adjacency(source_molecule)
    broken_neighbors: dict[int, set[int]] = {}
    for atom_1, atom_2 in rule.forward_bonds_broken:
        broken_neighbors.setdefault(atom_1, set()).add(atom_2)
        broken_neighbors.setdefault(atom_2, set()).add(atom_1)

    for anchor, neighbors in broken_neighbors.items():
        if anchor in product_atoms or anchor not in atoms_to_restore:
            continue
        if atom_symbol_in_molecule(source_molecule, anchor) != "C":
            continue

        bridge_atoms = [atom_id for atom_id in neighbors if atom_id in product_atoms]
        if not bridge_atoms:
            continue
        if not any(
            atom_symbol_in_molecule(source_molecule, atom_id) == "N"
            for atom_id in bridge_atoms
        ):
            continue

        missing_neighbors = [
            atom_id for atom_id in neighbors if atom_id in atoms_to_restore
        ]
        oxygen_neighbors = [
            atom_id
            for atom_id in missing_neighbors
            if atom_symbol_in_molecule(source_molecule, atom_id) == "O"
        ]
        carbon_neighbors = [
            atom_id
            for atom_id in missing_neighbors
            if atom_symbol_in_molecule(source_molecule, atom_id) == "C"
        ]
        for oxygen_atom in oxygen_neighbors:
            alternate_anchors = [
                atom_id
                for atom_id in adjacency.get(oxygen_atom, ())
                if atom_id in atoms_to_restore
                and atom_id != anchor
                and atom_symbol_in_molecule(source_molecule, atom_id) == "C"
            ]
            for alternate_anchor in alternate_anchors:
                for carbon_neighbor in carbon_neighbors:
                    path = shortest_atom_path(
                        adjacency,
                        alternate_anchor,
                        carbon_neighbor,
                        blocked={anchor, oxygen_atom} | product_atoms,
                    )
                    if path is None or len(path) < 3:
                        continue
                    replacement_path = [anchor] + list(reversed(path[1:]))
                    atom_id_map = {anchor: alternate_anchor}
                    atom_id_map.update(
                        {
                            old_atom: new_atom
                            for old_atom, new_atom in zip(path, replacement_path)
                        }
                    )
                    if set(atom_id_map) != set(atom_id_map.values()):
                        continue
                    return {
                        old_atom: new_atom
                        for old_atom, new_atom in atom_id_map.items()
                        if old_atom != new_atom
                    }
    return {}


def restore_broken_rule_on_product(
    molecule: Any,
    source_molecule: Any,
    rule: SingleCenterRule,
) -> Any:
    restored = molecule.copy()
    atoms_to_restore, atoms_to_update = atoms_to_restore_from_source(
        molecule,
        source_molecule,
        rule,
    )

    for atom_id in sorted(atoms_to_restore | atoms_to_update):
        try:
            source_atom = source_molecule.atom(atom_id)
        except Exception as exc:
            raise RouteSplitError(
                f"source molecule is missing atom {atom_id} for mapped split"
            ) from exc
        if not restored.has_atom(atom_id):
            restored.add_atom(
                source_atom.copy(),
                atom_id,
                charge=getattr(source_atom, "charge", 0),
                is_radical=getattr(source_atom, "is_radical", False),
                _skip_calculation=True,
            )
        else:
            restored_atom = restored.atom(atom_id)
            restored_atom.charge = getattr(source_atom, "charge", 0)
            restored_atom.is_radical = getattr(source_atom, "is_radical", False)

    for atom_1, atom_2, source_bond in source_molecule.bonds():
        atom_1 = int(atom_1)
        atom_2 = int(atom_2)
        if atom_1 not in atoms_to_restore and atom_2 not in atoms_to_restore:
            continue
        if not restored.has_atom(atom_1) or not restored.has_atom(atom_2):
            continue
        if molecule_has_bond(restored, atom_1, atom_2):
            continue
        restored.add_bond(
            atom_1,
            atom_2,
            int(source_bond),
            _skip_calculation=True,
        )

    try:
        restored.fix_structure()
    except Exception:
        pass
    try:
        restored.fix_stereo()
    except Exception:
        pass
    return restored


def split_metadata(
    plan: SplitPlan,
    *,
    step_index: int,
    step_count: int,
    rule_smarts: str,
    forward_change_kind: str,
) -> dict[str, Any]:
    return {
        "route_preprocessing_split": {
            "route_id": str(plan.route_id),
            "original_reaction_id": plan.reaction_id,
            "original_reaction_smiles": plan.original_reaction_smiles,
            "split_step": step_index,
            "split_steps": step_count,
            "rule_smarts": rule_smarts,
            "forward_change_kind": forward_change_kind,
            "split_method": "mapped_intermediate",
        }
    }


def rule_detail(rule: SingleCenterRule) -> dict[str, Any]:
    return {
        "rule_smarts": rule.rule_smarts,
        "center_atoms": sorted(rule.center_atoms),
        "forward_change_kind": rule.forward_change_kind,
        "forward_bonds_formed": [list(bond) for bond in rule.forward_bonds_formed],
        "forward_bonds_broken": [list(bond) for bond in rule.forward_bonds_broken],
        "forward_bonds_changed": [list(bond) for bond in rule.forward_bonds_changed],
    }


def split_reaction_node_via_mapped_intermediates(
    plan: SplitPlan,
    *,
    restore_rules: tuple[SingleCenterRule, ...],
    remaining_rules: tuple[SingleCenterRule, ...],
) -> dict[str, Any]:
    if not restore_rules:
        raise RouteSplitError("mapped fallback has no bond-breaking rule to restore")
    if not remaining_rules and len(plan.original_children) != 1:
        raise RouteSplitError(
            "mapped fallback without a final reaction requires one original reactant"
        )

    from chython import smiles as parse_smiles

    reaction = parse_smiles(plan.original_reaction_smiles)
    if len(reaction.products) != 1:
        raise RouteSplitError("mapped fallback requires one reaction product")

    original_reactants = original_reactants_smiles(plan.original_reaction_smiles)
    active_molecule = reaction.products[0]
    active_smiles = format(active_molecule, "m")
    generated_reaction_smiles: list[str] = []
    atom_id_reassignments: list[dict[int, int]] = []
    ordered_rules = restore_rules + remaining_rules
    step_count = len(restore_rules) + int(bool(remaining_rules))
    first_reaction_node: dict[str, Any] | None = None
    current_mol_node: dict[str, Any] | None = None
    last_child_holder: list[dict[str, Any]] | None = None

    def original_children_with_atom_id_reassignments() -> list[dict[str, Any]]:
        children = [copy.deepcopy(child) for child in plan.original_children]
        for atom_id_map in atom_id_reassignments:
            for child in children:
                remap_route_node_atom_ids(child, atom_id_map)
        return children

    for step_index, rule in enumerate(restore_rules, start=1):
        source_molecule = source_molecule_for_broken_rule(reaction, rule)
        intermediate_molecule = restore_broken_rule_on_product(
            active_molecule,
            source_molecule,
            rule,
        )
        intermediate_smiles = format(intermediate_molecule, "m")
        atom_id_map = restored_o_cycle_atom_id_reassignment(
            active_molecule,
            source_molecule,
            rule,
        )
        if atom_id_map:
            intermediate_smiles = remap_smiles_atom_ids(
                intermediate_smiles,
                atom_id_map,
            )
            intermediate_molecule = parse_smiles(intermediate_smiles)
            original_reactants = remap_smiles_atom_ids(
                original_reactants,
                atom_id_map,
            )
            atom_id_reassignments.append(atom_id_map)
        if same_molecule_smiles(intermediate_smiles, active_smiles):
            raise RouteSplitError("mapped fallback did not change the active molecule")

        child_mol_node = {
            "type": "mol",
            "smiles": intermediate_smiles,
            "mapped_smiles": intermediate_smiles,
            "in_stock": False,
            "children": [],
        }
        reaction_smiles = f"{intermediate_smiles}>>{active_smiles}"
        reaction_node = {
            "type": "reaction",
            "smiles": reaction_smiles,
            "metadata": {
                "smiles": reaction_smiles,
                **split_metadata(
                    plan,
                    step_index=step_index,
                    step_count=step_count,
                    rule_smarts=rule.rule_smarts,
                    forward_change_kind=rule.forward_change_kind,
                ),
            },
            "children": [child_mol_node],
        }
        generated_reaction_smiles.append(reaction_smiles)

        if current_mol_node is None:
            first_reaction_node = reaction_node
        else:
            current_mol_node["children"] = [reaction_node]
        current_mol_node = child_mol_node
        last_child_holder = reaction_node["children"]
        active_molecule = intermediate_molecule
        active_smiles = intermediate_smiles

    if first_reaction_node is None or current_mol_node is None:
        raise RouteSplitError("mapped fallback did not create a split route")

    if remaining_rules:
        rule_smarts = "$".join(rule.rule_smarts for rule in remaining_rules)
        forward_change_kind = "+".join(
            rule.forward_change_kind for rule in remaining_rules
        )
        reaction_smiles = f"{original_reactants}>>{active_smiles}"
        final_reaction_node = {
            "type": "reaction",
            "smiles": reaction_smiles,
            "metadata": {
                "smiles": reaction_smiles,
                **split_metadata(
                    plan,
                    step_index=len(restore_rules) + 1,
                    step_count=step_count,
                    rule_smarts=rule_smarts,
                    forward_change_kind=forward_change_kind,
                ),
            },
            "children": original_children_with_atom_id_reassignments(),
        }
        generated_reaction_smiles.append(reaction_smiles)
        current_mol_node["children"] = [final_reaction_node]
    else:
        assert last_child_holder is not None
        original_child = original_children_with_atom_id_reassignments()[0]
        if not same_molecule_smiles(active_smiles, molecule_node_smiles(original_child)):
            raise RouteSplitError(
                "mapped fallback final intermediate does not match original reactant"
            )
        last_child_holder[0] = original_child

    replace_reaction_node(
        plan.parent_mol_node,
        plan.reaction_node,
        first_reaction_node,
    )
    return {
        "reaction_id": plan.reaction_id,
        "parent_mol_id": plan.parent_mol_id,
        "original_reaction_smiles": plan.original_reaction_smiles,
        "multicenter_rule_smarts": plan.extraction.multicenter_rule_smarts,
        "single_center_rules": [rule.rule_smarts for rule in ordered_rules],
        "single_center_rule_details": [rule_detail(rule) for rule in ordered_rules],
        "single_center_count": len(ordered_rules),
        "protection_related": bool(plan.protection_matches),
        "protection_match_count": len(plan.protection_matches),
        "protection_atom_ids": sorted(plan.protection_atom_ids),
        "generated_reaction_smiles": generated_reaction_smiles,
        "split_method": "mapped_intermediate",
    }


def disconnect_formed_rule_from_product(
    molecule: Any,
    source_molecule: Any,
    rule: SingleCenterRule,
    substrate_molecule: Any | None = None,
) -> Any:
    disconnected = molecule.copy()
    source_atoms = molecule_atom_ids(source_molecule)
    product_atoms = molecule_atom_ids(molecule)
    substrate_atoms: set[int] = set()
    source_product_atoms: set[int] = set()

    for atom_1, atom_2 in rule.forward_bonds_formed:
        atom_1_in_source = atom_1 in source_atoms
        atom_2_in_source = atom_2 in source_atoms
        if atom_1_in_source and atom_2_in_source:
            raise RouteSplitError(
                "formed-bond fallback cannot disconnect an intrareagent bond"
            )
        if atom_1_in_source:
            source_product_atoms.add(atom_1)
            substrate_atoms.add(atom_2)
        elif atom_2_in_source:
            source_product_atoms.add(atom_2)
            substrate_atoms.add(atom_1)
        else:
            raise RouteSplitError(
                "formed-bond fallback could not map formed bond to source reagent"
            )

    atoms_to_delete = (source_atoms & product_atoms) - substrate_atoms
    if not atoms_to_delete:
        atoms_to_delete = source_product_atoms & product_atoms
    if not atoms_to_delete:
        raise RouteSplitError("formed-bond fallback did not remove reagent atoms")

    for atom_id in sorted(atoms_to_delete, reverse=True):
        disconnected.delete_atom(atom_id, _skip_calculation=True)

    if substrate_molecule is not None:
        for atom_id in sorted(substrate_atoms):
            if not disconnected.has_atom(atom_id) or not substrate_molecule.has_atom(
                atom_id
            ):
                continue
            neighbor_bonds: list[tuple[int, int]] = []
            for atom_1, atom_2, bond in list(disconnected.bonds()):
                atom_1 = int(atom_1)
                atom_2 = int(atom_2)
                if atom_1 == atom_id:
                    neighbor_bonds.append((atom_2, int(bond)))
                elif atom_2 == atom_id:
                    neighbor_bonds.append((atom_1, int(bond)))

            disconnected.delete_atom(atom_id, _skip_calculation=True)
            source_atom = substrate_molecule.atom(atom_id)
            disconnected.add_atom(
                source_atom.copy(),
                atom_id,
                charge=getattr(source_atom, "charge", 0),
                is_radical=getattr(source_atom, "is_radical", False),
                _skip_calculation=True,
            )
            for neighbor, bond_order in neighbor_bonds:
                if disconnected.has_atom(neighbor):
                    disconnected.add_bond(
                        atom_id,
                        neighbor,
                        bond_order,
                        _skip_calculation=True,
                    )

    try:
        disconnected.fix_structure()
    except Exception:
        pass
    try:
        disconnected.fix_stereo()
    except Exception:
        pass
    return disconnected


def substrate_molecule_for_formed_rule(
    reaction: Any,
    source_molecule: Any,
    rule: SingleCenterRule,
) -> Any | None:
    source_atoms = molecule_atom_ids(source_molecule)
    substrate_atoms = {
        atom
        for bond in rule.forward_bonds_formed
        for atom in bond
        if atom not in source_atoms
    }
    if not substrate_atoms:
        return None
    for molecule in reaction.reactants:
        if molecule is source_molecule:
            continue
        if all(molecule.has_atom(atom_id) for atom_id in substrate_atoms):
            return molecule
    return None


def source_molecules_for_broken_rule(
    reaction: Any,
    rule: SingleCenterRule,
) -> tuple[tuple[Any, SingleCenterRule], ...]:
    sources: list[tuple[Any, SingleCenterRule]] = []
    covered_bonds: set[tuple[int, int]] = set()
    for molecule in reaction.reactants:
        present_bonds = tuple(
            (atom_1, atom_2)
            for atom_1, atom_2 in rule.forward_bonds_broken
            if molecule_has_bond(molecule, atom_1, atom_2)
        )
        if not present_bonds:
            continue
        sources.append(
            (
                molecule,
                SingleCenterRule(
                    rule_smarts=rule.rule_smarts,
                    center_atoms=rule.center_atoms & molecule_atom_ids(molecule),
                    forward_change_kind=rule.forward_change_kind,
                    forward_bonds_formed=rule.forward_bonds_formed,
                    forward_bonds_broken=present_bonds,
                    forward_bonds_changed=rule.forward_bonds_changed,
                ),
            )
        )
        covered_bonds.update(
            (min(atom_1, atom_2), max(atom_1, atom_2))
            for atom_1, atom_2 in present_bonds
        )

    missing_bonds = [
        (atom_1, atom_2)
        for atom_1, atom_2 in rule.forward_bonds_broken
        if (min(atom_1, atom_2), max(atom_1, atom_2)) not in covered_bonds
    ]
    if missing_bonds:
        raise RouteSplitError(
            "could not find reactant source for broken bonds "
            f"{[list(bond) for bond in missing_bonds]}"
        )
    return tuple(sources)


def replace_atom_from_source(
    molecule: Any,
    source_molecule: Any,
    atom_id: int,
) -> None:
    if not molecule.has_atom(atom_id) or not source_molecule.has_atom(atom_id):
        return

    neighbor_bonds: list[tuple[int, int]] = []
    for atom_1, atom_2, bond in list(molecule.bonds()):
        atom_1 = int(atom_1)
        atom_2 = int(atom_2)
        if atom_1 == atom_id:
            neighbor_bonds.append((atom_2, int(bond)))
        elif atom_2 == atom_id:
            neighbor_bonds.append((atom_1, int(bond)))

    molecule.delete_atom(atom_id, _skip_calculation=True)
    source_atom = source_molecule.atom(atom_id)
    molecule.add_atom(
        source_atom.copy(),
        atom_id,
        charge=getattr(source_atom, "charge", 0),
        is_radical=getattr(source_atom, "is_radical", False),
        _skip_calculation=True,
    )
    for neighbor, bond_order in neighbor_bonds:
        if molecule.has_atom(neighbor):
            molecule.add_bond(
                atom_id,
                neighbor,
                bond_order,
                _skip_calculation=True,
            )


def restore_formed_endpoint_atoms_from_reactants(
    molecule: Any,
    reaction: Any,
    rule: SingleCenterRule,
) -> None:
    endpoint_atoms = {
        atom_id for bond in rule.forward_bonds_formed for atom_id in bond
    }
    for atom_id in sorted(endpoint_atoms):
        for reactant in reaction.reactants:
            if reactant.has_atom(atom_id):
                replace_atom_from_source(molecule, reactant, atom_id)
                break


def undo_substitution_rule_on_active_molecule(
    active_molecule: Any,
    reaction: Any,
    rule: SingleCenterRule,
) -> Any:
    if not rule.forward_bonds_formed or not rule.forward_bonds_broken:
        raise RouteSplitError("substitution fallback requires formed and broken bonds")
    working = active_molecule.copy()
    for source_molecule, source_rule in source_molecules_for_broken_rule(
        reaction,
        rule,
    ):
        working = restore_broken_rule_on_product(
            working,
            source_molecule,
            source_rule,
        )

    for atom_1, atom_2 in rule.forward_bonds_formed:
        if not molecule_has_bond(working, atom_1, atom_2):
            raise RouteSplitError(
                "substitution fallback could not find formed bond "
                f"{atom_1}-{atom_2}"
            )
        working.delete_bond(atom_1, atom_2, _skip_calculation=True)

    restore_formed_endpoint_atoms_from_reactants(working, reaction, rule)

    try:
        working.fix_structure()
    except Exception:
        pass
    try:
        working.fix_stereo()
    except Exception:
        pass
    return working


def molecule_components(molecule: Any) -> tuple[Any, ...]:
    try:
        return tuple(molecule.split())
    except Exception:
        return (molecule,)


def rule_formed_atoms(rule: SingleCenterRule) -> frozenset[int]:
    return frozenset(
        atom_id for bond in rule.forward_bonds_formed for atom_id in bond
    )


def component_has_atoms(component: Any, atoms: Iterable[int]) -> bool:
    return all(component.has_atom(int(atom_id)) for atom_id in atoms)


def active_component_index_for_remaining_rules(
    components: tuple[Any, ...],
    remaining_rules: tuple[SingleCenterRule, ...],
) -> int:
    required_atoms = frozenset(
        atom_id
        for rule in remaining_rules
        for atom_id in rule_formed_atoms(rule)
    )
    if not required_atoms:
        if len(components) == 1:
            return 0
        raise RouteSplitError(
            "substitution fallback cannot choose active component"
        )
    candidates = [
        index
        for index, component in enumerate(components)
        if component_has_atoms(component, required_atoms)
    ]
    if len(candidates) != 1:
        raise RouteSplitError(
            "substitution fallback cannot choose active component"
        )
    return candidates[0]


def find_matching_original_child_index(
    component_smiles: str,
    original_children: tuple[dict[str, Any], ...],
    used_indices: set[int],
) -> int | None:
    for index, child in enumerate(original_children):
        if index in used_indices:
            continue
        child_smiles = molecule_node_smiles(child)
        if same_molecule_smiles(component_smiles, child_smiles):
            return index
        if same_mapped_heavy_graph_smiles(component_smiles, child_smiles):
            return index
    return None


def same_mapped_heavy_graph_smiles(left_smiles: str, right_smiles: str) -> bool:
    from chython import smiles as parse_smiles

    try:
        left = parse_smiles(left_smiles)
        right = parse_smiles(right_smiles)
    except Exception:
        return False
    left_atoms = molecule_atom_ids(left)
    right_atoms = molecule_atom_ids(right)
    if left_atoms != right_atoms:
        return False
    for atom_id in left_atoms:
        if atom_symbol_in_molecule(left, atom_id) != atom_symbol_in_molecule(
            right,
            atom_id,
        ):
            return False

    def bond_pairs(molecule: Any) -> set[tuple[int, int]]:
        return {
            (min(int(atom_1), int(atom_2)), max(int(atom_1), int(atom_2)))
            for atom_1, atom_2, _bond in molecule.bonds()
        }

    return bond_pairs(left) == bond_pairs(right)


def component_node(
    component: Any,
    original_children: tuple[dict[str, Any], ...],
    used_original_child_indices: set[int],
    *,
    require_original: bool,
    in_stock: bool,
) -> dict[str, Any]:
    component_smiles = format(component, "m")
    original_index = find_matching_original_child_index(
        component_smiles,
        original_children,
        used_original_child_indices,
    )
    if original_index is not None:
        used_original_child_indices.add(original_index)
        return copy.deepcopy(original_children[original_index])
    if require_original:
        raise RouteSplitError(
            "substitution fallback final component does not match original reactants"
        )
    return {
        "type": "mol",
        "smiles": component_smiles,
        "mapped_smiles": component_smiles,
        "in_stock": in_stock,
        "children": [],
    }


def split_reaction_node_via_substitution_disconnections(
    plan: SplitPlan,
    *,
    ordered_rules: tuple[SingleCenterRule, ...],
) -> dict[str, Any]:
    from chython import smiles as parse_smiles

    reaction = parse_smiles(plan.original_reaction_smiles)
    if len(reaction.products) != 1:
        raise RouteSplitError("substitution fallback requires one reaction product")
    if not ordered_rules:
        raise RouteSplitError("substitution fallback has no rules")

    active_molecule = reaction.products[0]
    active_smiles = format(active_molecule, "m")
    generated_reaction_smiles: list[str] = []
    used_original_child_indices: set[int] = set()
    first_reaction_node: dict[str, Any] | None = None
    current_mol_node: dict[str, Any] | None = None

    for step_index, rule in enumerate(ordered_rules, start=1):
        remaining_rules = ordered_rules[step_index:]
        disconnected = undo_substitution_rule_on_active_molecule(
            active_molecule,
            reaction,
            rule,
        )
        components = molecule_components(disconnected)
        if not components:
            raise RouteSplitError("substitution fallback produced no components")

        if remaining_rules:
            active_index = active_component_index_for_remaining_rules(
                components,
                remaining_rules,
            )
            active_component = components[active_index]
            child_mol_node = {
                "type": "mol",
                "smiles": format(active_component, "m"),
                "mapped_smiles": format(active_component, "m"),
                "in_stock": False,
                "children": [],
            }
            child_nodes = [child_mol_node]
            for index, component in enumerate(components):
                if index == active_index:
                    continue
                child_nodes.append(
                    component_node(
                        component,
                        plan.original_children,
                        used_original_child_indices,
                        require_original=False,
                        in_stock=True,
                    )
                )
        else:
            child_nodes = [
                component_node(
                    component,
                    plan.original_children,
                    used_original_child_indices,
                    require_original=True,
                    in_stock=True,
                )
                for component in components
            ]
            child_mol_node = None

        child_smiles = ".".join(molecule_node_smiles(child) for child in child_nodes)
        reaction_smiles = f"{child_smiles}>>{active_smiles}"
        reaction_node = {
            "type": "reaction",
            "smiles": reaction_smiles,
            "metadata": {
                "smiles": reaction_smiles,
                **split_metadata(
                    plan,
                    step_index=step_index,
                    step_count=len(ordered_rules),
                    rule_smarts=rule.rule_smarts,
                    forward_change_kind=rule.forward_change_kind,
                ),
            },
            "children": child_nodes,
        }
        reaction_node["metadata"]["route_preprocessing_split"][
            "split_method"
        ] = "substitution_disconnection"
        generated_reaction_smiles.append(reaction_smiles)

        if current_mol_node is None:
            first_reaction_node = reaction_node
        else:
            current_mol_node["children"] = [reaction_node]

        if child_mol_node is None:
            current_mol_node = None
        else:
            current_mol_node = child_mol_node
            active_molecule = active_component
            active_smiles = format(active_component, "m")

    if first_reaction_node is None:
        raise RouteSplitError("substitution fallback did not create a split route")

    unmatched_original_children = [
        index
        for index, _child in enumerate(plan.original_children)
        if index not in used_original_child_indices
    ]
    if unmatched_original_children:
        raise RouteSplitError(
            "substitution fallback did not account for all original reactants"
        )

    replace_reaction_node(
        plan.parent_mol_node,
        plan.reaction_node,
        first_reaction_node,
    )
    return {
        "reaction_id": plan.reaction_id,
        "parent_mol_id": plan.parent_mol_id,
        "original_reaction_smiles": plan.original_reaction_smiles,
        "multicenter_rule_smarts": plan.extraction.multicenter_rule_smarts,
        "single_center_rules": [rule.rule_smarts for rule in ordered_rules],
        "single_center_rule_details": [rule_detail(rule) for rule in ordered_rules],
        "single_center_count": len(ordered_rules),
        "protection_related": bool(plan.protection_matches),
        "protection_match_count": len(plan.protection_matches),
        "protection_atom_ids": sorted(plan.protection_atom_ids),
        "generated_reaction_smiles": generated_reaction_smiles,
        "split_method": "substitution_disconnection",
    }


def split_reaction_node_via_formed_bond_disconnections(
    plan: SplitPlan,
    *,
    ordered_rules: tuple[SingleCenterRule, ...],
) -> dict[str, Any]:
    from chython import smiles as parse_smiles

    reaction = parse_smiles(plan.original_reaction_smiles)
    if len(reaction.products) != 1:
        raise RouteSplitError("formed-bond fallback requires one reaction product")

    active_molecule = reaction.products[0]
    active_smiles = format(active_molecule, "m")
    generated_reaction_smiles: list[str] = []
    first_reaction_node: dict[str, Any] | None = None
    current_mol_node: dict[str, Any] | None = None
    last_intermediate_slot: tuple[list[dict[str, Any]], int] | None = None

    for step_index, rule in enumerate(ordered_rules, start=1):
        if not rule.forward_bonds_formed or not rule.forward_bonds_broken:
            raise RouteSplitError(
                "formed-bond fallback requires forming/breaking rules"
            )
        source_molecule = source_molecule_for_broken_rule(reaction, rule)
        substrate_molecule = substrate_molecule_for_formed_rule(
            reaction,
            source_molecule,
            rule,
        )
        intermediate_molecule = disconnect_formed_rule_from_product(
            active_molecule,
            source_molecule,
            rule,
            substrate_molecule,
        )
        intermediate_smiles = format(intermediate_molecule, "m")
        if same_molecule_smiles(intermediate_smiles, active_smiles):
            raise RouteSplitError(
                "formed-bond fallback did not change the active molecule"
            )

        reagent_smiles = format(source_molecule, "m")
        child_mol_node = {
            "type": "mol",
            "smiles": intermediate_smiles,
            "mapped_smiles": intermediate_smiles,
            "in_stock": False,
            "children": [],
        }
        reagent_mol_node = {
            "type": "mol",
            "smiles": reagent_smiles,
            "mapped_smiles": reagent_smiles,
            "in_stock": True,
            "children": [],
        }
        reaction_smiles = f"{reagent_smiles}.{intermediate_smiles}>>{active_smiles}"
        reaction_node = {
            "type": "reaction",
            "smiles": reaction_smiles,
            "metadata": {
                "smiles": reaction_smiles,
                **split_metadata(
                    plan,
                    step_index=step_index,
                    step_count=len(ordered_rules),
                    rule_smarts=rule.rule_smarts,
                    forward_change_kind=rule.forward_change_kind,
                ),
            },
            "children": [child_mol_node, reagent_mol_node],
        }
        reaction_node["metadata"]["route_preprocessing_split"][
            "split_method"
        ] = "formed_bond_disconnection"
        generated_reaction_smiles.append(reaction_smiles)

        if current_mol_node is None:
            first_reaction_node = reaction_node
        else:
            current_mol_node["children"] = [reaction_node]
        current_mol_node = child_mol_node
        last_intermediate_slot = (reaction_node["children"], 0)
        active_molecule = intermediate_molecule
        active_smiles = intermediate_smiles

    if first_reaction_node is None or last_intermediate_slot is None:
        raise RouteSplitError("formed-bond fallback did not create a split route")

    matching_original_child = next(
        (
            child
            for child in plan.original_children
            if same_molecule_smiles(active_smiles, molecule_node_smiles(child))
        ),
        None,
    )
    if matching_original_child is None:
        raise RouteSplitError(
            "formed-bond fallback final intermediate does not match original reactants"
        )

    children, index = last_intermediate_slot
    children[index] = matching_original_child
    replace_reaction_node(
        plan.parent_mol_node,
        plan.reaction_node,
        first_reaction_node,
    )
    return {
        "reaction_id": plan.reaction_id,
        "parent_mol_id": plan.parent_mol_id,
        "original_reaction_smiles": plan.original_reaction_smiles,
        "multicenter_rule_smarts": plan.extraction.multicenter_rule_smarts,
        "single_center_rules": [rule.rule_smarts for rule in ordered_rules],
        "single_center_rule_details": [rule_detail(rule) for rule in ordered_rules],
        "single_center_count": len(ordered_rules),
        "protection_related": bool(plan.protection_matches),
        "protection_match_count": len(plan.protection_matches),
        "protection_atom_ids": sorted(plan.protection_atom_ids),
        "generated_reaction_smiles": generated_reaction_smiles,
        "split_method": "formed_bond_disconnection",
    }


def mapped_intermediate_rule_orders(
    rules: tuple[SingleCenterRule, ...],
    protection_atom_ids: frozenset[int] = frozenset(),
) -> Iterable[tuple[tuple[SingleCenterRule, ...], tuple[SingleCenterRule, ...]]]:
    restore_rules = tuple(
        rule
        for rule in rules
        if rule.forward_change_kind == "bond_breaking"
        and rule.forward_bonds_broken
        and not rule.forward_bonds_formed
    )
    remaining_rules = tuple(rule for rule in rules if rule not in restore_rules)
    if not restore_rules:
        return
    if (
        not any(
            rule.forward_bonds_formed or rule.forward_bonds_changed
            for rule in remaining_rules
        )
        and remaining_rules
    ):
        return

    if protection_atom_ids:
        restore_rules = tuple(
            sorted(
                restore_rules,
                key=lambda rule: 0
                if rule.center_atoms & protection_atom_ids
                else 1,
            )
        )

    yielded: set[tuple[str, ...]] = set()
    restore_orders: Iterable[tuple[SingleCenterRule, ...]]
    if len(restore_rules) <= 6:
        restore_orders = permutations(restore_rules)
    else:
        restore_orders = (restore_rules,)
    for restore_order in restore_orders:
        key = tuple(rule.rule_smarts for rule in restore_order + remaining_rules)
        if key in yielded:
            continue
        yielded.add(key)
        yield tuple(restore_order), remaining_rules


def formed_bond_disconnection_rule_orders(
    rules: tuple[SingleCenterRule, ...],
    protection_atom_ids: frozenset[int] = frozenset(),
) -> Iterable[tuple[SingleCenterRule, ...]]:
    if not rules:
        return
    if not all(rule.forward_bonds_formed and rule.forward_bonds_broken for rule in rules):
        return
    ordered_rules = rules
    if protection_atom_ids:
        ordered_rules = tuple(
            sorted(
                rules,
                key=lambda rule: 0
                if rule.center_atoms & protection_atom_ids
                else 1,
            )
        )
    yielded: set[tuple[str, ...]] = set()
    candidates: Iterable[tuple[SingleCenterRule, ...]]
    if len(ordered_rules) <= 6:
        candidates = permutations(ordered_rules)
    else:
        candidates = (ordered_rules,)
    for candidate in candidates:
        key = tuple(rule.rule_smarts for rule in candidate)
        if key in yielded:
            continue
        yielded.add(key)
        yield tuple(candidate)


def replace_reaction_node(
    parent_mol_node: dict[str, Any],
    old_reaction_node: dict[str, Any],
    new_reaction_node: dict[str, Any],
) -> None:
    children = parent_mol_node.get("children", []) or []
    for index, child in enumerate(children):
        if child is old_reaction_node:
            children[index] = new_reaction_node
            return
    raise RouteSplitError("original reaction node is no longer attached to its parent")


def split_reaction_node(
    plan: SplitPlan,
    *,
    protection_config: ProtectionAnalysisConfig,
    unwrapper: Callable[..., Any] = unwrap_rule_sequence,
) -> dict[str, Any]:
    if plan.protection_matches and not plan.protection_atom_ids:
        raise RouteSplitError("protection match did not expose mapped atoms")

    last_error: Exception | None = None
    for ordered_rules in split_candidate_rule_orders(
        plan.extraction.single_center_rules,
        plan.protection_atom_ids,
        deprotection_first=protection_config.deprotection_first,
    ):
        try:
            generated_root = unwrap_split_route(
                plan,
                ordered_rules,
                unwrapper=unwrapper,
            )
            first_reactions = [
                child
                for child in generated_root.get("children", []) or []
                if isinstance(child, dict) and child.get("type") == "reaction"
            ]
            if len(first_reactions) != 1:
                raise RouteSplitError("split route did not produce one root reaction")

            generated_reactions = collect_reaction_nodes(generated_root)
            if len(generated_reactions) != len(ordered_rules):
                raise RouteSplitError(
                    "split route reaction count does not match single-center rules"
                )
            for step_index, reaction_node in enumerate(generated_reactions, start=1):
                metadata = reaction_node.setdefault("metadata", {})
                metadata["route_preprocessing_split"] = {
                    "route_id": str(plan.route_id),
                    "original_reaction_id": plan.reaction_id,
                    "original_reaction_smiles": plan.original_reaction_smiles,
                    "split_step": step_index,
                    "split_steps": len(generated_reactions),
                    "rule_smarts": (
                        ordered_rules[step_index - 1].rule_smarts
                        if step_index <= len(ordered_rules)
                        else ""
                    ),
                    "forward_change_kind": (
                        ordered_rules[step_index - 1].forward_change_kind
                        if step_index <= len(ordered_rules)
                        else ""
                    ),
                }

            replace_reaction_node(
                plan.parent_mol_node,
                plan.reaction_node,
                first_reactions[0],
            )
            return {
                "reaction_id": plan.reaction_id,
                "parent_mol_id": plan.parent_mol_id,
                "original_reaction_smiles": plan.original_reaction_smiles,
                "multicenter_rule_smarts": plan.extraction.multicenter_rule_smarts,
                "single_center_rules": [rule.rule_smarts for rule in ordered_rules],
                "single_center_rule_details": [
                    rule_detail(rule) for rule in ordered_rules
                ],
                "single_center_count": len(ordered_rules),
                "protection_related": bool(plan.protection_matches),
                "protection_match_count": len(plan.protection_matches),
                "protection_atom_ids": sorted(plan.protection_atom_ids),
                "generated_reaction_smiles": [
                    reaction_smiles_from_node(reaction)
                    for reaction in generated_reactions
                ],
            }
        except Exception as exc:
            last_error = exc
            continue

    for restore_rules, remaining_rules in mapped_intermediate_rule_orders(
        plan.extraction.single_center_rules,
        plan.protection_atom_ids,
    ):
        try:
            return split_reaction_node_via_mapped_intermediates(
                plan,
                restore_rules=restore_rules,
                remaining_rules=remaining_rules,
            )
        except Exception as exc:
            last_error = exc
            continue

    for ordered_rules in formed_bond_disconnection_rule_orders(
        plan.extraction.single_center_rules,
        plan.protection_atom_ids,
    ):
        try:
            return split_reaction_node_via_substitution_disconnections(
                plan,
                ordered_rules=ordered_rules,
            )
        except Exception as exc:
            last_error = exc
            continue

    for ordered_rules in formed_bond_disconnection_rule_orders(
        plan.extraction.single_center_rules,
        plan.protection_atom_ids,
    ):
        try:
            return split_reaction_node_via_formed_bond_disconnections(
                plan,
                ordered_rules=ordered_rules,
            )
        except Exception as exc:
            last_error = exc
            continue

    raise RouteSplitError(str(last_error) if last_error else "no split order generated")


def unresolved_reaction_record(
    route_id: Any,
    reaction_id: str,
    reaction_smiles: str,
    reason: str,
    extraction: ReactionGranularity | None = None,
    message: str = "",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "route_id": str(route_id),
        "reaction_id": reaction_id,
        "reaction_smiles": reaction_smiles,
        "reason": reason,
    }
    if extraction is not None:
        record["multicenter_rule_smarts"] = extraction.multicenter_rule_smarts
        record["single_center_rules"] = [
            rule.rule_smarts for rule in extraction.single_center_rules
        ]
        record["single_center_rule_details"] = [
            {
                "rule_smarts": rule.rule_smarts,
                "center_atoms": sorted(rule.center_atoms),
                "forward_change_kind": rule.forward_change_kind,
                "forward_bonds_formed": [
                    list(bond) for bond in rule.forward_bonds_formed
                ],
                "forward_bonds_broken": [
                    list(bond) for bond in rule.forward_bonds_broken
                ],
                "forward_bonds_changed": [
                    list(bond) for bond in rule.forward_bonds_changed
                ],
            }
            for rule in extraction.single_center_rules
        ]
        record["center_components"] = [
            sorted(component) for component in extraction.center_components
        ]
    if message:
        record["message"] = message
    return record


def error_record(route_id: Any, stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "route_id": str(route_id),
        "stage": stage,
        "error_type": type(exc).__qualname__,
        "message": str(exc) or traceback.format_exc(limit=1).strip(),
    }


def annotate_route(
    route: dict[str, Any],
    *,
    route_id: Any,
    status: str,
    changes: list[dict[str, Any]],
    unresolved_reactions: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    metadata = route.setdefault("metadata", {})
    metadata["route_preprocessing"] = {
        "route_id": str(route_id),
        "status": status,
        "changes": changes,
        "unresolved_reactions": unresolved_reactions,
        "errors": errors,
    }


def process_route(
    route_id: Any,
    route: dict[str, Any],
    *,
    extractor: Any,
    protection_rules: dict[str, ProtectionRule],
    protection_config: ProtectionAnalysisConfig,
    normalizer: Callable[[dict[str, Any]], dict[str, Any]] = normalize_route_tree,
    protection_detector: Callable[
        [RouteIndex, ReactionRecord, dict[str, ProtectionRule], ProtectionAnalysisConfig],
        list[Any],
    ] = detect_deprotections,
    protection_event_analyzer: Callable[..., tuple[list[Any], list[Any], RouteIndex]] = (
        analyze_route_protection
    ),
    unwrapper: Callable[..., Any] = unwrap_rule_sequence,
    ignore_errors: bool = False,
) -> RoutePreprocessResult:
    errors: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    split_plans: list[SplitPlan] = []
    multicenter_reactions = 0
    protection_multicenter_reactions = 0

    try:
        normalized_route = normalizer(route)
    except Exception as exc:
        if not ignore_errors:
            raise
        errors.append(error_record(route_id, "normalize_route", exc))
        route_copy = copy.deepcopy(route)
        annotate_route(
            route_copy,
            route_id=route_id,
            status="error",
            changes=[],
            unresolved_reactions=[],
            errors=errors,
        )
        return RoutePreprocessResult(
            route_id=route_id,
            route=route_copy,
            normalized=False,
            modified=False,
            multicenter_reactions=0,
            protection_multicenter_reactions=0,
            split_reactions=0,
            protection_split_reactions=0,
            changes=[],
            unresolved_reactions=[],
            errors=errors,
        )

    source_correction_changes = apply_reaction_smiles_corrections(normalized_route)
    if source_correction_changes:
        try:
            normalized_route = normalizer(normalized_route)
        except Exception as exc:
            if not ignore_errors:
                raise
            errors.append(error_record(route_id, "normalize_corrected_route", exc))
            annotate_route(
                normalized_route,
                route_id=route_id,
                status="error",
                changes=source_correction_changes,
                unresolved_reactions=[],
                errors=errors,
            )
            return RoutePreprocessResult(
                route_id=route_id,
                route=normalized_route,
                normalized=True,
                modified=True,
                multicenter_reactions=0,
                protection_multicenter_reactions=0,
                split_reactions=0,
                protection_split_reactions=0,
                changes=source_correction_changes,
                unresolved_reactions=[],
                errors=errors,
            )

    try:
        index = build_route_index(normalized_route)
    except Exception as exc:
        if not ignore_errors:
            raise
        errors.append(error_record(route_id, "build_route_index", exc))
        annotate_route(
            normalized_route,
            route_id=route_id,
            status="error",
            changes=source_correction_changes,
            unresolved_reactions=[],
            errors=errors,
        )
        return RoutePreprocessResult(
            route_id=route_id,
            route=normalized_route,
            normalized=True,
            modified=bool(source_correction_changes),
            multicenter_reactions=0,
            protection_multicenter_reactions=0,
            split_reactions=0,
            protection_split_reactions=0,
            changes=source_correction_changes,
            unresolved_reactions=[],
            errors=errors,
        )

    protection_events_by_protection_node: dict[str, tuple[Any, ...]] | None = None

    def protection_events_for_node(reaction_id: str) -> tuple[Any, ...]:
        nonlocal protection_events_by_protection_node
        if protection_events_by_protection_node is None:
            event_config = replace(protection_config, collect_interval_rules=False)
            events, _interval_rules, _event_index = protection_event_analyzer(
                index.route,
                route_id,
                protection_rules,
                config=event_config,
            )
            grouped: dict[str, list[Any]] = {}
            for event in events:
                protection_node_id = getattr(event, "protection_node_id", "")
                if not protection_node_id:
                    continue
                grouped.setdefault(str(protection_node_id), []).append(event)
            protection_events_by_protection_node = {
                node_id: tuple(node_events)
                for node_id, node_events in grouped.items()
            }
        return protection_events_by_protection_node.get(reaction_id, ())

    for reaction_id in index.reaction_order:
        rxn_record = index.reaction_records[reaction_id]
        try:
            extraction = extractor.extract(rxn_record.reaction_smiles)
        except Exception as exc:
            if not ignore_errors:
                raise
            errors.append(error_record(route_id, "extract_reaction_rules", exc))
            unresolved.append(
                unresolved_reaction_record(
                    route_id,
                    reaction_id,
                    rxn_record.reaction_smiles,
                    "rule_extraction_error",
                    message=str(exc),
                )
            )
            continue

        if extraction.skipped or not extraction.is_multicenter:
            continue

        source_error_reason = source_reaction_error_reason(
            rxn_record.reaction_smiles
        )
        if source_error_reason is not None:
            multicenter_reactions += 1
            unresolved.append(
                unresolved_reaction_record(
                    route_id,
                    reaction_id,
                    rxn_record.reaction_smiles,
                    "source_reaction_error",
                    extraction,
                    message=source_error_reason,
                )
            )
            continue

        try:
            if semantic_single_center_reason(rxn_record.reaction_smiles, extraction):
                continue
        except Exception as exc:
            if not ignore_errors:
                raise
            errors.append(error_record(route_id, "classify_reaction_center", exc))

        multicenter_reactions += 1
        try:
            matches = tuple(
                protection_detector(
                    index,
                    rxn_record,
                    protection_rules,
                    protection_config,
                )
            )
        except Exception as exc:
            if not ignore_errors:
                raise
            errors.append(error_record(route_id, "detect_deprotections", exc))
            matches = ()

        if not matches:
            try:
                matches = protection_events_for_node(reaction_id)
            except Exception as exc:
                if not ignore_errors:
                    raise
                errors.append(error_record(route_id, "analyze_route_protection", exc))
                matches = ()

        protection_atom_ids = frozenset()
        if matches:
            protection_atom_ids = protection_atom_ids_from_matches(matches)
            if not any(
                rule.center_atoms & protection_atom_ids
                for rule in extraction.single_center_rules
            ):
                unresolved.append(
                    unresolved_reaction_record(
                        route_id,
                        reaction_id,
                        rxn_record.reaction_smiles,
                        "protection_center_not_matched_to_single_center_rule",
                        extraction,
                    )
                )
                continue
            protection_multicenter_reactions += 1

        parent_mol_id = index.parent_mol_by_reaction[reaction_id]
        parent_mol_node = index.molecule_records[parent_mol_id].node
        original_children = tuple(
            child
            for child in rxn_record.node.get("children", []) or []
            if isinstance(child, dict) and child.get("type") == "mol"
        )
        split_plans.append(
            SplitPlan(
                route_id=route_id,
                reaction_id=reaction_id,
                parent_mol_id=parent_mol_id,
                reaction_node=rxn_record.node,
                parent_mol_node=parent_mol_node,
                original_children=original_children,
                original_reaction_smiles=rxn_record.reaction_smiles,
                parent_smiles=index.molecule_records[parent_mol_id].smiles,
                extraction=extraction,
                protection_matches=matches,
                protection_atom_ids=protection_atom_ids,
            )
        )

    if unresolved:
        annotate_route(
            index.route,
            route_id=route_id,
            status="unresolved",
            changes=source_correction_changes,
            unresolved_reactions=unresolved,
            errors=errors,
        )
        return RoutePreprocessResult(
            route_id=route_id,
            route=index.route,
            normalized=True,
            modified=False,
            multicenter_reactions=multicenter_reactions,
            protection_multicenter_reactions=protection_multicenter_reactions,
            split_reactions=0,
            protection_split_reactions=0,
            changes=[],
            unresolved_reactions=unresolved,
            errors=errors,
        )

    changes: list[dict[str, Any]] = []
    protection_split_reactions = 0
    for plan in split_plans:
        try:
            change = split_reaction_node(
                plan,
                protection_config=protection_config,
                unwrapper=unwrapper,
            )
            changes.append(change)
            protection_split_reactions += int(bool(change.get("protection_related")))
        except Exception as exc:
            if not ignore_errors:
                raise
            errors.append(error_record(route_id, "split_multicenter_reaction", exc))
            reason = (
                "protection_related_split_failed"
                if plan.protection_matches
                else "multicenter_split_failed"
            )
            unresolved.append(
                unresolved_reaction_record(
                    route_id,
                    plan.reaction_id,
                    plan.original_reaction_smiles,
                    reason,
                    plan.extraction,
                    message=str(exc),
                )
            )

    if unresolved:
        annotate_route(
            normalized_route,
            route_id=route_id,
            status="unresolved",
            changes=[],
            unresolved_reactions=unresolved,
            errors=errors,
        )
        return RoutePreprocessResult(
            route_id=route_id,
            route=normalized_route,
            normalized=True,
            modified=False,
            multicenter_reactions=multicenter_reactions,
            protection_multicenter_reactions=protection_multicenter_reactions,
            split_reactions=0,
            protection_split_reactions=0,
            changes=[],
            unresolved_reactions=unresolved,
            errors=errors,
        )

    if changes:
        all_route_changes = source_correction_changes + changes
        try:
            final_route = normalizer(index.route)
        except Exception as exc:
            if not ignore_errors:
                raise
            errors.append(error_record(route_id, "normalize_split_route", exc))
            unresolved = [
                unresolved_reaction_record(
                    route_id,
                    plan.reaction_id,
                    plan.original_reaction_smiles,
                    "split_route_normalization_failed",
                    plan.extraction,
                    message=str(exc),
                )
                for plan in split_plans
            ]
            annotate_route(
                normalized_route,
                route_id=route_id,
                status="unresolved",
                changes=source_correction_changes,
                unresolved_reactions=unresolved,
                errors=errors,
            )
            return RoutePreprocessResult(
                route_id=route_id,
                route=normalized_route,
                normalized=True,
                modified=False,
                multicenter_reactions=multicenter_reactions,
                protection_multicenter_reactions=protection_multicenter_reactions,
                split_reactions=0,
                protection_split_reactions=0,
                changes=[],
                unresolved_reactions=unresolved,
                errors=errors,
            )
        annotate_route(
            final_route,
            route_id=route_id,
            status="modified",
            changes=all_route_changes,
            unresolved_reactions=[],
            errors=errors,
        )
        return RoutePreprocessResult(
            route_id=route_id,
            route=final_route,
            normalized=True,
            modified=True,
            multicenter_reactions=multicenter_reactions,
            protection_multicenter_reactions=protection_multicenter_reactions,
            split_reactions=len(changes),
            protection_split_reactions=protection_split_reactions,
            changes=all_route_changes,
            unresolved_reactions=[],
            errors=errors,
        )

    if source_correction_changes:
        annotate_route(
            index.route,
            route_id=route_id,
            status="modified",
            changes=source_correction_changes,
            unresolved_reactions=[],
            errors=errors,
        )
        return RoutePreprocessResult(
            route_id=route_id,
            route=index.route,
            normalized=True,
            modified=True,
            multicenter_reactions=multicenter_reactions,
            protection_multicenter_reactions=protection_multicenter_reactions,
            split_reactions=0,
            protection_split_reactions=0,
            changes=source_correction_changes,
            unresolved_reactions=[],
            errors=errors,
        )

    return RoutePreprocessResult(
        route_id=route_id,
        route=index.route,
        normalized=True,
        modified=False,
        multicenter_reactions=multicenter_reactions,
        protection_multicenter_reactions=protection_multicenter_reactions,
        split_reactions=0,
        protection_split_reactions=0,
        changes=[],
        unresolved_reactions=[],
        errors=errors,
    )


def empty_collection_like(routes_json: Any) -> Any:
    if isinstance(routes_json, list):
        return []
    if isinstance(routes_json, dict):
        return {}
    raise TypeError(f"unsupported routes JSON root: {type(routes_json)!r}")


def add_route_to_collection(collection: Any, route_id: Any, route: dict[str, Any]) -> None:
    if isinstance(collection, list):
        collection.append(route)
        return
    collection[str(route_id)] = route


def route_id_sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, str) and value.isdigit():
        return (0, int(value))
    return (1, str(value))


def preprocess_routes_json(
    routes_json: Any,
    *,
    extractor: Any,
    protection_rules: dict[str, ProtectionRule],
    protection_config: ProtectionAnalysisConfig,
    normalizer: Callable[[dict[str, Any]], dict[str, Any]] = normalize_route_tree,
    protection_detector: Callable[
        [RouteIndex, ReactionRecord, dict[str, ProtectionRule], ProtectionAnalysisConfig],
        list[Any],
    ] = detect_deprotections,
    protection_event_analyzer: Callable[..., tuple[list[Any], list[Any], RouteIndex]] = (
        analyze_route_protection
    ),
    unwrapper: Callable[..., Any] = unwrap_rule_sequence,
    ignore_errors: bool = False,
    limit: int | None = None,
    progress_interval: int = 0,
) -> tuple[Any, dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    cleaned_routes = empty_collection_like(routes_json)
    resolved_routes: dict[str, dict[str, Any]] = {}
    unresolved_routes: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    modified_route_ids: list[str] = []
    unresolved_route_ids: list[str] = []
    all_changes: list[dict[str, Any]] = []
    total_routes = 0
    normalized_routes = 0
    multicenter_reactions = 0
    protection_multicenter_reactions = 0
    split_reactions = 0
    protection_split_reactions = 0

    for index, (route_id, route) in enumerate(route_items(routes_json), start=1):
        if limit is not None and index > limit:
            break
        result = process_route(
            route_id,
            route,
            extractor=extractor,
            protection_rules=protection_rules,
            protection_config=protection_config,
            normalizer=normalizer,
            protection_detector=protection_detector,
            protection_event_analyzer=protection_event_analyzer,
            unwrapper=unwrapper,
            ignore_errors=ignore_errors,
        )
        total_routes += 1
        normalized_routes += int(result.normalized)
        multicenter_reactions += result.multicenter_reactions
        protection_multicenter_reactions += result.protection_multicenter_reactions
        split_reactions += result.split_reactions
        protection_split_reactions += result.protection_split_reactions
        errors.extend(result.errors)
        add_route_to_collection(cleaned_routes, route_id, result.route)

        route_id_text = str(route_id)
        if result.modified:
            modified_route_ids.append(route_id_text)
            resolved_routes[route_id_text] = result.route
            all_changes.extend(
                {"route_id": route_id_text, **change} for change in result.changes
            )
        if result.unresolved_reactions:
            unresolved_route_ids.append(route_id_text)
            unresolved_routes[route_id_text] = result.route

        if progress_interval and total_routes % progress_interval == 0:
            print(
                "[preprocess-routes] processed "
                f"{total_routes} routes; modified={len(modified_route_ids)}; "
                f"unresolved={len(unresolved_route_ids)}; errors={len(errors)}",
                file=sys.stderr,
                flush=True,
            )

    summary = {
        "total_routes_processed": total_routes,
        "number_of_normalized_routes": normalized_routes,
        "number_of_routes_modified": len(modified_route_ids),
        "number_of_multicenter_reactions_found": multicenter_reactions,
        "number_of_multicenter_reactions_split": split_reactions,
        "number_of_protection_related_multicenter_reactions_split": (
            protection_split_reactions
        ),
        "number_of_non_protection_multicenter_reactions_split": (
            split_reactions - protection_split_reactions
        ),
        "number_of_protection_related_multicenter_reactions_found": (
            protection_multicenter_reactions
        ),
        "number_of_unresolved_multicenter_routes": len(unresolved_route_ids),
        "route_ids_for_modified_routes": sorted(
            modified_route_ids,
            key=route_id_sort_key,
        ),
        "route_ids_for_unresolved_routes": sorted(
            unresolved_route_ids,
            key=route_id_sort_key,
        ),
        "changes": all_changes,
        "errors": errors,
        "number_of_errors": len(errors),
    }
    return cleaned_routes, resolved_routes, unresolved_routes, summary


def dataset_output_paths(
    output_dir: Path,
    dataset_name: str,
    *,
    sidecar_dir: Path | None = None,
) -> dict[str, Path]:
    output_path = output_dir / dataset_name
    report_dir = sidecar_dir or output_dir
    stem = output_path.stem
    return {
        "cleaned": output_path,
        "resolved": report_dir / f"{stem}_multicenter_resolved.json",
        "unresolved": report_dir / f"{stem}_multicenter_unresolved.json",
        "summary": report_dir / f"{stem}_preprocess_summary.json",
    }


def resolve_dataset_path(input_dir: Path, dataset: str | Path) -> Path:
    dataset_path = Path(dataset)
    candidates = []
    if dataset_path.is_absolute():
        candidates.append(dataset_path)
    else:
        candidates.append(input_dir / dataset_path)
        name = dataset_path.name
        candidates.append(input_dir / name.replace("_", "-"))
        candidates.append(input_dir / name.replace("-", "_"))

    for candidate in candidates:
        resolved = resolve_existing_path(candidate)
        if resolved.exists():
            return resolved
    return candidates[0]


def preprocess_routes_file(
    input_path: Path,
    output_paths: dict[str, Path],
    *,
    extractor: Any,
    protection_rules: dict[str, ProtectionRule],
    protection_config: ProtectionAnalysisConfig,
    ignore_errors: bool,
    limit: int | None = None,
    progress_interval: int = 0,
    normalizer: Callable[[dict[str, Any]], dict[str, Any]] = normalize_route_tree,
    protection_detector: Callable[
        [RouteIndex, ReactionRecord, dict[str, ProtectionRule], ProtectionAnalysisConfig],
        list[Any],
    ] = detect_deprotections,
    protection_event_analyzer: Callable[..., tuple[list[Any], list[Any], RouteIndex]] = (
        analyze_route_protection
    ),
    unwrapper: Callable[..., Any] = unwrap_rule_sequence,
) -> dict[str, Any]:
    routes_json = read_json(input_path)
    cleaned, resolved, unresolved, summary = preprocess_routes_json(
        routes_json,
        extractor=extractor,
        protection_rules=protection_rules,
        protection_config=protection_config,
        ignore_errors=ignore_errors,
        limit=limit,
        progress_interval=progress_interval,
        normalizer=normalizer,
        protection_detector=protection_detector,
        protection_event_analyzer=protection_event_analyzer,
        unwrapper=unwrapper,
    )
    summary = {
        "input_file": str(input_path),
        "output_files": {name: str(path) for name, path in output_paths.items()},
        **summary,
    }

    write_json(output_paths["cleaned"], cleaned)
    write_json(output_paths["resolved"], resolved)
    write_json(output_paths["unresolved"], unresolved)
    write_json(output_paths["summary"], summary)
    return summary


def preprocess_datasets(args: argparse.Namespace) -> dict[str, Any]:
    setup_runtime_cache_dirs()
    input_dir = resolve_existing_path(args.input_dir)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_root = (
        Path(args.summary_dir).expanduser()
        if getattr(args, "summary_dir", None)
        else None
    )
    protection_config_path = (
        resolve_existing_path(args.protection_config)
        if getattr(args, "protection_config", None)
        else None
    )
    protection_config = ProtectionAnalysisConfig.from_yaml(protection_config_path)
    protection_config.collect_interval_rules = False
    if getattr(args, "ignore_errors", False):
        protection_config.ignore_errors = True
    extractor = SynPlannerGranularityExtractor.from_args(args)
    protection_rules = load_chython_protection_rules()

    dataset_summaries = {}
    for dataset in args.datasets:
        input_path = resolve_dataset_path(input_dir, dataset)
        stage_dir = (
            stage_output_dir(
                summary_root,
                dataset_prefix_from_path(dataset),
                "preprocess",
            )
            if summary_root is not None
            else None
        )
        output_paths = dataset_output_paths(
            output_dir,
            Path(dataset).name,
            sidecar_dir=stage_dir,
        )
        print(
            f"[preprocess-routes] processing {input_path} -> {output_paths['cleaned']}",
            file=sys.stderr,
            flush=True,
        )
        dataset_summary = preprocess_routes_file(
            input_path,
            output_paths,
            extractor=extractor,
            protection_rules=protection_rules,
            protection_config=protection_config,
            ignore_errors=getattr(args, "ignore_errors", False),
            limit=getattr(args, "limit", None),
            progress_interval=getattr(args, "progress_interval", 0),
        )
        write_standard_sidecars(
            output_paths["summary"].parent,
            command_name="preprocess-routes",
            summary=dataset_summary,
            errors=dataset_summary.get("errors", []),
            input_files=[input_path],
            output_files=dataset_summary["output_files"],
            config_path=getattr(args, "config", None),
            cli_args=args,
            write_summary=False,
        )
        dataset_summaries[Path(dataset).name] = dataset_summary

    aggregate = {
        "datasets": dataset_summaries,
        "total_routes_processed": sum(
            summary["total_routes_processed"]
            for summary in dataset_summaries.values()
        ),
        "number_of_normalized_routes": sum(
            summary["number_of_normalized_routes"]
            for summary in dataset_summaries.values()
        ),
        "number_of_routes_modified": sum(
            summary["number_of_routes_modified"]
            for summary in dataset_summaries.values()
        ),
        "number_of_multicenter_reactions_found": sum(
            summary["number_of_multicenter_reactions_found"]
            for summary in dataset_summaries.values()
        ),
        "number_of_multicenter_reactions_split": sum(
            summary["number_of_multicenter_reactions_split"]
            for summary in dataset_summaries.values()
        ),
        "number_of_protection_related_multicenter_reactions_split": sum(
            summary["number_of_protection_related_multicenter_reactions_split"]
            for summary in dataset_summaries.values()
        ),
        "number_of_non_protection_multicenter_reactions_split": sum(
            summary["number_of_non_protection_multicenter_reactions_split"]
            for summary in dataset_summaries.values()
        ),
        "number_of_unresolved_multicenter_routes": sum(
            summary["number_of_unresolved_multicenter_routes"]
            for summary in dataset_summaries.values()
        ),
        "number_of_errors": sum(
            summary["number_of_errors"] for summary in dataset_summaries.values()
        ),
    }
    aggregate_path = (summary_root or output_dir) / "preprocess_routes_summary.json"
    aggregate["summary_file"] = str(aggregate_path)
    write_json(aggregate_path, aggregate)
    return aggregate


def run(args: argparse.Namespace) -> int:
    if normalize_n_cpu(getattr(args, "n_cpu", 1)) != 1:
        print(
            "[preprocess-routes] --n-cpu is accepted for CLI consistency, "
            "but preprocessing currently runs sequentially.",
            file=sys.stderr,
            flush=True,
        )
    summary = preprocess_datasets(args)
    print(json.dumps(summary, indent=2), flush=True)
    return 0
