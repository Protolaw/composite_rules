from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def add_import_paths(*paths: str | Path | None) -> None:
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def split_composite_rule(composite_rule: str) -> list[str]:
    rules = [part.strip() for part in composite_rule.split("$") if part.strip()]
    if not rules:
        raise ValueError("Composite rule is empty")
    return rules


def read_composite_rule_from_tsv(path: Path, row_index: int) -> str:
    with path.open(encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        if "Composite_rule" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no Composite_rule column")
        for index, row in enumerate(reader):
            if index == row_index:
                return row["Composite_rule"]
    raise IndexError(f"row index {row_index} not found in {path}")


@dataclass
class UnwrapResult:
    routes_json: dict[int, dict[str, Any]]
    target_molecule: Any
    leaf_molecules: list[Any]
    reactions: list[Any]


class RuleApplicationError(ValueError):
    """Raised when an extracted rule sequence cannot be applied to a molecule."""


def first_retro_reaction(reactor: Any, molecule: Any) -> Any | None:
    for reaction in reactor(molecule):
        return reaction
    return None


def select_retro_reaction(
    reactor: Any,
    molecule: Any,
    next_reactor: Any | None = None,
) -> tuple[Any, list[Any], list[tuple[int, Any]]] | None:
    for reaction in reactor(molecule):
        products = list(reaction.products)
        if not products:
            continue
        if next_reactor is None:
            return reaction, products, []

        next_candidates = []
        for product_index, product in enumerate(products):
            if first_retro_reaction(next_reactor, product) is not None:
                next_candidates.append((product_index, product))
        if next_candidates:
            return reaction, products, next_candidates

    return None


def molecule_node(molecule: Any, *, in_stock: bool = False) -> dict[str, Any]:
    return {"type": "mol", "smiles": str(molecule), "in_stock": in_stock}


def mark_leaf_molecules_in_stock(node: dict[str, Any]) -> None:
    if node.get("type") == "mol" and not node.get("children"):
        node["in_stock"] = True
        return
    for child in node.get("children", []) or []:
        if isinstance(child, dict):
            mark_leaf_molecules_in_stock(child)


def unwrap_composite_rule(
    target_smiles: str,
    composite_rule: str,
    *,
    route_id: int = 0,
    mark_leaves_in_stock: bool = True,
) -> dict[int, dict[str, Any]]:
    return unwrap_rule_sequence(
        target_smiles,
        split_composite_rule(composite_rule),
        route_id=route_id,
        rule_key_prefix="composite",
        mark_leaves_in_stock=mark_leaves_in_stock,
    ).routes_json


def unwrap_rule_sequence(
    target_smiles: str,
    rule_smarts: list[str],
    *,
    route_id: int = 0,
    rule_key_prefix: str = "rule",
    mark_leaves_in_stock: bool = True,
) -> UnwrapResult:
    from chython import smiles as parse_smiles
    from chython.reactor import Reactor

    reactors = [
        Reactor.from_smarts(rule, delete_atoms=False, one_shot=True)
        for rule in rule_smarts
    ]

    target_molecule = parse_smiles(target_smiles)
    root = molecule_node(target_molecule, in_stock=False)
    active_node = root
    active_molecule = target_molecule
    node_molecules = {id(root): target_molecule}
    reactions = []

    for step_index, reactor in enumerate(reactors):
        next_reactor = (
            reactors[step_index + 1] if step_index < len(reactors) - 1 else None
        )
        selected = select_retro_reaction(reactor, active_molecule, next_reactor)
        if selected is None:
            raise RuleApplicationError(
                f"rule {step_index + 1} did not match active molecule {active_molecule}"
            )

        reaction, products, next_candidates = selected
        reactions.append(reaction)
        child_nodes = [molecule_node(product, in_stock=False) for product in products]
        node_molecules.update(
            {
                id(child_node): product
                for child_node, product in zip(child_nodes, products)
            }
        )
        reaction_node = {
            "type": "reaction",
            "smiles": format(reaction, "m"),
            "rule_key": f"{rule_key_prefix}:{step_index + 1}",
            "children": child_nodes,
        }
        active_node["children"] = [reaction_node]

        if step_index == len(reactors) - 1:
            break

        if not next_candidates:
            raise RuleApplicationError(
                f"rule {step_index + 2} did not match any reactant produced by "
                f"rule {step_index + 1}"
            )
        if len(next_candidates) > 1:
            # Deterministic first-match behavior keeps the unwrapped route as a
            # single route. The JSON keeps all sibling precursors from each step.
            pass

        product_index, active_molecule = next_candidates[0]
        active_node = child_nodes[product_index]

    if mark_leaves_in_stock:
        mark_leaf_molecules_in_stock(root)

    leaf_molecules = []

    def collect_leaf_molecules(node: dict[str, Any]) -> None:
        if node.get("type") == "mol" and not node.get("children"):
            leaf_molecules.append(node_molecules[id(node)])
            return
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                collect_leaf_molecules(child)

    collect_leaf_molecules(root)

    return UnwrapResult(
        routes_json={route_id: root},
        target_molecule=target_molecule,
        leaf_molecules=leaf_molecules,
        reactions=reactions,
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    add_import_paths(args.synplanner_root)

    composite_rule = args.composite_rule
    if composite_rule is None:
        composite_rule = read_composite_rule_from_tsv(args.composite_rule_tsv, args.row)

    routes_json = unwrap_composite_rule(
        args.smiles,
        composite_rule,
        route_id=args.route_id,
        mark_leaves_in_stock=not args.do_not_mark_leaves_in_stock,
    )

    if args.output_json:
        write_json(args.output_json, routes_json)
    else:
        print(json.dumps(routes_json, indent=2))

    if args.output_svg:
        from synplan.utils.visualisation import get_route_svg_from_json

        svg = get_route_svg_from_json(routes_json, args.route_id, labeled=args.labeled)
        args.output_svg.parent.mkdir(parents=True, exist_ok=True)
        args.output_svg.write_text(svg, encoding="utf-8")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequentially apply a composite rule to unwrap a target molecule."
    )
    parser.add_argument("--smiles", required=True, help="Target molecule SMILES.")
    rule_source = parser.add_mutually_exclusive_group(required=True)
    rule_source.add_argument("--composite-rule", help="Composite rule string.")
    rule_source.add_argument(
        "--composite-rule-tsv",
        type=Path,
        help="TSV containing a Composite_rule column.",
    )
    parser.add_argument("--row", type=int, default=0, help="0-based TSV row index.")
    parser.add_argument("--route-id", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-svg", type=Path, default=None)
    parser.add_argument("--labeled", action="store_true")
    parser.add_argument("--do-not-mark-leaves-in-stock", action="store_true")
    parser.add_argument(
        "--synplanner-root",
        type=Path,
        default=None,
        help="Optional path containing the synplan package, e.g. SynPlanner.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
