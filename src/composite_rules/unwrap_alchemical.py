from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

PACKAGE_SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent

if __package__ in (None, "") and str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from agents.composite_rules.src.composite_rules.unwrap import unwrap_rule_sequence, write_json


def resolve_existing_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.exists() or path.is_absolute():
        return path

    candidates = [Path.cwd() / path, PROJECT_ROOT / path, WORKSPACE_ROOT / path]
    if path.parts and path.parts[0] == "composite_rules":
        candidates.append(PROJECT_ROOT.joinpath(*path.parts[1:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def add_import_paths(*paths: str | Path | None) -> None:
    for path in paths:
        if path is None:
            continue
        path = resolve_existing_path(path)
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def read_alchemical_rule_from_tsv(path: Path, row_index: int) -> str:
    with path.open(encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        fieldnames = reader.fieldnames or []
        rule_column = None
        for candidate in ("Alchemical_rule", "Alchemical_rules"):
            if candidate in fieldnames:
                rule_column = candidate
                break
        if rule_column is None:
            raise ValueError(f"{path} has no Alchemical_rule column")
        for index, row in enumerate(reader):
            if index == row_index:
                return row[rule_column]
    raise IndexError(f"row index {row_index} not found in {path}")


def unwrap_alchemical_rule(
    target_smiles: str,
    alchemical_rule: str,
    *,
    route_id: int = 0,
    mark_leaves_in_stock: bool = True,
) -> dict[int, dict[str, Any]]:
    return unwrap_rule_sequence(
        target_smiles,
        [alchemical_rule],
        route_id=route_id,
        rule_key_prefix="alchemical",
        mark_leaves_in_stock=mark_leaves_in_stock,
    ).routes_json


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    add_import_paths(args.synplanner_root)

    alchemical_rule = args.alchemical_rule
    if alchemical_rule is None:
        alchemical_rule = read_alchemical_rule_from_tsv(
            resolve_existing_path(args.alchemical_rule_tsv),
            args.row,
        )

    routes_json = unwrap_alchemical_rule(
        args.smiles,
        alchemical_rule,
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
        description="Apply one alchemical rule to a target molecule."
    )
    parser.add_argument("--smiles", required=True, help="Target molecule SMILES.")
    rule_source = parser.add_mutually_exclusive_group(required=True)
    rule_source.add_argument("--alchemical-rule", help="Alchemical rule SMARTS.")
    rule_source.add_argument(
        "--alchemical-rule-tsv",
        type=Path,
        help="TSV containing an Alchemical_rule column.",
    )
    parser.add_argument("--row", type=int, default=0, help="0-based TSV row index.")
    parser.add_argument("--route-id", type=int, default=0)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-svg", type=Path, default=None)
    parser.add_argument("--labeled", action="store_true")
    parser.add_argument("--do-not-mark-leaves-in-stock", action="store_true")
    parser.add_argument("--synplanner-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
