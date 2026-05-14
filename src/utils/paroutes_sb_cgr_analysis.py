from __future__ import annotations

import argparse
import json
import os
import pickle
import traceback
from collections import Counter
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from chython import smiles as smiles_chython
from synplan.chem.reaction_routes.route_cgr import compose_sb_cgr

from composite_rules.src.composite_rules.my_utils.cgr_process import (
    extract_all_route_cgrs,
    filter_unique_routes,
    normalise_route_tree_for_chython,
)


def _load_routes(path: Path, limit: int | None = None):
    with path.open() as file:
        routes = json.load(file)
    if limit is not None:
        routes = routes[:limit] if isinstance(routes, list) else dict(list(routes.items())[:limit])
    return routes


def _record_tree(record):
    return record["dict"] if isinstance(record, dict) and "dict" in record else record


def _record_id(record, fallback):
    if isinstance(record, dict):
        return record.get("route_id", fallback)
    return fallback


def _target_atom_count(record):
    tree = _record_tree(record)
    return len(smiles_chython(tree["smiles"]))


def _strip_tracebacks(errors):
    for error in errors:
        error.pop("traceback", None)
    return errors


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2)


def run(args):
    routes = _load_routes(args.routes_json, args.limit)
    input_count = len(routes)

    filtered_records = filter_unique_routes(routes)
    record_by_id = {
        _record_id(record, idx): record for idx, record in enumerate(filtered_records)
    }

    if args.converted_json:
        converted = {
            str(route_id): normalise_route_tree_for_chython(_record_tree(record))
            for route_id, record in record_by_id.items()
        }
        _write_json(args.converted_json, converted)

    route_cgrs_dict, errors = extract_all_route_cgrs(
        filtered_records,
        check_trans_error=True,
        collect_errors=True,
        progress_interval=args.progress_interval,
    )

    sb_cgrs_dict = {}
    for processed, (route_id, route_cgr) in enumerate(route_cgrs_dict.items(), start=1):
        try:
            sb_cgrs_dict[route_id] = compose_sb_cgr(route_cgr)
        except Exception as exc:
            tree = _record_tree(record_by_id.get(route_id, {}))
            errors.append(
                {
                    "route_id": route_id,
                    "stage": "compose_sb_cgr",
                    "target_smiles": tree.get("smiles") if isinstance(tree, dict) else None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
        if args.progress_interval and processed % args.progress_interval == 0:
            print(
                f"composed {processed} SB-CGR candidates; sb_cgrs={len(sb_cgrs_dict)} errors={len(errors)}",
                flush=True,
            )

    for route_id, sb_cgr in sb_cgrs_dict.items():
        record = record_by_id[route_id]
        tree = _record_tree(record)
        try:
            target_atoms = _target_atom_count(record)
        except Exception as exc:
            errors.append(
                {
                    "route_id": route_id,
                    "stage": "parse_target",
                    "target_smiles": tree.get("smiles"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            continue

        sb_atoms = len(sb_cgr)
        if sb_atoms != target_atoms:
            errors.append(
                {
                    "route_id": route_id,
                    "stage": "validate_sb_cgr_atom_count",
                    "target_smiles": tree.get("smiles"),
                    "target_atoms": target_atoms,
                    "sb_cgr_atoms": sb_atoms,
                    "error": "SB-CGR atom count does not match target molecule atom count",
                }
            )

    if not args.keep_tracebacks:
        errors = _strip_tracebacks(errors)

    errors_by_stage = Counter(error["stage"] for error in errors)
    mismatch_ids = [
        error["route_id"]
        for error in errors
        if error["stage"] == "validate_sb_cgr_atom_count"
    ]
    summary = {
        "input_json": str(args.routes_json),
        "input_routes": input_count,
        "unique_routes": len(filtered_records),
        "duplicates_removed": input_count - len(filtered_records),
        "route_cgrs": len(route_cgrs_dict),
        "sb_cgrs": len(sb_cgrs_dict),
        "errors": len(errors),
        "errors_by_stage": dict(sorted(errors_by_stage.items())),
        "atom_count_mismatches": len(mismatch_ids),
        "atom_count_mismatch_route_ids": mismatch_ids[:100],
    }

    summary_path = args.output_prefix.with_name(args.output_prefix.name + "_summary.json")
    errors_path = args.output_prefix.with_name(args.output_prefix.name + "_errors.json")
    _write_json(summary_path, summary)
    _write_json(errors_path, errors)

    if args.pickle_cgrs:
        route_cgrs_path = args.output_prefix.with_name(
            args.output_prefix.name + "_route_cgrs.pkl"
        )
        sb_cgrs_path = args.output_prefix.with_name(args.output_prefix.name + "_sb_cgrs.pkl")
        with route_cgrs_path.open("wb") as file:
            pickle.dump(route_cgrs_dict, file)
        with sb_cgrs_path.open("wb") as file:
            pickle.dump(sb_cgrs_dict, file)
        summary["route_cgrs_pickle"] = str(route_cgrs_path)
        summary["sb_cgrs_pickle"] = str(sb_cgrs_path)
        _write_json(summary_path, summary)

    print(json.dumps(summary, indent=2), flush=True)
    return 1 if args.fail_on_errors and errors else 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build RouteCGR/SB-CGR objects for PaRoutes JSON and validate atom counts."
    )
    parser.add_argument(
        "--routes-json",
        type=Path,
        default=Path("PaRoutes/data/n1-routes.json"),
        help="PaRoutes route JSON file.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("data/paroutes_n1_sb_cgr"),
        help="Prefix for *_summary.json and *_errors.json outputs.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N route limit.")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=250,
        help="Print progress every N processed routes; use 0 to silence.",
    )
    parser.add_argument(
        "--converted-json",
        type=Path,
        default=None,
        help="Optional path for a chython-readable JSON copy with reaction smiles filled in.",
    )
    parser.add_argument(
        "--pickle-cgrs",
        action="store_true",
        help="Also pickle RouteCGR and SB-CGR dictionaries.",
    )
    parser.add_argument(
        "--keep-tracebacks",
        action="store_true",
        help="Keep full Python tracebacks in the error report.",
    )
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Exit non-zero if any processing or validation errors are collected.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
