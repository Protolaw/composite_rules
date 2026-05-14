from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PACKAGE_SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent

if __package__ in (None, "") and str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from agents.composite_rules.src.composite_rules.alchemical import rule_cgr_key


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


def output_base(output: Path, suffix: str) -> Path:
    prefix = output.with_suffix("")
    base_name = re.sub(r"_classified_alchemical_rules$", "", prefix.name)
    base_name = re.sub(r"_alchemical_rules$", "", base_name)
    return prefix.with_name(f"{base_name}_{suffix}")


def default_output_path(alchemical_rules_tsv: Path) -> Path:
    return output_base(alchemical_rules_tsv, "classified_alchemical_rules").with_suffix(
        ".tsv"
    )


def default_summary_path(output: Path) -> Path:
    return output_base(output, "alchemical_rule_classification_summary").with_suffix(
        ".json"
    )


def rule_column(fieldnames: list[str], preferred: tuple[str, ...]) -> str:
    for candidate in preferred:
        if candidate in fieldnames:
            return candidate
    if fieldnames:
        return fieldnames[0]
    raise ValueError("TSV header is empty")


def load_default_rule_cgrs(
    rules_tsv: Path,
) -> tuple[dict[str, list[tuple[int, str]]], int, int]:
    default_rules: dict[str, list[tuple[int, str]]] = defaultdict(list)
    parsed = 0
    errors = 0
    with rules_tsv.open(encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        fieldnames = reader.fieldnames or []
        column = rule_column(fieldnames, ("rule_smarts", "Rule", "SMARTS"))
        for index, row in enumerate(reader):
            smarts = row.get(column, "").strip()
            if not smarts:
                continue
            try:
                default_rules[rule_cgr_key(smarts)].append((index, smarts))
                parsed += 1
            except Exception:
                errors += 1
    return default_rules, parsed, errors


def classify_alchemical_rules(
    alchemical_rules_tsv: Path,
    default_rules_tsv: Path,
    output: Path,
    summary_path: Path,
) -> dict[str, Any]:
    default_cgrs, default_rules_parsed, default_rule_errors = load_default_rule_cgrs(
        default_rules_tsv
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    positive = 0
    negative = 0
    rows_seen = 0
    cgr_errors = 0

    with (
        alchemical_rules_tsv.open(encoding="utf-8") as input_file,
        output.open("w", newline="", encoding="utf-8") as output_file,
    ):
        reader = csv.DictReader(input_file, delimiter="\t")
        input_fieldnames = reader.fieldnames or []
        alchemical_column = rule_column(
            input_fieldnames,
            ("Alchemical_rule", "Alchemical_rules", "rule_smarts"),
        )
        fieldnames = input_fieldnames + [
            "classification",
            "Matched_default_rule_ids",
            "Matched_default_rules",
        ]
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()

        for row in reader:
            rows_seen += 1
            cgr_key = row.get("Alchemical_cgr", "").strip()
            try:
                if not cgr_key:
                    cgr_key = rule_cgr_key(row[alchemical_column])
            except Exception:
                cgr_errors += 1
                cgr_key = ""

            matches = default_cgrs.get(cgr_key, []) if cgr_key else []
            if matches:
                classification = "negative"
                negative += 1
            else:
                classification = "positive"
                positive += 1

            row["classification"] = classification
            row["Matched_default_rule_ids"] = ",".join(
                str(index) for index, _smarts in matches
            )
            row["Matched_default_rules"] = " || ".join(
                smarts for _index, smarts in matches
            )
            writer.writerow(row)

    summary = {
        "alchemical_rules_tsv": str(alchemical_rules_tsv),
        "default_rules_tsv": str(default_rules_tsv),
        "output": str(output),
        "rows_seen": rows_seen,
        "positive": positive,
        "negative": negative,
        "default_rules_parsed": default_rules_parsed,
        "default_rule_parse_errors": default_rule_errors,
        "alchemical_cgr_errors": cgr_errors,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    summary["summary_file"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def run(args: argparse.Namespace) -> int:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    add_import_paths(args.synplanner_root)
    output = args.output or default_output_path(args.alchemical_rules_tsv)
    summary_path = args.summary or default_summary_path(output)
    summary = classify_alchemical_rules(
        resolve_existing_path(args.alchemical_rules_tsv),
        resolve_existing_path(args.default_rules_tsv),
        output,
        summary_path,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify alchemical rules as negative if their QueryCGR matches a "
            "default non-alchemical rule, otherwise positive."
        )
    )
    parser.add_argument("--alchemical-rules-tsv", type=Path, required=True)
    parser.add_argument("--default-rules-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--synplanner-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
