from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __package__ in (None, "") and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alchems.alchemical_rules.alchemical import rule_cgr_key
from alchems.io import (
    default_classification_output_path,
    default_classification_summary_path,
    read_tsv_rows,
    resolve_classification_output_paths,
    resolve_existing_path,
    rule_column,
    setup_runtime_cache_dirs,
    write_json,
    write_tsv,
)


def load_default_rule_cgrs(
    rules_tsv: Path,
) -> tuple[dict[str, list[tuple[int, str]]], int, int]:
    default_rules: dict[str, list[tuple[int, str]]] = defaultdict(list)
    parsed = 0
    errors = 0
    fieldnames, rows = read_tsv_rows(rules_tsv)
    column = rule_column(fieldnames, ("rule_smarts", "Rule", "SMARTS"))
    for index, row in enumerate(rows):
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
    alchemical_rules_tsv = resolve_existing_path(alchemical_rules_tsv)
    default_rules_tsv = resolve_existing_path(default_rules_tsv)
    output, summary_path = resolve_classification_output_paths(
        alchemical_rules_tsv,
        output,
        summary_path,
    )
    default_cgrs, default_rules_parsed, default_rule_errors = load_default_rule_cgrs(
        default_rules_tsv
    )
    positive = 0
    negative = 0
    rows_seen = 0
    cgr_errors = 0

    input_fieldnames, input_rows = read_tsv_rows(alchemical_rules_tsv)
    alchemical_column = rule_column(
        input_fieldnames,
        ("Alchemical_rule", "Alchemical_rules", "rule_smarts"),
    )
    fieldnames = list(input_fieldnames)
    for column in (
        "classification",
        "Matched_default_rule_ids",
        "Matched_default_rules",
    ):
        if column not in fieldnames:
            fieldnames.append(column)

    output_rows = []
    positive_rows = []
    negative_rows = []
    for row in input_rows:
        rows_seen += 1
        try:
            cgr_key = rule_cgr_key(row[alchemical_column])
        except Exception:
            cgr_errors += 1
            cgr_key = row.get("Alchemical_cgr", "").strip()

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
        output_rows.append(row)
        if classification == "positive":
            positive_rows.append(row.copy())
        else:
            negative_rows.append(row.copy())

    write_tsv(output, fieldnames, output_rows)
    positive_output = output.with_name(f"{output.stem}_pos{output.suffix or '.tsv'}")
    negative_output = output.with_name(f"{output.stem}_neg{output.suffix or '.tsv'}")
    write_tsv(positive_output, fieldnames, positive_rows)
    write_tsv(negative_output, fieldnames, negative_rows)

    summary = {
        "alchemical_rules_tsv": str(alchemical_rules_tsv),
        "default_rules_tsv": str(default_rules_tsv),
        "output": str(output),
        "positive_output": str(positive_output),
        "negative_output": str(negative_output),
        "rows_seen": rows_seen,
        "positive": positive,
        "negative": negative,
        "default_rules_parsed": default_rules_parsed,
        "default_rule_parse_errors": default_rule_errors,
        "alchemical_cgr_errors": cgr_errors,
    }
    write_json(summary_path, summary)
    summary["summary_file"] = str(summary_path)
    write_json(summary_path, summary)
    return summary


def run(args: argparse.Namespace) -> int:
    setup_runtime_cache_dirs()
    output = args.output or default_classification_output_path(args.alchemical_rules_tsv)
    summary_path = args.summary or (
        args.output if args.output else default_classification_summary_path(output)
    )
    summary = classify_alchemical_rules(
        resolve_existing_path(args.alchemical_rules_tsv),
        resolve_existing_path(args.default_rules_tsv),
        output,
        summary_path,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0
