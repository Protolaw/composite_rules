from __future__ import annotations

import json
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if __package__ in (None, "") and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alchems.composite_rules.extract import (
    RouteProcessingStats,
    SynPlannerRuleExtractor,
    extract_route_composites,
    route_items,
)
from alchems.io import (
    expand_composite_rule_tsv_paths,
    read_json,
    read_tsv_rows,
    resolve_existing_path,
    setup_runtime_cache_dirs,
    split_cell,
    write_json,
    write_tsv,
)


@dataclass
class CompositeRuleSet:
    source: str
    popularity_by_rule: dict[str, int]
    references_by_rule: dict[str, set[str]]
    rows_seen: int = 0
    parse_errors: int = 0

    @property
    def rules(self) -> set[str]:
        return set(self.popularity_by_rule)

    @property
    def total_popularity(self) -> int:
        return sum(self.popularity_by_rule.values())


@dataclass
class CompositeRuleClassificationSet:
    source: str
    positive_weight_by_rule: dict[str, int]
    negative_weight_by_rule: dict[str, int]
    rows_seen: int = 0
    parse_errors: int = 0

    def weights(self, rule: str) -> tuple[int, int]:
        return (
            self.positive_weight_by_rule.get(rule, 0),
            self.negative_weight_by_rule.get(rule, 0),
        )


def ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def normalize_composite_rule(rule: str) -> str:
    parts = [part.strip() for part in rule.split("$") if part.strip()]
    if len(parts) < 2:
        raise ValueError("composite rule must contain at least two rule SMARTS")
    return "$".join(parts)


def popularity_from_row(row: dict[str, str]) -> int:
    for column in ("popularity", "route_ids_size"):
        value = row.get(column, "").strip()
        if value.isdigit():
            return int(value)
    references = split_cell(row.get("Reference"))
    return max(len(references), 1)


def split_composite_rules_cell(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(" || ") if part.strip()]


def classification_from_row(row: dict[str, str], path: Path) -> str | None:
    value = row.get("classification", "").strip().lower()
    if value in {"positive", "pos"}:
        return "positive"
    if value in {"negative", "neg"}:
        return "negative"

    stem = path.stem.lower()
    if stem.endswith("_pos") or stem.endswith("_positive"):
        return "positive"
    if stem.endswith("_neg") or stem.endswith("_negative"):
        return "negative"
    return None


def expand_classification_tsv_paths(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = resolve_existing_path(raw_path)
        if path.is_dir():
            candidate_paths = sorted(
                candidate
                for candidate in path.glob("*_classified_alchemical_rules.tsv")
                if not candidate.stem.endswith(("_pos", "_neg"))
            )
            if not candidate_paths:
                candidate_paths = sorted(path.glob("*.tsv"))
        else:
            candidate_paths = [path]

        for candidate in candidate_paths:
            key = candidate.resolve() if candidate.exists() else candidate
            if key not in seen:
                seen.add(key)
                expanded.append(candidate)
    return expanded


def empty_classification_set() -> CompositeRuleClassificationSet:
    return CompositeRuleClassificationSet(
        source="",
        positive_weight_by_rule={},
        negative_weight_by_rule={},
    )


def load_composite_rule_classifications(
    paths: list[Path] | None,
) -> CompositeRuleClassificationSet:
    if not paths:
        return empty_classification_set()

    positive_weight_by_rule: dict[str, int] = defaultdict(int)
    negative_weight_by_rule: dict[str, int] = defaultdict(int)
    rows_seen = 0
    parse_errors = 0
    resolved_paths = expand_classification_tsv_paths(paths)

    for path in resolved_paths:
        fieldnames, rows = read_tsv_rows(path)
        if "Composite_rules" not in fieldnames:
            raise ValueError(f"{path} has no Composite_rules column")
        for row in rows:
            rows_seen += 1
            classification = classification_from_row(row, path)
            if classification is None:
                parse_errors += 1
                continue
            weight = popularity_from_row(row)
            for raw_rule in split_composite_rules_cell(row.get("Composite_rules")):
                try:
                    rule = normalize_composite_rule(raw_rule)
                except ValueError:
                    parse_errors += 1
                    continue
                if classification == "positive":
                    positive_weight_by_rule[rule] += weight
                else:
                    negative_weight_by_rule[rule] += weight

    return CompositeRuleClassificationSet(
        source=",".join(str(path) for path in resolved_paths),
        positive_weight_by_rule=dict(positive_weight_by_rule),
        negative_weight_by_rule=dict(negative_weight_by_rule),
        rows_seen=rows_seen,
        parse_errors=parse_errors,
    )


def load_extracted_composite_rule_set(paths: list[Path]) -> CompositeRuleSet:
    popularity_by_rule: dict[str, int] = defaultdict(int)
    references_by_rule: dict[str, set[str]] = defaultdict(set)
    rows_seen = 0
    parse_errors = 0
    resolved_paths = expand_composite_rule_tsv_paths(paths)

    for path in resolved_paths:
        fieldnames, rows = read_tsv_rows(path)
        if "Composite_rule" not in fieldnames:
            raise ValueError(f"{path} has no Composite_rule column")
        for row in rows:
            rows_seen += 1
            try:
                rule = normalize_composite_rule(row.get("Composite_rule", ""))
            except ValueError:
                parse_errors += 1
                continue
            popularity_by_rule[rule] += popularity_from_row(row)
            references_by_rule[rule].update(split_cell(row.get("Reference")))

    return CompositeRuleSet(
        source=",".join(str(path) for path in resolved_paths),
        popularity_by_rule=dict(popularity_by_rule),
        references_by_rule={rule: set(refs) for rule, refs in references_by_rule.items()},
        rows_seen=rows_seen,
        parse_errors=parse_errors,
    )


def reference_composite_rules_from_routes(
    routes_json_path: Path,
    rule_extractor: SynPlannerRuleExtractor,
    *,
    min_length: int,
    max_length: int | None,
    limit: int | None = None,
    ignore_errors: bool = False,
    progress_interval: int = 250,
) -> tuple[CompositeRuleSet, RouteProcessingStats, list[dict[str, Any]]]:
    routes_json_path = resolve_existing_path(routes_json_path)
    routes_json = read_json(routes_json_path)
    popularity_by_rule: dict[str, int] = defaultdict(int)
    references_by_rule: dict[str, set[str]] = defaultdict(set)
    errors: list[dict[str, Any]] = []
    stats = RouteProcessingStats()

    for index, (route_id, route) in enumerate(route_items(routes_json), start=1):
        if limit is not None and index > limit:
            break
        stats.routes_seen += 1
        try:
            route_sequences = extract_route_composites(
                route,
                rule_extractor,
                min_length=min_length,
                max_length=max_length,
                stats=stats,
            )
            if route_sequences:
                stats.routes_with_composites += 1
            for sequence in route_sequences:
                rule = "$".join(sequence)
                popularity_by_rule[rule] += 1
                references_by_rule[rule].add(str(route_id))
        except Exception as exc:
            stats.errors += 1
            errors.append(
                {
                    "route_id": route_id,
                    "stage": "extract_reference_composite_rules",
                    "error_type": type(exc).__qualname__,
                    "message": str(exc) or traceback.format_exc(limit=1).strip(),
                }
            )
            if not ignore_errors:
                raise

        if progress_interval and index % progress_interval == 0:
            print(
                f"processed reference routes={index} "
                f"reference_composite_rules={len(popularity_by_rule)} "
                f"errors={stats.errors}",
                flush=True,
            )

    return (
        CompositeRuleSet(
            source=str(routes_json_path),
            popularity_by_rule=dict(popularity_by_rule),
            references_by_rule={
                rule: set(refs) for rule, refs in references_by_rule.items()
            },
            rows_seen=stats.routes_seen,
            parse_errors=stats.errors,
        ),
        stats,
        errors,
    )


def overlap_score_row(
    extracted: CompositeRuleSet,
    reference: CompositeRuleSet,
    classifications: CompositeRuleClassificationSet | None = None,
) -> dict[str, Any]:
    overlap = extracted.rules & reference.rules
    union = extracted.rules | reference.rules
    overlapping_popularity = sum(extracted.popularity_by_rule[rule] for rule in overlap)
    classifications = classifications or empty_classification_set()

    positive_overlap_popularity = 0.0
    negative_overlap_popularity = 0.0
    classified_overlap_rules = 0
    positive_overlap_rules = 0
    negative_overlap_rules = 0
    mixed_overlap_rules = 0
    for rule in overlap:
        positive_weight, negative_weight = classifications.weights(rule)
        classification_weight = positive_weight + negative_weight
        if classification_weight == 0:
            continue

        classified_overlap_rules += 1
        if positive_weight:
            positive_overlap_rules += 1
        if negative_weight:
            negative_overlap_rules += 1
        if positive_weight and negative_weight:
            mixed_overlap_rules += 1

        extracted_popularity = extracted.popularity_by_rule[rule]
        positive_overlap_popularity += extracted_popularity * ratio(
            positive_weight,
            classification_weight,
        )
        negative_overlap_popularity += extracted_popularity * ratio(
            negative_weight,
            classification_weight,
        )

    return {
        "extracted_tsv": extracted.source,
        "reference_routes_json": reference.source,
        "classification_tsv": classifications.source,
        "extracted_rows": extracted.rows_seen,
        "reference_routes": reference.rows_seen,
        "classification_rows": classifications.rows_seen,
        "extracted_unique_composite_rules": len(extracted.rules),
        "reference_unique_composite_rules": len(reference.rules),
        "overlap_unique_composite_rules": len(overlap),
        "classified_overlap_unique_composite_rules": classified_overlap_rules,
        "positive_overlap_unique_composite_rules": positive_overlap_rules,
        "negative_overlap_unique_composite_rules": negative_overlap_rules,
        "mixed_classification_overlap_unique_composite_rules": mixed_overlap_rules,
        "extracted_overlap_ratio": ratio(len(overlap), len(extracted.rules)),
        "reference_coverage_ratio": ratio(len(overlap), len(reference.rules)),
        "jaccard": ratio(len(overlap), len(union)),
        "extracted_popularity": extracted.total_popularity,
        "overlapping_popularity": overlapping_popularity,
        "popularity_overlap_ratio": ratio(
            overlapping_popularity,
            extracted.total_popularity,
        ),
        "positive_overlapping_popularity": positive_overlap_popularity,
        "negative_overlapping_popularity": negative_overlap_popularity,
        "pos_overlap": ratio(
            positive_overlap_popularity,
            extracted.total_popularity,
        ),
        "neg_overlap": ratio(
            negative_overlap_popularity,
            extracted.total_popularity,
        ),
        "extracted_parse_errors": extracted.parse_errors,
        "reference_errors": reference.parse_errors,
        "classification_parse_errors": classifications.parse_errors,
    }


def matched_rule_rows(
    extracted: CompositeRuleSet,
    reference: CompositeRuleSet,
    classifications: CompositeRuleClassificationSet | None = None,
) -> list[dict[str, Any]]:
    rows = []
    classifications = classifications or empty_classification_set()
    for rule in sorted(
        extracted.rules,
        key=lambda item: (-extracted.popularity_by_rule[item], item),
    ):
        reference_ids = sorted(reference.references_by_rule.get(rule, set()))
        positive_weight, negative_weight = classifications.weights(rule)
        classification_weight = positive_weight + negative_weight
        if positive_weight and negative_weight:
            classification = "mixed"
        elif positive_weight:
            classification = "positive"
        elif negative_weight:
            classification = "negative"
        else:
            classification = "unclassified"
        rows.append(
            {
                "Composite_rule": rule,
                "present_in_reference": bool(reference_ids),
                "classification": classification,
                "classification_positive_weight": positive_weight,
                "classification_negative_weight": negative_weight,
                "classification_positive_share": ratio(
                    positive_weight,
                    classification_weight,
                ),
                "classification_negative_share": ratio(
                    negative_weight,
                    classification_weight,
                ),
                "extracted_popularity": extracted.popularity_by_rule[rule],
                "extracted_reference_ids": ",".join(
                    sorted(extracted.references_by_rule.get(rule, set()))
                ),
                "reference_route_ids": ",".join(reference_ids),
            }
        )
    return rows


def output_paths(output: Path) -> tuple[Path, Path, Path]:
    if output.is_dir() or output.suffix == "":
        return (
            output / "composite_rule_overlap_scores.tsv",
            output / "composite_rule_overlap_matches.tsv",
            output / "composite_rule_overlap_summary.json",
        )
    prefix = output.with_suffix("")
    return (
        output,
        prefix.with_name(f"{prefix.name}_matches").with_suffix(".tsv"),
        prefix.with_name(f"{prefix.name}_summary").with_suffix(".json"),
    )


def score_composite_rule_overlap(
    extracted_tsvs: list[Path],
    reference_routes_json: Path,
    output: Path,
    rule_extractor: SynPlannerRuleExtractor,
    *,
    min_length: int = 2,
    max_length: int | None = 5,
    limit: int | None = None,
    ignore_errors: bool = False,
    progress_interval: int = 250,
    classification_tsvs: list[Path] | None = None,
) -> dict[str, Any]:
    extracted = load_extracted_composite_rule_set(extracted_tsvs)
    classifications = load_composite_rule_classifications(classification_tsvs)
    reference, reference_stats, errors = reference_composite_rules_from_routes(
        reference_routes_json,
        rule_extractor,
        min_length=min_length,
        max_length=max_length,
        limit=limit,
        ignore_errors=ignore_errors,
        progress_interval=progress_interval,
    )
    score_row = overlap_score_row(extracted, reference, classifications)
    score_path, matches_path, summary_path = output_paths(output)

    score_fieldnames = [
        "extracted_tsv",
        "reference_routes_json",
        "classification_tsv",
        "extracted_rows",
        "reference_routes",
        "classification_rows",
        "extracted_unique_composite_rules",
        "reference_unique_composite_rules",
        "overlap_unique_composite_rules",
        "classified_overlap_unique_composite_rules",
        "positive_overlap_unique_composite_rules",
        "negative_overlap_unique_composite_rules",
        "mixed_classification_overlap_unique_composite_rules",
        "extracted_overlap_ratio",
        "reference_coverage_ratio",
        "jaccard",
        "extracted_popularity",
        "overlapping_popularity",
        "popularity_overlap_ratio",
        "positive_overlapping_popularity",
        "negative_overlapping_popularity",
        "pos_overlap",
        "neg_overlap",
        "extracted_parse_errors",
        "reference_errors",
        "classification_parse_errors",
    ]
    write_tsv(score_path, score_fieldnames, [score_row])

    match_fieldnames = [
        "Composite_rule",
        "present_in_reference",
        "classification",
        "classification_positive_weight",
        "classification_negative_weight",
        "classification_positive_share",
        "classification_negative_share",
        "extracted_popularity",
        "extracted_reference_ids",
        "reference_route_ids",
    ]
    write_tsv(
        matches_path,
        match_fieldnames,
        matched_rule_rows(extracted, reference, classifications),
    )

    summary = {
        "extracted_tsvs": [str(resolve_existing_path(path)) for path in extracted_tsvs],
        "reference_routes_json": str(resolve_existing_path(reference_routes_json)),
        "classification_tsvs": [
            str(resolve_existing_path(path)) for path in classification_tsvs or []
        ],
        "output": str(score_path),
        "matches_output": str(matches_path),
        "summary_file": str(summary_path),
        "min_length": min_length,
        "max_length": max_length,
        "reference_routes_seen": reference_stats.routes_seen,
        "reference_routes_with_composite_rules": reference_stats.routes_with_composites,
        "reference_reactions_seen": reference_stats.reactions_seen,
        "reference_skipped_reactions": reference_stats.skipped_reactions,
        "reference_errors": reference_stats.errors,
        "reference_error_examples": errors[:25],
        **score_row,
    }
    write_json(summary_path, summary)
    return summary


def run(args: Any) -> int:
    setup_runtime_cache_dirs()
    if args.min_length < 2:
        raise ValueError("--min-length must be at least 2")
    if args.max_length is not None and args.max_length <= 0:
        args.max_length = None
    if args.max_length is not None and args.max_length < args.min_length:
        raise ValueError("--max-length must be greater than or equal to --min-length")

    rule_extractor = SynPlannerRuleExtractor.from_args(args)
    summary = score_composite_rule_overlap(
        [resolve_existing_path(path) for path in args.extracted_tsv],
        resolve_existing_path(args.reference_routes_json),
        args.output,
        rule_extractor,
        min_length=args.min_length,
        max_length=args.max_length,
        limit=args.limit,
        ignore_errors=args.ignore_errors,
        progress_interval=args.progress_interval,
        classification_tsvs=(
            [resolve_existing_path(path) for path in args.classification_tsv]
            if args.classification_tsv
            else None
        ),
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0
