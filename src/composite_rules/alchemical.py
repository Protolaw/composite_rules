from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

PACKAGE_SRC = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent

if __package__ in (None, "") and str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from agents.composite_rules.src.composite_rules.unwrap import (
    RuleApplicationError,
    split_composite_rule,
    unwrap_rule_sequence,
)


@dataclass(frozen=True)
class CompositeRuleApplication:
    source_tsv: Path
    row_index: int
    composite_rule: str
    composite_size: int
    route_ids: tuple[str, ...]
    target_smiles: str


@dataclass(frozen=True)
class ExtractedAlchemicalRule:
    rule_smarts: str
    cgr_key: str


@dataclass
class PseudoReactionRecord:
    pseudo_reaction_id: str
    alchemical_cgr: str
    reaction_smiles: str
    source_tsv: str
    source_row: int
    route_ids: tuple[str, ...]
    target_smiles: str
    composite_size: int
    composite_rule: str


@dataclass
class AlchemicalRuleAggregate:
    rule_smarts: str
    cgr_key: str
    route_ids: set[str] = field(default_factory=set)
    target_molecules: set[str] = field(default_factory=set)
    composite_rules: set[str] = field(default_factory=set)
    composite_sizes: set[int] = field(default_factory=set)
    source_rows: set[str] = field(default_factory=set)
    pseudo_reaction_ids: list[str] = field(default_factory=list)


@dataclass
class AlchemicalCollectionStats:
    composite_rows_seen: int = 0
    applications_seen: int = 0
    pseudo_reactions_built: int = 0
    alchemical_rules_extracted: int = 0
    skipped_unwrap_applications: int = 0
    skipped_rule_extractions: int = 0
    skipped_rule_extraction_errors: int = 0
    errors: int = 0


def add_import_paths(*paths: str | Path | None) -> None:
    for path in paths:
        if path is None:
            continue
        path = resolve_existing_path(path)
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


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


def setup_runtime_cache_dirs() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def split_cell(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def reference_sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, str) and value.isdigit():
        return (0, int(value))
    return (1, str(value))


def is_standardization_error(exc: Exception) -> bool:
    return type(exc).__name__ == "StandardizationError"


def output_base(output: Path, suffix: str) -> Path:
    prefix = output.with_suffix("")
    base_name = re.sub(r"_alchemical_rules$", "", prefix.name)
    return prefix.with_name(f"{base_name}_{suffix}")


def is_directory_path(path: Path) -> bool:
    return path.is_dir() or path.suffix == ""


def composite_output_stem(composite_rule_tsvs: Iterable[Path]) -> str:
    stems = []
    for path in composite_rule_tsvs:
        match = re.match(r"(.+)_t\d+_composite_rules$", path.stem)
        if match:
            stems.append(match.group(1))

    unique_stems = sorted(set(stems))
    if len(unique_stems) == 1:
        return unique_stems[0]
    if unique_stems:
        return "merged"
    return "alchemical"


def resolve_optional_sidecar_path(
    path: Path | None,
    output_dir: Path,
    filename: str,
) -> Path:
    if path is None:
        return output_dir / filename
    if is_directory_path(path):
        return path / filename
    return path


def resolve_alchemical_output_paths(
    output: Path,
    composite_rule_tsvs: list[Path],
    *,
    output_smi: Path | None = None,
    summary: Path | None = None,
    errors: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    if is_directory_path(output):
        stem = composite_output_stem(composite_rule_tsvs)
        output_dir = output
        rules_path = output_dir / f"{stem}_alchemical_rules.tsv"
        smi_path = resolve_optional_sidecar_path(
            output_smi,
            output_dir,
            f"{stem}_alchemical_reactions.smi",
        )
        summary_path = resolve_optional_sidecar_path(
            summary,
            output_dir,
            f"{stem}_alchemical_rule_collection_summary.json",
        )
        error_path = resolve_optional_sidecar_path(
            errors,
            output_dir,
            f"{stem}_alchemical_rule_collection_errors.tsv",
        )
        return rules_path, smi_path, summary_path, error_path

    rules_path = output
    smi_path = output_smi or default_smi_path(rules_path)
    summary_path = summary or default_summary_path(rules_path)
    error_path = errors or default_error_path(rules_path)
    return rules_path, smi_path, summary_path, error_path


def default_smi_path(output: Path) -> Path:
    return output_base(output, "alchemical_reactions").with_suffix(".smi")


def default_summary_path(output: Path) -> Path:
    return output_base(output, "alchemical_rule_collection_summary").with_suffix(
        ".json"
    )


def default_error_path(output: Path) -> Path:
    return output_base(output, "alchemical_rule_collection_errors").with_suffix(".tsv")


def iter_composite_rule_applications(
    tsv_paths: Iterable[Path],
) -> Iterable[CompositeRuleApplication]:
    for tsv_path in tsv_paths:
        with tsv_path.open(encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t")
            fieldnames = reader.fieldnames or []
            if "Composite_rule" not in fieldnames:
                raise ValueError(f"{tsv_path} has no Composite_rule column")
            if "Target_molecules" not in fieldnames:
                raise ValueError(f"{tsv_path} has no Target_molecules column")

            for row_index, row in enumerate(reader):
                composite_rule = row["Composite_rule"].strip()
                if not composite_rule:
                    continue
                route_ids = tuple(split_cell(row.get("Reference")))
                targets = split_cell(row.get("Target_molecules"))
                composite_size = len(split_composite_rule(composite_rule))
                for target_smiles in targets:
                    yield CompositeRuleApplication(
                        source_tsv=tsv_path,
                        row_index=row_index,
                        composite_rule=composite_rule,
                        composite_size=composite_size,
                        route_ids=route_ids,
                        target_smiles=target_smiles,
                    )


def expand_composite_rule_tsv_paths(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = resolve_existing_path(raw_path)
        if path.is_dir():
            matches = sorted(path.glob("*_composite_rules.tsv"))
            if not matches:
                matches = sorted(path.glob("*.tsv"))
            candidate_paths = matches
        else:
            candidate_paths = [path]

        for candidate in candidate_paths:
            key = candidate.resolve() if candidate.exists() else candidate
            if key not in seen:
                seen.add(key)
                expanded.append(candidate)
    return expanded


def compose_pseudo_reaction_smiles(
    target_smiles: str,
    composite_rule: str,
) -> str:
    from chython.containers.reaction import ReactionContainer

    unwrapped = unwrap_rule_sequence(
        target_smiles,
        split_composite_rule(composite_rule),
        route_id=0,
        rule_key_prefix="composite",
        mark_leaves_in_stock=True,
    )
    reactants, product = normalize_pseudo_reaction_mapping(
        unwrapped.leaf_molecules,
        unwrapped.target_molecule,
    )
    pseudo_reaction = ReactionContainer(
        reactants=tuple(reactants),
        products=(product,),
    )
    return format(pseudo_reaction, "m")


def normalize_pseudo_reaction_mapping(
    reactants: Iterable[Any],
    product: Any,
) -> tuple[list[Any], Any]:
    """Make atom numbers unique across pseudo-reaction reactants.

    Chython template applications preserve useful atom numbers for atoms that
    continue into the target, but generated leaving-group atoms from independent
    steps can reuse the same numbers. Rule extraction expects atom numbers to be
    globally unique on each side unless they denote the same atom, so remap only
    the colliding or element-incompatible reactant atoms.
    """

    product = product.copy()
    product_atoms = {
        atom_number: atom.atomic_number for atom_number, atom in product.atoms()
    }
    used_reactant_numbers: set[int] = set()
    next_atom_number = max(product_atoms, default=0) + 1
    normalized_reactants = []

    for reactant in reactants:
        reactant = reactant.copy()
        remapping: dict[int, int] = {}
        for atom_number, atom in reactant.atoms():
            product_atomic_number = product_atoms.get(atom_number)
            can_keep_number = (
                atom_number not in used_reactant_numbers
                and (
                    product_atomic_number is None
                    or product_atomic_number == atom.atomic_number
                )
            )
            if can_keep_number:
                used_reactant_numbers.add(atom_number)
                continue

            while (
                next_atom_number in product_atoms
                or next_atom_number in used_reactant_numbers
            ):
                next_atom_number += 1
            remapping[atom_number] = next_atom_number
            used_reactant_numbers.add(next_atom_number)
            next_atom_number += 1

        if remapping:
            reactant.remap(remapping)
        normalized_reactants.append(reactant)

    return normalized_reactants, product


def rule_cgr_key(rule_smarts: str) -> str:
    from chython import smarts
    from chython.containers.reaction import ReactionContainer
    from chython.reactor import Reactor

    try:
        return str(~smarts(rule_smarts))
    except Exception:
        reactor = Reactor.from_smarts(rule_smarts, delete_atoms=False)
        reaction = ReactionContainer(
            reactor.__dict__["_patterns"],
            reactor.__dict__["_products"],
        )
        return str(~reaction)


class AlchemicalRuleExtractor:
    def __init__(self, config: Any):
        self.config = config
        self.cache: dict[str, ExtractedAlchemicalRule | None] = {}

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AlchemicalRuleExtractor":
        from synplan.utils.config import RuleExtractionConfig

        if args.config:
            config = RuleExtractionConfig.from_yaml(
                str(resolve_existing_path(args.config))
            )
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

    def extract(self, reaction_smiles: str) -> ExtractedAlchemicalRule | None:
        if reaction_smiles in self.cache:
            return self.cache[reaction_smiles]

        from chython import smiles as parse_smiles
        from synplan.chem.reaction_rules.extraction import (
            _rule_to_reactor_smarts,
            extract_rules,
        )

        reaction = parse_smiles(reaction_smiles)
        rules, skipped = extract_rules(self.config, reaction)
        if skipped or not rules:
            self.cache[reaction_smiles] = None
            return None
        if len(rules) != 1:
            raise ValueError(f"expected one alchemical rule, got {len(rules)}")

        rule = rules[0]
        extracted = ExtractedAlchemicalRule(
            rule_smarts=_rule_to_reactor_smarts(rule),
            cgr_key=str(~rule),
        )
        self.cache[reaction_smiles] = extracted
        return extracted


def collect_alchemical_rules(
    composite_rule_tsvs: list[Path],
    extractor: AlchemicalRuleExtractor,
    *,
    limit_rows: int | None = None,
    limit_applications: int | None = None,
    ignore_errors: bool = False,
    progress_interval: int = 250,
) -> tuple[
    dict[str, AlchemicalRuleAggregate],
    list[PseudoReactionRecord],
    AlchemicalCollectionStats,
    list[dict[str, Any]],
]:
    aggregates: dict[str, AlchemicalRuleAggregate] = {}
    pseudo_reactions: list[PseudoReactionRecord] = []
    errors: list[dict[str, Any]] = []
    stats = AlchemicalCollectionStats()
    rows_seen: set[tuple[Path, int]] = set()

    for application in iter_composite_rule_applications(composite_rule_tsvs):
        if (
            limit_applications is not None
            and stats.applications_seen >= limit_applications
        ):
            break

        row_key = (application.source_tsv, application.row_index)
        if row_key not in rows_seen:
            if limit_rows is not None and stats.composite_rows_seen >= limit_rows:
                break
            rows_seen.add(row_key)
            stats.composite_rows_seen += 1

        stats.applications_seen += 1

        try:
            pseudo_reaction_smiles = compose_pseudo_reaction_smiles(
                application.target_smiles,
                application.composite_rule,
            )
            stats.pseudo_reactions_built += 1
            extracted = extractor.extract(pseudo_reaction_smiles)
            if extracted is None:
                stats.skipped_rule_extractions += 1
                continue
            stats.alchemical_rules_extracted += 1

            pseudo_reaction_id = f"p{len(pseudo_reactions)}"
            pseudo_reactions.append(
                PseudoReactionRecord(
                    pseudo_reaction_id=pseudo_reaction_id,
                    alchemical_cgr=extracted.cgr_key,
                    reaction_smiles=pseudo_reaction_smiles,
                    source_tsv=str(application.source_tsv),
                    source_row=application.row_index,
                    route_ids=application.route_ids,
                    target_smiles=application.target_smiles,
                    composite_size=application.composite_size,
                    composite_rule=application.composite_rule,
                )
            )

            aggregate = aggregates.get(extracted.cgr_key)
            if aggregate is None:
                aggregate = AlchemicalRuleAggregate(
                    rule_smarts=extracted.rule_smarts,
                    cgr_key=extracted.cgr_key,
                )
                aggregates[extracted.cgr_key] = aggregate

            aggregate.route_ids.update(application.route_ids)
            aggregate.target_molecules.add(application.target_smiles)
            aggregate.composite_rules.add(application.composite_rule)
            aggregate.composite_sizes.add(application.composite_size)
            aggregate.source_rows.add(
                f"{application.source_tsv.name}:{application.row_index}"
            )
            aggregate.pseudo_reaction_ids.append(pseudo_reaction_id)
        except Exception as exc:
            if isinstance(exc, RuleApplicationError):
                stats.skipped_unwrap_applications += 1
                continue
            if is_standardization_error(exc):
                stats.skipped_rule_extraction_errors += 1
                continue

            stats.errors += 1
            error = {
                "source_tsv": str(application.source_tsv),
                "row_index": application.row_index,
                "target_smiles": application.target_smiles,
                "error_type": type(exc).__qualname__,
                "message": str(exc) or traceback.format_exc(limit=1).strip(),
            }
            errors.append(error)
            if not ignore_errors:
                raise

        if progress_interval and stats.applications_seen % progress_interval == 0:
            print(
                "processed applications="
                f"{stats.applications_seen} alchemical_rules={len(aggregates)} "
                f"skipped_unwrap={stats.skipped_unwrap_applications} "
                f"errors={stats.errors}",
                flush=True,
            )

    return aggregates, pseudo_reactions, stats, errors


def sorted_aggregates(
    aggregates: dict[str, AlchemicalRuleAggregate],
) -> list[AlchemicalRuleAggregate]:
    return sorted(
        aggregates.values(),
        key=lambda aggregate: (
            -len(aggregate.route_ids),
            -len(aggregate.pseudo_reaction_ids),
            aggregate.rule_smarts,
        ),
    )


def write_alchemical_rules_tsv(
    output: Path,
    aggregates: dict[str, AlchemicalRuleAggregate],
) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Alchemical_rule",
        "popularity",
        "route_ids_size",
        "Reference",
        "Target_molecules",
        "composite_rules_size",
        "Composite_rule_sizes",
        "Composite_rules",
        "Source_composite_rows",
        "pseudo_reactions_size",
        "Pseudo_reaction_ids",
        "Alchemical_cgr",
    ]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for aggregate in sorted_aggregates(aggregates):
            route_ids = sorted(aggregate.route_ids, key=reference_sort_key)
            writer.writerow(
                {
                    "Alchemical_rule": aggregate.rule_smarts,
                    "popularity": len(route_ids),
                    "route_ids_size": len(route_ids),
                    "Reference": ",".join(route_ids),
                    "Target_molecules": ",".join(sorted(aggregate.target_molecules)),
                    "composite_rules_size": len(aggregate.composite_rules),
                    "Composite_rule_sizes": ",".join(
                        map(str, sorted(aggregate.composite_sizes))
                    ),
                    "Composite_rules": " || ".join(sorted(aggregate.composite_rules)),
                    "Source_composite_rows": ",".join(sorted(aggregate.source_rows)),
                    "pseudo_reactions_size": len(aggregate.pseudo_reaction_ids),
                    "Pseudo_reaction_ids": ",".join(aggregate.pseudo_reaction_ids),
                    "Alchemical_cgr": aggregate.cgr_key,
                }
            )
    return {"alchemical_rules": len(aggregates)}


def write_pseudo_reactions_smi(
    output: Path,
    pseudo_reactions: list[PseudoReactionRecord],
    aggregates: dict[str, AlchemicalRuleAggregate],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    alchemical_rule_ids = {
        aggregate.cgr_key: f"a{index}"
        for index, aggregate in enumerate(sorted_aggregates(aggregates))
    }
    with output.open("w", encoding="utf-8") as file:
        for record in pseudo_reactions:
            file.write(
                "\t".join(
                    [
                        record.reaction_smiles,
                        record.pseudo_reaction_id,
                        alchemical_rule_ids[record.alchemical_cgr],
                        ",".join(record.route_ids),
                        record.target_smiles,
                        str(record.composite_size),
                        f"{Path(record.source_tsv).name}:{record.source_row}",
                    ]
                )
                + "\n"
            )


def write_errors(path: Path, errors: list[dict[str, Any]]) -> None:
    if not errors:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_tsv",
                "row_index",
                "target_smiles",
                "error_type",
                "message",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(errors)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


def run(args: argparse.Namespace) -> int:
    setup_runtime_cache_dirs()
    add_import_paths(args.synplanner_root)
    composite_rule_tsvs = expand_composite_rule_tsv_paths(args.composite_rule_tsv)

    extractor = AlchemicalRuleExtractor.from_args(args)
    aggregates, pseudo_reactions, stats, errors = collect_alchemical_rules(
        composite_rule_tsvs,
        extractor,
        limit_rows=args.limit_rows,
        limit_applications=args.limit_applications,
        ignore_errors=args.ignore_errors,
        progress_interval=args.progress_interval,
    )

    rules_path, smi_path, summary_path, error_path = resolve_alchemical_output_paths(
        args.output,
        composite_rule_tsvs,
        output_smi=args.output_smi,
        summary=args.summary,
        errors=args.errors,
    )

    output_stats = write_alchemical_rules_tsv(rules_path, aggregates)
    write_pseudo_reactions_smi(smi_path, pseudo_reactions, aggregates)
    write_errors(error_path, errors)

    summary = {
        "composite_rule_tsv": [str(path) for path in composite_rule_tsvs],
        "output": str(rules_path),
        "pseudo_reactions_smi": str(smi_path),
        "errors_file": str(error_path) if errors else None,
        "composite_rows_seen": stats.composite_rows_seen,
        "applications_seen": stats.applications_seen,
        "pseudo_reactions_built": stats.pseudo_reactions_built,
        "alchemical_rules_extracted": stats.alchemical_rules_extracted,
        "skipped_unwrap_applications": stats.skipped_unwrap_applications,
        "skipped_rule_extractions": stats.skipped_rule_extractions,
        "skipped_rule_extraction_errors": stats.skipped_rule_extraction_errors,
        "errors": stats.errors,
        "unique_alchemical_rules": len(aggregates),
        **output_stats,
    }
    write_summary(summary_path, summary)
    summary["summary_file"] = str(summary_path)
    write_summary(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collapse composite-rule unwrappings into pseudo-reactions and "
            "extract alchemical rules."
        )
    )
    parser.add_argument(
        "--composite-rule-tsv",
        type=Path,
        nargs="+",
        required=True,
        help="One or more composite rule TSV files or directories containing them.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Output TSV file or output directory. When a directory is given, "
            "<prefix>_alchemical_rules.tsv and sidecar files are written there."
        ),
    )
    parser.add_argument(
        "--output-smi",
        "--output_smi",
        type=Path,
        default=None,
        dest="output_smi",
        help=(
            "Optional pseudo-reaction .smi file or directory. If omitted and "
            "--output is a directory, the .smi is written into that directory."
        ),
    )
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--errors", type=Path, default=None)
    parser.add_argument("--synplanner-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--limit-applications", type=int, default=None)
    parser.add_argument("--environment-atom-count", type=int, default=1)
    parser.add_argument("--include-rings", action="store_true")
    parser.add_argument("--keep-leaving-groups", action="store_true", default=True)
    parser.add_argument(
        "--drop-leaving-groups",
        dest="keep_leaving_groups",
        action="store_false",
    )
    parser.add_argument("--keep-incoming-groups", action="store_true")
    parser.add_argument("--reactor-validation", action="store_true")
    parser.add_argument("--ignore-errors", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=250)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
