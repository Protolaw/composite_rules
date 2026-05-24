from __future__ import annotations

import csv
import gzip
import json
import os
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

STAGE_DIRS = {
    "preprocess": "00_preprocess",
    "composite_rules": "10_composite_rules",
    "alchemical_rules": "20_alchemical_rules",
    "alchemical_classification": "30_alchemical_classification",
    "scoring": "40_scoring",
    "protection_analysis": "50_protection_analysis",
}


def increase_csv_field_limit() -> None:
    """Raise the CSV parser field limit for large SMARTS and route fields.

    Keeping this helper in the IO layer avoids duplicating low-level file and table
    behavior in the chemistry-focused modules.
    """
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


increase_csv_field_limit()


@dataclass(frozen=True)
class CompositeRuleApplication:
    source_tsv: Path
    row_index: int
    composite_rule: str
    composite_size: int
    route_ids: tuple[str, ...]
    target_smiles: str


def setup_runtime_cache_dirs() -> None:
    """Create runtime cache directories used by plotting and chemistry dependencies.

    Keeping this helper in the IO layer avoids duplicating low-level file and table
    behavior in the chemistry-focused modules.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
    os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def normalize_n_cpu(n_cpu: int | None) -> int:
    """Normalize n CPU for route-inspector processing.

    Keeping this helper in the IO layer avoids duplicating low-level file and table
    behavior in the chemistry-focused modules.
    """
    if n_cpu is None:
        return 1
    n_cpu = int(n_cpu)
    if n_cpu == 0:
        return os.cpu_count() or 1
    return max(1, n_cpu)


def resolve_existing_path(path: str | Path) -> Path:
    """Resolve output path information for resolve existing path.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    path = Path(path).expanduser()
    if path.exists() or path.is_absolute():
        return path

    candidates = [Path.cwd() / path, PROJECT_ROOT / path, WORKSPACE_ROOT / path]
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        candidates.append(PROJECT_ROOT.joinpath(*path.parts[1:]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def open_text(path: Path):
    """Open a plain-text or gzip-compressed file for reading.

    Keeping this helper in the IO layer avoids duplicating low-level file and table
    behavior in the chemistry-focused modules.
    """
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read JSON data from a plain or gzip-compressed file.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    with open_text(path) as file:
        return json.load(file)


def read_tsv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a TSV header and row dictionaries from a tab-separated file.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    increase_csv_field_limit()
    with path.open(encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        return reader.fieldnames or [], list(reader)


def write_tsv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    """Write TSV to disk.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    """Write JSON to disk.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def json_safe(value: Any) -> Any:
    """Return a JSON-serializable representation of common CLI values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def dataset_prefix_from_path(path: str | Path) -> str:
    """Infer a compact dataset prefix from route or artifact paths."""
    path = Path(path)
    stem = path.stem
    for pattern in (
        r"[-_]?routes$",
        r"_classified_alchemical_rules$",
        r"_alchemical_rules$",
        r"_alchemical_reactions$",
        r"_alchemical_rule_collection_summary$",
        r"_alchemical_rule_collection_errors$",
        r"_alchemical_rule_classification_summary$",
        r"_composite_rule_extraction_summary$",
        r"_routes_without_composite_rules$",
        r"_t\d+_composite_rules$",
        r"_t\d+_single_rules$",
        r"_protection_.+$",
    ):
        stem = re.sub(pattern, "", stem)
    return stem or path.stem or "dataset"


def stage_output_dir(output_root: Path, dataset: str, stage: str) -> Path:
    """Return the standard dataset-first, stage-second output directory."""
    stage_name = STAGE_DIRS.get(stage, stage)
    return Path(output_root).expanduser() / dataset / stage_name


def resolve_output_path(
    *,
    output: Path | None,
    output_dir: Path | None,
    default_filename: str,
) -> Path:
    """Resolve explicit file output or directory-based default output path."""
    if output is not None:
        return Path(output).expanduser()
    if output_dir is None:
        raise ValueError("either --output or --output-dir is required")
    return Path(output_dir).expanduser() / default_filename


def git_commit_hash() -> str:
    """Return the current git commit hash when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def route_inspector_version() -> str:
    """Return installed package version when available."""
    try:
        return metadata.version("route-inspector")
    except Exception:
        return ""


def cli_command_text(argv: list[str] | None = None) -> str:
    """Return a shell-escaped command line for reproducibility metadata."""
    argv = list(sys.argv if argv is None else argv)
    return " ".join(shlex.quote(part) for part in argv)


def write_manifest(
    output_dir: Path,
    *,
    command_name: str,
    input_files: Iterable[str | Path] = (),
    output_files: Iterable[str | Path] | dict[str, str | Path] = (),
    config_path: str | Path | None = None,
    cli_args: Any | None = None,
    argv: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a standard manifest sidecar for a command output directory."""
    output_dir = Path(output_dir)
    if isinstance(output_files, dict):
        output_files_value = {
            key: str(value) for key, value in sorted(output_files.items())
        }
    else:
        output_files_value = [str(path) for path in output_files]
    manifest = {
        "command_name": command_name,
        "full_cli_command": cli_command_text(argv),
        "input_files": [str(path) for path in input_files],
        "output_directory": str(output_dir),
        "output_files": output_files_value,
        "config_path": str(config_path) if config_path else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "route_inspector_version": route_inspector_version(),
        "git_commit_hash": git_commit_hash(),
        "cli_args": json_safe(vars(cli_args)) if cli_args is not None else None,
        **(extra or {}),
    }
    path = output_dir / "manifest.json"
    write_json(path, manifest)
    return path


def write_errors_tsv(
    path: Path,
    errors: Iterable[dict[str, Any]],
    *,
    fieldnames: list[str] | None = None,
) -> Path:
    """Write generic tabular error sidecar, including an empty file header."""
    rows = list(errors)
    base_fields = fieldnames or ["route_id", "stage", "error_type", "message"]
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in base_fields
        }
    )
    all_fields = base_fields + extra_fields
    write_tsv(path, all_fields, rows)
    return path


def write_summary_sidecar(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Write generic summary.json sidecar beside command-specific summaries."""
    path = Path(output_dir) / "summary.json"
    write_json(path, summary)
    return path


def write_standard_sidecars(
    output_dir: Path,
    *,
    command_name: str,
    summary: dict[str, Any],
    errors: Iterable[dict[str, Any]] = (),
    input_files: Iterable[str | Path] = (),
    output_files: Iterable[str | Path] | dict[str, str | Path] = (),
    config_path: str | Path | None = None,
    cli_args: Any | None = None,
    argv: list[str] | None = None,
    extra_manifest: dict[str, Any] | None = None,
    write_summary: bool = True,
) -> dict[str, str]:
    """Write generic summary, errors, and manifest sidecars for a command."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = (
        write_summary_sidecar(output_dir, summary) if write_summary else None
    )
    errors_path = write_errors_tsv(output_dir / "errors.tsv", errors)
    manifest_path = write_manifest(
        output_dir,
        command_name=command_name,
        input_files=input_files,
        output_files=output_files,
        config_path=config_path,
        cli_args=cli_args,
        argv=argv,
        extra=extra_manifest,
    )
    sidecars = {
        "errors": str(errors_path),
        "manifest": str(manifest_path),
    }
    if summary_path is not None:
        sidecars["summary"] = str(summary_path)
    return sidecars


def resolve_file_or_dir_path(path: Path, default_filename: str) -> Path:
    """Resolve output path information for resolve file or dir path.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    if path.is_dir() or path.suffix == "":
        return path / default_filename
    return path


def split_cell(value: str | None) -> list[str]:
    """Split cell into normalized values.

    Keeping this helper in the IO layer avoids duplicating low-level file and table
    behavior in the chemistry-focused modules.
    """
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def reference_sort_key(value: Any) -> tuple[int, Any]:
    """Return a deterministic sort key for route references.

    Keeping this helper in the IO layer avoids duplicating low-level file and table
    behavior in the chemistry-focused modules.
    """
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, str) and value.isdigit():
        return (0, int(value))
    return (1, str(value))


def rule_column(fieldnames: list[str], preferred: tuple[str, ...]) -> str:
    """Return rule column used for rule extraction or comparison.

    Keeping this helper in the IO layer avoids duplicating low-level file and table
    behavior in the chemistry-focused modules.
    """
    for candidate in preferred:
        if candidate in fieldnames:
            return candidate
    if fieldnames:
        return fieldnames[0]
    raise ValueError("TSV header is empty")


def read_rule_from_tsv(
    path: Path,
    row_index: int,
    *,
    columns: tuple[str, ...],
) -> str:
    """Read a selected rule string from a TSV row.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    with path.open(encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")
        fieldnames = reader.fieldnames or []
        column = None
        for candidate in columns:
            if candidate in fieldnames:
                column = candidate
                break
        if column is None:
            raise ValueError(f"{path} has none of these columns: {', '.join(columns)}")
        for index, row in enumerate(reader):
            if index == row_index:
                return row[column]
    raise IndexError(f"row index {row_index} not found in {path}")


def read_composite_rule_from_tsv(path: Path, row_index: int) -> str:
    """Read a selected composite rule string from a TSV row.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    return read_rule_from_tsv(path, row_index, columns=("Composite_rule",))


def read_alchemical_rule_from_tsv(path: Path, row_index: int) -> str:
    """Read a selected alchemical rule string from a TSV row.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    return read_rule_from_tsv(
        path,
        row_index,
        columns=("Alchemical_rule", "Alchemical_rules"),
    )


def write_composite_rules(
    output: Path,
    references_by_sequence: dict[tuple[str, ...], set[Any]],
    target_molecules_by_sequence: (
        dict[tuple[str, ...], dict[Any, set[str]]] | None
    ) = None,
) -> dict[str, Any]:
    """Write composite rules to disk.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    output_prefix = output.with_suffix("")
    output_suffix = output.suffix or ".tsv"
    output_paths: dict[int, Path] = {}
    counts_by_size: dict[int, int] = {}

    sequences_by_size: dict[int, list[tuple[tuple[str, ...], set[Any]]]] = (
        defaultdict(list)
    )
    for sequence, references in references_by_sequence.items():
        sequences_by_size[len(sequence)].append((sequence, references))

    for size in sorted(sequences_by_size):
        rows = sorted(
            sequences_by_size[size],
            key=lambda item: (-len(item[1]), "$".join(item[0])),
        )
        if size == 1:
            size_output = output_prefix.with_name(
                f"{output_prefix.name}_t1_single_rules"
            ).with_suffix(output_suffix)
            header = ["Rule", "popularity", "Reference", "Target_molecules"]
        else:
            size_output = output_prefix.with_name(
                f"{output_prefix.name}_t{size}_composite_rules"
            ).with_suffix(output_suffix)
            header = [
                "Composite_rule",
                "output_reactants_num",
                "popularity",
                "Reference",
                "Target_molecules",
            ]
        output_paths[size] = size_output
        counts_by_size[size] = len(rows)
        with size_output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file, delimiter="\t", lineterminator="\n")
            writer.writerow(header)
            for sequence, references in rows:
                sorted_references = sorted(references, key=reference_sort_key)
                target_molecules: list[str] = []
                seen_target_molecules: set[str] = set()
                if target_molecules_by_sequence:
                    route_targets = target_molecules_by_sequence.get(sequence, {})
                    for route_id in sorted_references:
                        for target_smiles in sorted(route_targets.get(route_id, set())):
                            if (
                                target_smiles
                                and target_smiles not in seen_target_molecules
                            ):
                                seen_target_molecules.add(target_smiles)
                                target_molecules.append(target_smiles)
                if size == 1:
                    writer.writerow(
                        [
                            sequence[0],
                            len(references),
                            ",".join(map(str, sorted_references)),
                            ",".join(target_molecules),
                        ]
                    )
                else:
                    writer.writerow(
                        [
                            "$".join(sequence),
                            composite_output_reactants_num(sequence),
                            len(references),
                            ",".join(map(str, sorted_references)),
                            ",".join(target_molecules),
                        ]
                    )

    return {
        "output_files": {str(size): str(path) for size, path in output_paths.items()},
        "unique_rules_by_size": {
            str(size): counts_by_size[size] for size in sorted(counts_by_size)
        },
        "unique_single_rules": counts_by_size.get(1, 0),
        "unique_composite_rules_by_size": {
            str(size): counts_by_size[size]
            for size in sorted(counts_by_size)
            if size > 1
        },
    }


def composite_summary_path_from_output(output: Path) -> Path:
    """Build the composite summary path from an output prefix.

    The returned path follows the project naming conventions so downstream commands can
    discover related summaries and error files automatically.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.with_suffix("")
    base_name = re.sub(r"(?:_t\d+)?_composite_rules$", "", prefix.name)
    return prefix.with_name(f"{base_name}_composite_rule_extraction_summary.json")


def composite_routes_without_rules_path_from_output(output: Path) -> Path:
    """Build the composite routes without rules path from an output prefix.

    The returned path follows the project naming conventions so downstream commands can
    discover related summaries and error files automatically.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.with_suffix("")
    base_name = re.sub(r"(?:_t\d+)?_composite_rules$", "", prefix.name)
    return prefix.with_name(f"{base_name}_routes_without_composite_rules.json")


def write_composite_summary(output: Path, summary: dict[str, Any]) -> Path:
    """Write composite summary to disk.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    path = composite_summary_path_from_output(output)
    write_json(path, summary)
    return path


def write_composite_routes_without_rules(
    output: Path,
    routes: dict[Any, dict[str, Any]],
    path: Path | None = None,
) -> Path:
    """Write composite routes without rules to disk.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    no_rules_path = path or composite_routes_without_rules_path_from_output(output)
    write_json(
        no_rules_path,
        {str(route_id): route for route_id, route in routes.items()},
    )
    return no_rules_path


def write_composite_errors(output: Path, errors: list[dict[str, Any]]) -> None:
    """Write composite errors to disk.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    if not errors:
        return
    path = output.with_suffix(output.suffix + ".errors.tsv")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["route_id", "stage", "error_type", "message"]
    write_tsv(path, fieldnames, errors)


def output_base(output: Path, suffix: str) -> Path:
    """Build the normalized output stem shared by related artifacts.

    The returned path follows the project naming conventions so downstream commands can
    discover related summaries and error files automatically.
    """
    prefix = output.with_suffix("")
    base_name = re.sub(r"_classified_alchemical_rules$", "", prefix.name)
    base_name = re.sub(r"_alchemical_rules$", "", base_name)
    return prefix.with_name(f"{base_name}_{suffix}")


def default_smi_path(output: Path) -> Path:
    """Resolve output path information for default SMI path.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    return output_base(output, "alchemical_reactions").with_suffix(".smi")


def default_alchemical_summary_path(output: Path) -> Path:
    """Resolve output path information for default alchemical summary path.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    return output_base(output, "alchemical_rule_collection_summary").with_suffix(
        ".json"
    )


def default_alchemical_error_path(output: Path) -> Path:
    """Resolve output path information for default alchemical error path.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    return output_base(output, "alchemical_rule_collection_errors").with_suffix(".tsv")


def default_classification_output_path(alchemical_rules_tsv: Path) -> Path:
    """Resolve output path information for default classification output path.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    return output_base(alchemical_rules_tsv, "classified_alchemical_rules").with_suffix(
        ".tsv"
    )


def default_classification_summary_path(output: Path) -> Path:
    """Resolve output path information for default classification summary path.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    return output_base(output, "alchemical_rule_classification_summary").with_suffix(
        ".json"
    )


def resolve_classification_output_paths(
    alchemical_rules_tsv: Path,
    output: Path,
    summary: Path,
) -> tuple[Path, Path]:
    """Resolve output path information for resolve classification output paths.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    base_name = re.sub(r"_alchemical_rules$", "", alchemical_rules_tsv.stem)
    output_path = resolve_file_or_dir_path(
        output,
        f"{base_name}_classified_alchemical_rules.tsv",
    )
    summary_path = resolve_file_or_dir_path(
        summary,
        f"{base_name}_alchemical_rule_classification_summary.json",
    )
    return output_path, summary_path


def is_directory_path(path: Path) -> bool:
    """Resolve output path information for is directory path.

    The returned path follows the project naming conventions so downstream commands can
    discover related summaries and error files automatically.
    """
    return path.is_dir() or path.suffix == ""


def composite_output_stem(composite_rule_tsvs: Iterable[Path]) -> str:
    """Infer the shared output stem from composite-rule TSV paths.

    The returned path follows the project naming conventions so downstream commands can
    discover related summaries and error files automatically.
    """
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
    """Resolve output path information for resolve optional sidecar path.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
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
    """Resolve output path information for resolve alchemical output paths.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    if is_directory_path(output):
        stem = composite_output_stem(composite_rule_tsvs)
        output_dir = output
        return (
            output_dir / f"{stem}_alchemical_rules.tsv",
            resolve_optional_sidecar_path(
                output_smi,
                output_dir,
                f"{stem}_alchemical_reactions.smi",
            ),
            resolve_optional_sidecar_path(
                summary,
                output_dir,
                f"{stem}_alchemical_rule_collection_summary.json",
            ),
            resolve_optional_sidecar_path(
                errors,
                output_dir,
                f"{stem}_alchemical_rule_collection_errors.tsv",
            ),
        )

    rules_path = output
    smi_path = output_smi or default_smi_path(rules_path)
    summary_path = summary or default_alchemical_summary_path(rules_path)
    error_path = errors or default_alchemical_error_path(rules_path)
    return rules_path, smi_path, summary_path, error_path


def iter_composite_rule_applications(
    tsv_paths: Iterable[Path],
) -> Iterable[CompositeRuleApplication]:
    """Yield composite-rule applications from one or more TSV files.

    Keeping this helper in the IO layer avoids duplicating low-level file and table
    behavior in the chemistry-focused modules.
    """
    from route_inspector.composite_rules.unwrap import split_composite_rule

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
    """Resolve output path information for expand composite rule TSV paths.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    expanded: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = resolve_existing_path(raw_path)
        candidate_paths = sorted(path.glob("*_composite_rules.tsv")) if path.is_dir() else [path]
        if path.is_dir() and not candidate_paths:
            candidate_paths = sorted(path.glob("*.tsv"))

        for candidate in candidate_paths:
            key = candidate.resolve() if candidate.exists() else candidate
            if key not in seen:
                seen.add(key)
                expanded.append(candidate)
    return expanded


def sorted_aggregates(aggregates: dict[str, Any]) -> list[Any]:
    """Return aggregate records sorted by popularity and rule text.

    Keeping this helper in the IO layer avoids duplicating low-level file and table
    behavior in the chemistry-focused modules.
    """
    return sorted(
        aggregates.values(),
        key=lambda aggregate: (
            -len(aggregate.route_ids),
            -len(aggregate.pseudo_reaction_ids),
            aggregate.rule_smarts,
        ),
    )


def reaction_output_reactants_num(rule_smarts: str) -> int:
    """Return reaction output reactants num from a reaction or route node.

    The returned path follows the project naming conventions so downstream commands can
    discover related summaries and error files automatically.
    """

    _left, separator, right = rule_smarts.partition(">>")
    if not separator:
        return 0
    return len([part for part in right.split(".") if part.strip()])


def composite_output_reactants_num(sequence: Iterable[str]) -> int:
    """Count output reactants for the last rule in a composite sequence.

    The returned path follows the project naming conventions so downstream commands can
    discover related summaries and error files automatically.
    """

    total = 1
    seen_any = False
    for rule_smarts in sequence:
        seen_any = True
        total += max(reaction_output_reactants_num(rule_smarts), 1) - 1
    return total if seen_any else 0


def write_alchemical_rules_tsv(
    output: Path,
    aggregates: dict[str, Any],
) -> dict[str, int]:
    """Write alchemical rules TSV to disk.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    fieldnames = [
        "Alchemical_rule",
        "output_reactants_num",
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
    ]
    rows = []
    for aggregate in sorted_aggregates(aggregates):
        route_ids = sorted(aggregate.route_ids, key=reference_sort_key)
        rows.append(
            {
                "Alchemical_rule": aggregate.rule_smarts,
                "output_reactants_num": reaction_output_reactants_num(
                    aggregate.rule_smarts
                ),
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
            }
        )
    write_tsv(output, fieldnames, rows)
    return {"alchemical_rules": len(aggregates)}


def write_pseudo_reactions_smi(
    output: Path,
    pseudo_reactions: list[Any],
    aggregates: dict[str, Any],
) -> None:
    """Write pseudo reactions SMI to disk.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
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


def write_alchemical_errors(path: Path, errors: list[dict[str, Any]]) -> None:
    """Write alchemical errors to disk.

    Centralizing this logic keeps file naming, compressed input handling, and TSV/JSON
    formatting consistent across extraction, scoring, and tutorials.
    """
    if not errors:
        if path.exists():
            path.unlink()
        return
    fieldnames = [
        "row_index",
        "Target_smiles",
        "Composite_rule",
        "source_tsv",
        "Composite_size",
        "Route_ids",
    ]
    write_tsv(
        path,
        fieldnames,
        (alchemical_error_row_for_output(row) for row in errors),
    )


def source_tsv_prefix(path: str | Path) -> str:
    """Return the compact dataset and rule-size prefix for a source TSV.

    Keeping this helper in the IO layer avoids duplicating low-level file and table
    behavior in the chemistry-focused modules.
    """
    stem = Path(path).stem
    return re.sub(r"_composite_rules$", "", stem)


def alchemical_error_row_for_output(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize an alchemical extraction error row for TSV output.

    The returned path follows the project naming conventions so downstream commands can
    discover related summaries and error files automatically.
    """
    return {
        "row_index": row.get("row_index", ""),
        "Target_smiles": row.get("Target_smiles")
        or row.get("target_smiles")
        or row.get("Target_molecules", ""),
        "Composite_rule": row.get("Composite_rule", ""),
        "source_tsv": source_tsv_prefix(row.get("source_tsv", "")),
        "Composite_size": row.get("Composite_size")
        or row.get("composite_size", ""),
        "Route_ids": row.get("Route_ids") or row.get("route_ids", ""),
    }
