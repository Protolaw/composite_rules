from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from alchems.io import write_json, write_tsv
from alchems.protection.analysis import ProtectionAnalysisResult


ROUTE_STATS_FIELDS = [
    "route_id",
    "target_smiles",
    "n_steps",
    "n_molecule_nodes",
    "n_reaction_nodes",
    "n_in_stock_leaves",
    "n_protecting_groups_total",
    "n_pg_types",
    "pg_types",
    "n_pg_introduced",
    "n_pg_stock",
    "n_pg_ambiguous",
    "n_pg_failed",
    "n_deprotection_events",
    "n_multicenter_deprotection_events",
    "n_deprotective_combo_events",
    "max_simultaneous_pg",
    "mean_pg_lifetime_steps",
    "median_pg_lifetime_steps",
    "n_interval_composite_rules",
    "n_unique_interval_composite_rule_families",
]

EVENT_FIELDS = [
    "event_id",
    "route_id",
    "target_smiles",
    "pg_type",
    "protected_functional_group",
    "source_type",
    "trace_status",
    "confidence",
    "deprotection_node_id",
    "deprotection_reaction_smiles",
    "protection_node_id",
    "protection_reaction_smiles",
    "stock_node_id",
    "stock_smiles",
    "protected_precursor_smiles",
    "deprotected_product_smiles",
    "protected_atom_ids",
    "protected_substructure_smiles",
    "depth_deprotection_from_target",
    "depth_source_from_target",
    "lifetime_steps",
    "n_intervening_steps",
    "multicenter_status",
    "n_other_reaction_centers",
    "n_matching_pg_sites_before_deprotection",
    "n_sites_deprotected",
    "selective_deprotection",
    "multiple_deprotection_rules",
    "matched_deprotection_rules",
    "failure_reason",
]

INTERVAL_RULE_FIELDS = [
    "event_id",
    "route_id",
    "pg_type",
    "trace_status",
    "interval_step_ids",
    "interval_reaction_smiles",
    "single_step_rules",
    "composite_rule_size",
    "composite_rule_smarts",
    "composite_rule_querycgr",
    "composite_rule_querycgr_hash",
    "composite_rule_family_id",
    "composite_rule_support_in_routes",
    "is_full_interval_rule",
    "is_adjacent_window_rule",
    "macro_depth_saving",
    "min_distance_pg_to_reaction_center",
    "median_distance_pg_to_reaction_center",
    "first_reaction_after_source",
    "last_reaction_before_deprotection",
]

GROUP_SUMMARY_FIELDS = [
    "pg_type",
    "popularity",
    "route_count",
    "target_count",
    "introduced_count",
    "stock_count",
    "ambiguous_count",
    "failed_count",
    "introduction_fraction",
    "stock_fraction",
    "failure_fraction",
    "median_lifetime_steps",
    "mean_lifetime_steps",
    "max_lifetime_steps",
    "n_unique_composite_rules",
    "n_unique_composite_rule_families",
    "top_composite_rule_families",
    "top_first_reactions_after_source",
    "top_last_reactions_before_deprotection",
    "top_protected_functional_groups",
    "n_multicenter_deprotection_events",
    "n_deprotective_combo_events",
    "n_selective_deprotections",
    "selective_deprotection_fraction",
]

RULE_FAMILY_FIELDS = [
    "pg_type",
    "composite_rule_family_id",
    "family_popularity",
    "route_count",
    "target_count",
    "representative_composite_rule_smarts",
    "family_size_rules",
    "median_pairwise_similarity",
    "min_pairwise_similarity",
    "max_pairwise_similarity",
    "example_route_ids",
    "example_event_ids",
    "example_target_smiles",
    "interpretation_label",
]

NETWORK_EDGE_FIELDS = [
    "source_pg_type",
    "target_rule_family_id",
    "edge_weight",
    "route_count",
    "target_count",
    "median_lifetime_steps",
    "representative_rule",
    "interpretation_label",
]

TRACE_FAILURE_FIELDS = [
    "event_id",
    "route_id",
    "pg_type",
    "deprotection_node_id",
    "protected_precursor_smiles",
    "protected_atom_ids",
    "trace_status",
    "failure_reason",
    "last_successful_node_id",
    "last_successful_smiles",
    "candidate_next_nodes",
    "n_candidate_traces",
    "debug_message",
]


def dataset_prefix_from_routes_path(routes_json: Path) -> str:
    stem = routes_json.stem
    stem = re.sub(r"[-_]?routes$", "", stem)
    return stem or "routes"


def write_protection_outputs(
    result: ProtectionAnalysisResult,
    output_dir: Path,
    *,
    dataset_prefix: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / dataset_prefix
    paths = {
        "route_stats": prefix.with_name(
            f"{prefix.name}_protection_route_stats.tsv"
        ),
        "events": prefix.with_name(f"{prefix.name}_protection_events.tsv"),
        "interval_rules": prefix.with_name(
            f"{prefix.name}_protection_interval_rules.tsv"
        ),
        "group_summary": prefix.with_name(
            f"{prefix.name}_protection_group_summary.tsv"
        ),
        "rule_families": prefix.with_name(
            f"{prefix.name}_protection_rule_families.tsv"
        ),
        "network_edges": prefix.with_name(
            f"{prefix.name}_protection_network_edges.tsv"
        ),
        "trace_failures": prefix.with_name(
            f"{prefix.name}_protection_trace_failures.tsv"
        ),
        "summary": prefix.with_name(f"{prefix.name}_protection_summary.json"),
    }
    write_tsv(paths["route_stats"], ROUTE_STATS_FIELDS, result.route_stats_rows)
    write_tsv(paths["events"], EVENT_FIELDS, result.event_rows)
    write_tsv(
        paths["interval_rules"],
        INTERVAL_RULE_FIELDS,
        result.interval_rule_rows,
    )
    write_tsv(
        paths["group_summary"],
        GROUP_SUMMARY_FIELDS,
        result.group_summary_rows,
    )
    write_tsv(
        paths["rule_families"],
        RULE_FAMILY_FIELDS,
        result.rule_family_rows,
    )
    write_tsv(
        paths["network_edges"],
        NETWORK_EDGE_FIELDS,
        result.network_edge_rows,
    )
    write_tsv(
        paths["trace_failures"],
        TRACE_FAILURE_FIELDS,
        result.trace_failure_rows,
    )
    summary = dict(result.summary)
    summary["dataset"] = dataset_prefix
    summary["output_files"] = {key: str(path) for key, path in paths.items()}
    write_json(paths["summary"], summary)

    if result.debug_routes:
        debug_dir = output_dir / "debug" / "routes"
        for route_id, route in result.debug_routes.items():
            safe_route_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(route_id))
            write_json(debug_dir / f"{safe_route_id}.json", {str(route_id): route})

    return {"output_files": {key: str(path) for key, path in paths.items()}}
