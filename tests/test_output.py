import csv
import json

from route_inspector.io import (
    dataset_prefix_from_path,
    resolve_output_path,
    stage_output_dir,
    write_manifest,
    write_standard_sidecars,
    write_composite_routes_without_rules,
    write_composite_rules,
    write_composite_summary,
)


def test_write_composite_rules_splits_by_size_and_adds_popularity(tmp_path):
    output = tmp_path / "n1.tsv"
    summary = write_composite_rules(
        output,
        {
            ("z",): {4},
            ("a", "b"): {2, 1},
            ("b", "c"): {3},
            ("a", "b", "c"): {1},
        },
        target_molecules_by_sequence={
            ("z",): {4: {"CCCl"}},
            ("a", "b"): {2: {"CCO"}, 1: {"CCN"}},
            ("b", "c"): {3: {"CCC"}},
            ("a", "b", "c"): {1: {"CCN"}},
        },
    )

    t1 = tmp_path / "n1_t1_single_rules.tsv"
    t2 = tmp_path / "n1_t2_composite_rules.tsv"
    t3 = tmp_path / "n1_t3_composite_rules.tsv"
    assert set(summary["output_files"]) == {"1", "2", "3"}
    assert summary["unique_single_rules"] == 1
    assert summary["unique_composite_rules_by_size"] == {"2": 2, "3": 1}
    assert t1.exists()
    assert t2.exists()
    assert t3.exists()

    with t1.open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))

    assert rows[0] == {
        "Rule": "z",
        "popularity": "1",
        "Reference": "4",
        "Target_molecules": "CCCl",
    }

    with t2.open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))

    assert rows[0] == {
        "Composite_rule": "a$b",
        "output_reactants_num": "1",
        "popularity": "2",
        "Reference": "1,2",
        "Target_molecules": "CCN,CCO",
    }
    assert rows[1] == {
        "Composite_rule": "b$c",
        "output_reactants_num": "1",
        "popularity": "1",
        "Reference": "3",
        "Target_molecules": "CCC",
    }


def test_summary_file_name(tmp_path):
    output = tmp_path / "n1.tsv"
    path = write_composite_summary(output, {"ok": True})

    assert path == tmp_path / "n1_composite_rule_extraction_summary.json"
    assert json.loads(path.read_text()) == {"ok": True}


def test_write_composite_routes_without_rules(tmp_path):
    output = tmp_path / "n1.tsv"
    path = write_composite_routes_without_rules(
        output,
        {
            7: {
                "target_smiles": "CCO",
            }
        },
    )

    assert path == tmp_path / "n1_routes_without_composite_rules.json"
    assert json.loads(path.read_text()) == {
        "7": {
            "target_smiles": "CCO",
        }
    }


def test_dataset_prefix_from_clean_routes_path():
    assert dataset_prefix_from_path("data/clean/n1_routes.json") == "n1"
    assert dataset_prefix_from_path("data/clean/n5-routes.json") == "n5"


def test_stage_output_dir_uses_standard_stage_name(tmp_path):
    assert stage_output_dir(tmp_path, "n1", "composite_rules") == (
        tmp_path / "n1" / "10_composite_rules"
    )


def test_resolve_output_path_preserves_explicit_output(tmp_path):
    explicit = tmp_path / "custom.tsv"

    assert (
        resolve_output_path(
            output=explicit,
            output_dir=tmp_path / "ignored",
            default_filename="n1.tsv",
        )
        == explicit
    )


def test_resolve_output_path_uses_output_dir_default_filename(tmp_path):
    assert (
        resolve_output_path(
            output=None,
            output_dir=tmp_path / "stage",
            default_filename="n1.tsv",
        )
        == tmp_path / "stage" / "n1.tsv"
    )


def test_write_manifest_creates_reproducibility_sidecar(tmp_path):
    path = write_manifest(
        tmp_path,
        command_name="extract-composite-rules",
        input_files=["data/clean/n1_routes.json"],
        output_files={"summary": tmp_path / "summary.json"},
        config_path="configs/rules.yaml",
        argv=["route-inspector", "extract-composite-rules"],
    )

    manifest = json.loads(path.read_text())
    assert path == tmp_path / "manifest.json"
    assert manifest["command_name"] == "extract-composite-rules"
    assert manifest["input_files"] == ["data/clean/n1_routes.json"]
    assert manifest["output_directory"] == str(tmp_path)
    assert manifest["config_path"] == "configs/rules.yaml"
    assert "created_at" in manifest


def test_write_standard_sidecars_can_skip_generic_summary(tmp_path):
    sidecars = write_standard_sidecars(
        tmp_path,
        command_name="preprocess-routes",
        summary={"ok": True},
        errors=[],
        write_summary=False,
    )

    assert "summary" not in sidecars
    assert not (tmp_path / "summary.json").exists()
    assert (tmp_path / "errors.tsv").exists()
    assert (tmp_path / "manifest.json").exists()
