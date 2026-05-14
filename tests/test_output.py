import csv
import json

from composite_rules.extract import write_composite_rules, write_summary


def test_write_composite_rules_splits_by_size_and_adds_popularity(tmp_path):
    output = tmp_path / "n1.tsv"
    summary = write_composite_rules(
        output,
        {
            ("a", "b"): {2, 1},
            ("b", "c"): {3},
            ("a", "b", "c"): {1},
        },
        target_molecules_by_sequence={
            ("a", "b"): {2: {"CCO"}, 1: {"CCN"}},
            ("b", "c"): {3: {"CCC"}},
            ("a", "b", "c"): {1: {"CCN"}},
        },
    )

    t2 = tmp_path / "n1_t2_composite_rules.tsv"
    t3 = tmp_path / "n1_t3_composite_rules.tsv"
    assert set(summary["output_files"]) == {"2", "3"}
    assert t2.exists()
    assert t3.exists()

    with t2.open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))

    assert rows[0] == {
        "Composite_rule": "a$b",
        "popularity": "2",
        "route_ids_size": "2",
        "Reference": "1,2",
        "Target_molecules": "CCN,CCO",
    }
    assert rows[1] == {
        "Composite_rule": "b$c",
        "popularity": "1",
        "route_ids_size": "1",
        "Reference": "3",
        "Target_molecules": "CCC",
    }


def test_summary_file_name(tmp_path):
    output = tmp_path / "n1.tsv"
    path = write_summary(output, {"ok": True})

    assert path == tmp_path / "n1_composite_rule_extraction_summary.json"
    assert json.loads(path.read_text()) == {"ok": True}
