import argparse
import json

from alchems.composite_rules import extract


def test_extract_run_writes_routes_without_composites_file(tmp_path, monkeypatch):
    routes_json = tmp_path / "routes.json"
    routes_json.write_text(
        json.dumps(
            [
                {
                    "smiles": "CCO",
                    "type": "mol",
                    "in_stock": False,
                    "children": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "n1.tsv"

    monkeypatch.setattr(
        extract.SynPlannerRuleExtractor,
        "from_args",
        classmethod(lambda cls, args: object()),
    )

    exit_code = extract.run(
        argparse.Namespace(
            routes_json=routes_json,
            output=output,
            config=None,
            environment_atom_count=1,
            include_rings=False,
            keep_leaving_groups=True,
            keep_incoming_groups=False,
            reactor_validation=False,
            limit=None,
            min_length=2,
            max_length=5,
            routes_without_composites_output=None,
            ignore_errors=False,
            progress_interval=0,
        )
    )

    assert exit_code == 0

    no_rules_path = tmp_path / "n1_routes_without_composite_rules.json"
    routes_without_rules = json.loads(no_rules_path.read_text())

    assert list(routes_without_rules) == ["0"]
    assert routes_without_rules["0"]["smiles"] == "CCO"
    assert routes_without_rules["0"]["metadata"]["composite_rule_extraction"] == {
        "route_id": 0,
        "target_smiles": "CCO",
        "reactions_seen": 0,
        "extracted_reaction_rules": 0,
        "skipped_reactions": 0,
        "reason": "no_reactions",
    }

    summary = json.loads(
        (tmp_path / "n1_composite_rule_extraction_summary.json").read_text()
    )
    assert summary["routes_without_composite_rules"] == 1
    assert summary["routes_without_composite_rules_file"] == str(no_rules_path)
    assert summary["routes_without_composite_rules_by_reason"] == {"no_reactions": 1}
