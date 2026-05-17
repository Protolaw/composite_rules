import csv

from alchems.alchemical_rules.alchemical import (
    AlchemicalRuleAggregate,
    ExtractedAlchemicalRule,
    PseudoReactionRecord,
    collect_alchemical_rules,
    compose_pseudo_reaction_smiles,
    normalize_pseudo_reaction_mapping,
    query_cgr_isomorphic,
    rule_cgr_key,
    rule_query_cgr,
)
from alchems.io import (
    expand_composite_rule_tsv_paths,
    reaction_output_reactants_num,
    resolve_alchemical_output_paths,
    write_alchemical_errors,
    write_alchemical_rules_tsv,
    write_pseudo_reactions_smi,
)


def test_compose_pseudo_reaction_smiles_collects_final_leaves():
    reaction_smiles = compose_pseudo_reaction_smiles(
        "CCO",
        "[C:1]-[O:2]>>[C:1].[O:2]$[C:1]-[C:2]>>[C:1].[C:2]",
    )

    assert reaction_smiles == "[CH4:1].[CH4:2].[OH2:3]>>[CH3:1][CH2:2][OH:3]"


def test_normalize_pseudo_reaction_mapping_remaps_generated_collisions():
    from chython import smiles

    reactants, product = normalize_pseudo_reaction_mapping(
        [smiles("[Cl:3]"), smiles("[OH2:3]")],
        smiles("[CH3:1][CH2:2][OH:3]"),
    )

    assert format(product, "m") == "[CH3:1][CH2:2][OH:3]"
    assert [format(reactant, "m") for reactant in reactants] == [
        "[ClH:4]",
        "[OH2:3]",
    ]


def test_rule_cgr_key_uses_query_cgr_identity():
    assert rule_cgr_key("[C:1]-[O:2]>>[C:1].[O:2]") == rule_cgr_key(
        "[C:7]-[O:9]>>[C:7].[O:9]"
    )


def test_reaction_output_reactants_num_counts_rule_products():
    assert (
        reaction_output_reactants_num(
            "[C;D3:1]-[C;D3:2](=[O;D1:3])-[N;D2:4]-[C;D3:5]"
            ">>"
            "[C;D3:1]-[C;D3:2](=[O;D1:3])-[O;D2:6]-[C;D1:7]."
            "[N;D1:4]-[C;D3:5]"
        )
        == 2
    )
    assert (
        reaction_output_reactants_num(
            "[Si;D4:1]-[C;D3:2]:1:[C;D2:3]:[N;D3:4](-[C;D3:5]):"
            "[N;D2:6]:[N;D2:7]:1"
            ">>"
            "[Si;D4:1]-[C;D2:2]#[C;D1:3]."
            "[N;D2+:6](=[N;D1-:7])=[N;D1-:8]."
            "[N;D1:4]-[C;D3:5]"
        )
        == 3
    )


def test_query_cgr_isomorphic_ignores_rule_atom_map_numbering():
    left = rule_query_cgr("[C:1]-[O:2]>>[C:1].[O:2]")
    right = rule_query_cgr("[C:7]-[O:9]>>[C:7].[O:9]")

    assert query_cgr_isomorphic(left, right)


def test_expand_composite_rule_tsv_paths_accepts_directories_and_deduplicates(tmp_path):
    t2_path = tmp_path / "n1_t2_composite_rules.tsv"
    t3_path = tmp_path / "n1_t3_composite_rules.tsv"
    other_path = tmp_path / "notes.tsv"
    t2_path.write_text("Composite_rule\n", encoding="utf-8")
    t3_path.write_text("Composite_rule\n", encoding="utf-8")
    other_path.write_text("ignored\n", encoding="utf-8")

    assert expand_composite_rule_tsv_paths([t2_path, tmp_path]) == [
        t2_path,
        t3_path,
    ]


def test_write_errors_removes_stale_file_when_run_is_clean(tmp_path):
    error_path = tmp_path / "errors.tsv"
    error_path.write_text("old errors\n", encoding="utf-8")

    write_alchemical_errors(error_path, [])

    assert not error_path.exists()


def test_write_alchemical_errors_uses_compact_schema(tmp_path):
    error_path = tmp_path / "errors.tsv"
    write_alchemical_errors(
        error_path,
        [
            {
                "source_tsv": str(tmp_path / "n1_t2_composite_rules.tsv"),
                "row_index": 3,
                "Target_smiles": "CCO",
                "Composite_rule": "a$b",
                "Composite_size": 2,
                "Route_ids": "1,2",
            }
        ],
    )

    with error_path.open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))

    assert list(rows[0]) == [
        "row_index",
        "Target_smiles",
        "Composite_rule",
        "source_tsv",
        "Composite_size",
        "Route_ids",
    ]
    assert rows[0] == {
        "row_index": "3",
        "Target_smiles": "CCO",
        "Composite_rule": "a$b",
        "source_tsv": "n1_t2",
        "Composite_size": "2",
        "Route_ids": "1,2",
    }


def test_resolve_alchemical_output_paths_accepts_output_directory(tmp_path):
    output_dir = tmp_path / "alchemical_out"
    tsv_paths = [
        tmp_path / "n1_t2_composite_rules.tsv",
        tmp_path / "n1_t3_composite_rules.tsv",
    ]

    rules_path, smi_path, summary_path, error_path = resolve_alchemical_output_paths(
        output_dir,
        tsv_paths,
    )

    assert rules_path == output_dir / "n1_alchemical_rules.tsv"
    assert smi_path == output_dir / "n1_alchemical_reactions.smi"
    assert summary_path == output_dir / "n1_alchemical_rule_collection_summary.json"
    assert error_path == output_dir / "n1_alchemical_rule_collection_errors.tsv"


def test_resolve_alchemical_output_paths_uses_merged_stem_for_mixed_prefixes(tmp_path):
    output_dir = tmp_path / "alchemical_out"
    tsv_paths = [
        tmp_path / "n1_t2_composite_rules.tsv",
        tmp_path / "n5_t2_composite_rules.tsv",
    ]

    rules_path, smi_path, _summary_path, _error_path = resolve_alchemical_output_paths(
        output_dir,
        tsv_paths,
    )

    assert rules_path == output_dir / "merged_alchemical_rules.tsv"
    assert smi_path == output_dir / "merged_alchemical_reactions.smi"


def test_write_alchemical_outputs(tmp_path):
    aggregate = AlchemicalRuleAggregate(
        rule_smarts="[C:1]-[O:2]>>[C:1].[O:2]",
        cgr_key="[C][->.][O]",
    )
    aggregate.route_ids.update({"2", "1"})
    aggregate.target_molecules.add("CCO")
    aggregate.composite_rules.add("a$b")
    aggregate.composite_sizes.add(2)
    aggregate.source_rows.add("n1.tsv:0")
    aggregate.pseudo_reaction_ids.append("p0")

    rules_path = tmp_path / "rules.tsv"
    smi_path = tmp_path / "reactions.smi"
    write_alchemical_rules_tsv(rules_path, {"[C][->.][O]": aggregate})
    write_pseudo_reactions_smi(
        smi_path,
        [
            PseudoReactionRecord(
                pseudo_reaction_id="p0",
                alchemical_cgr="[C][->.][O]",
                reaction_smiles="[OH2:2]>>[CH3:1][OH:2]",
                source_tsv="n1.tsv",
                source_row=0,
                route_ids=("1", "2"),
                target_smiles="CO",
                composite_size=2,
                composite_rule="a$b",
            )
        ],
        {"[C][->.][O]": aggregate},
    )

    with rules_path.open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))

    assert rows[0]["Alchemical_rule"] == "[C:1]-[O:2]>>[C:1].[O:2]"
    assert rows[0]["output_reactants_num"] == "2"
    assert rows[0]["Reference"] == "1,2"
    assert rows[0]["Composite_rules"] == "a$b"
    assert "Alchemical_cgr" not in rows[0]
    assert smi_path.read_text().startswith("[OH2:2]>>[CH3:1][OH:2]\tp0\ta0")


def test_collect_alchemical_rules_merges_query_cgr_duplicates(
    tmp_path,
    monkeypatch,
):
    composite_path = tmp_path / "n1_t2_composite_rules.tsv"
    composite_path.write_text(
        "Composite_rule\tReference\tTarget_molecules\n"
        "a$b\t1\tCCO\n"
        "c$d\t2\tCCN\n",
        encoding="utf-8",
    )
    rules = [
        "[C:1]-[O:2]>>[C:1].[O:2]",
        "[C:7]-[O:9]>>[C:7].[O:9]",
    ]

    monkeypatch.setattr(
        "alchems.alchemical_rules.alchemical.compose_pseudo_reaction_smiles",
        lambda target_smiles, composite_rule: f"{target_smiles}>{composite_rule}",
    )

    class FakeExtractor:
        def __init__(self):
            self.index = 0

        def extract(self, _reaction_smiles):
            rule = rules[self.index]
            self.index += 1
            return ExtractedAlchemicalRule(
                rule_smarts=rule,
                cgr_key=rule_cgr_key(rule),
                query_cgr=rule_query_cgr(rule),
            )

    aggregates, pseudo_reactions, stats, errors = collect_alchemical_rules(
        [composite_path],
        FakeExtractor(),
    )

    assert not errors
    assert stats.alchemical_rules_extracted == 2
    assert len(aggregates) == 1
    aggregate = next(iter(aggregates.values()))
    assert aggregate.route_ids == {"1", "2"}
    assert aggregate.target_molecules == {"CCO", "CCN"}
    assert {record.alchemical_cgr for record in pseudo_reactions} == {
        aggregate.cgr_key
    }


def test_collect_alchemical_rules_writes_skipped_unwrap_context(
    tmp_path,
    monkeypatch,
):
    from alchems.composite_rules.unwrap import RuleApplicationError

    composite_path = tmp_path / "n1_t2_composite_rules.tsv"
    composite_path.write_text(
        "Composite_rule\tReference\tTarget_molecules\n"
        "a$b\t1,2\tCCO\n",
        encoding="utf-8",
    )

    def fail_unwrap(_target_smiles, _composite_rule):
        raise RuleApplicationError("rule did not match active molecule")

    monkeypatch.setattr(
        "alchems.alchemical_rules.alchemical.compose_pseudo_reaction_smiles",
        fail_unwrap,
    )

    aggregates, pseudo_reactions, stats, errors = collect_alchemical_rules(
        [composite_path],
        object(),
    )

    assert not aggregates
    assert not pseudo_reactions
    assert stats.skipped_unwrap_applications == 1
    assert errors == [
        {
            "source_tsv": str(composite_path),
            "row_index": 0,
            "Target_smiles": "CCO",
            "Composite_rule": "a$b",
            "Composite_size": 2,
            "Route_ids": "1,2",
        }
    ]
