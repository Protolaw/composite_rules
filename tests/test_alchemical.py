import csv

from composite_rules.alchemical import (
    AlchemicalRuleAggregate,
    PseudoReactionRecord,
    compose_pseudo_reaction_smiles,
    expand_composite_rule_tsv_paths,
    normalize_pseudo_reaction_mapping,
    resolve_alchemical_output_paths,
    rule_cgr_key,
    write_errors,
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
    assert rule_cgr_key("[C:1]-[O:2]>>[C:1].[O:2]") == "[C][->.][O]"


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

    write_errors(error_path, [])

    assert not error_path.exists()


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
    assert rows[0]["Reference"] == "1,2"
    assert rows[0]["Composite_rules"] == "a$b"
    assert smi_path.read_text().startswith("[OH2:2]>>[CH3:1][OH:2]\tp0\ta0")
