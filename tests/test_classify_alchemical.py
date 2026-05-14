import csv

from composite_rules.classify_alchemical import classify_alchemical_rules


def test_classify_alchemical_rules_marks_default_cgr_matches_negative(tmp_path):
    alchemical_path = tmp_path / "alchemical.tsv"
    default_path = tmp_path / "default.tsv"
    output_path = tmp_path / "classified.tsv"
    summary_path = tmp_path / "summary.json"

    alchemical_path.write_text(
        "Alchemical_rule\tpopularity\tAlchemical_cgr\n"
        "[C:1]-[O:2]>>[C:1].[O:2]\t1\t[C][->.][O]\n",
        encoding="utf-8",
    )
    default_path.write_text(
        "rule_smarts\tpopularity\treaction_indices\n"
        "[C:1]-[O:2]>>[C:1].[O:2]\t1\t0\n",
        encoding="utf-8",
    )

    summary = classify_alchemical_rules(
        alchemical_path,
        default_path,
        output_path,
        summary_path,
    )

    with output_path.open() as file:
        rows = list(csv.DictReader(file, delimiter="\t"))

    assert summary["negative"] == 1
    assert summary["positive"] == 0
    assert rows[0]["classification"] == "negative"
    assert rows[0]["Matched_default_rule_ids"] == "0"
