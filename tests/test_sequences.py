from composite_rules.extract import (
    ReactionRuleStep,
    valid_composite_sequence_occurrences,
    valid_composite_sequences,
)


def step(name, centers):
    return ReactionRuleStep(
        rule_smarts=name,
        center_atoms=frozenset(centers),
        reaction_smiles=f"{name}>>{name}",
        target_smiles=f"{name}_target",
    )


def test_contiguous_sequences_only_when_adjacent_centers_overlap():
    path = [
        step("t1", {1, 2}),
        step("t2", {2, 3}),
        step("t3", {3, 4}),
    ]

    assert set(valid_composite_sequences(path, min_length=2, max_length=5)) == {
        ("t1", "t2"),
        ("t2", "t3"),
        ("t1", "t2", "t3"),
    }


def test_non_contiguous_overlap_is_not_a_composite_rule():
    path = [
        step("t1", {1, 2}),
        step("t2", {3, 4}),
        step("t3", {2, 5}),
    ]

    assert set(valid_composite_sequences(path, min_length=2, max_length=5)) == set()


def test_sequence_break_starts_new_segment():
    path = [
        step("t1", {1, 2}),
        step("t2", {2, 3}),
        step("t3", {8}),
        step("t4", {8, 9}),
    ]

    assert set(valid_composite_sequences(path, min_length=2, max_length=5)) == {
        ("t1", "t2"),
        ("t3", "t4"),
    }


def test_sequence_occurrences_keep_start_molecule():
    path = [
        step("t1", {1, 2}),
        step("t2", {2, 3}),
        step("t3", {3, 4}),
    ]

    assert set(
        valid_composite_sequence_occurrences(path, min_length=2, max_length=5)
    ) == {
        (("t1", "t2"), "t1_target"),
        (("t2", "t3"), "t2_target"),
        (("t1", "t2", "t3"), "t1_target"),
    }
