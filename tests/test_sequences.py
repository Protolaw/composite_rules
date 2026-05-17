from alchems.composite_rules.extract import (
    MoleculeCenterProjection,
    ReactionRuleStep,
    adjacent_centers_overlap,
    is_excluded_adjacent_pair,
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


def projected_step(name, target_smiles, reactant_centers=(), product_centers=()):
    from chython import smiles

    molecule = smiles(target_smiles)
    return ReactionRuleStep(
        rule_smarts=name,
        center_atoms=frozenset(),
        reaction_smiles=f"{name}>>{name}",
        target_smiles=target_smiles,
        reactant_center_molecules=(
            MoleculeCenterProjection(molecule, frozenset(reactant_centers)),
        ),
        product_center_molecules=(
            MoleculeCenterProjection(molecule, frozenset(product_centers)),
        ),
    )


def test_projected_adjacent_centers_allow_functional_group_center_shift():
    left = projected_step("protect_amine", "NC", reactant_centers={1})
    right = projected_step("amide_reduction", "NC", product_centers={2})

    assert adjacent_centers_overlap(left, right)


def test_projected_adjacent_centers_reject_aromatic_neighbor_only_contacts():
    left = projected_step("left_aryl", "c1ccccc1", reactant_centers={1})
    right = projected_step("right_aryl", "c1ccccc1", product_centers={2})

    assert not adjacent_centers_overlap(left, right)


def test_projected_adjacent_centers_do_not_bridge_parallel_components():
    left = projected_step("left", "NCCO", reactant_centers={1})
    right = projected_step("parallel", "NCCO", product_centers={1, 4})

    assert not adjacent_centers_overlap(left, right)


def test_excluded_adjacent_pair_filters_deprotection_to_sulfonyl_activation():
    left = step(
        "[C;D2:1]-[O;D2:2]-[S;D4:3](-[C;D1:4])(=[O;D1:5])=[O;D1:6]>>"
        "[S;D4:3](-[C;D1:4])(=[O;D1:5])(=[O;D1:6])-[Cl;D1:7]."
        "[C;D2:1]-[O;D1:2]",
        {1},
    )
    right = step(
        "[C;D2:1]-[O;D1:2]>>"
        "[C;D2:1]-[O;D2:2]-[C;D3:4](-[C;D1:3])=[O;D1:5]",
        {1},
    )

    assert is_excluded_adjacent_pair(left, right)
