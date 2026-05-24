import copy
import json
from dataclasses import dataclass

from route_inspector.preprocess_routes import (
    ReactionGranularity,
    SingleCenterRule,
    dataset_output_paths,
    molecule_has_bond,
    preprocess_routes_file,
    preprocess_routes_json,
    same_molecule_smiles,
)
from route_inspector.protection.analysis import ProtectionAnalysisConfig, parse_molecule


PARENT = "[CH2:1]=[O:2]"
CHILD = "[CH3:1][OH:2]"
REACTION = f"{CHILD}>>{PARENT}"


class FakeExtractor:
    def __init__(self, extraction):
        self.extraction = extraction
        self.reactions = []

    def extract(self, reaction_smiles):
        self.reactions.append(reaction_smiles)
        return self.extraction


@dataclass(frozen=True)
class FakeProtectionMatch:
    protected_atom_ids: tuple[int, ...] = (2,)
    raw_mapping: tuple[tuple[int, int], ...] = ((1, 2),)


@dataclass(frozen=True)
class FakeProtectionEvent:
    protection_node_id: str = "r0"
    protected_atom_ids: tuple[int, ...] = (2,)


def route_with_one_reaction():
    return {
        "smiles": PARENT,
        "type": "mol",
        "in_stock": False,
        "children": [
            {
                "type": "reaction",
                "metadata": {"smiles": REACTION},
                "children": [
                    {
                        "smiles": CHILD,
                        "type": "mol",
                        "in_stock": True,
                        "children": [],
                    }
                ],
            }
        ],
    }


def route_from_reaction(reaction_smiles):
    reactants, product = reaction_smiles.split(">>", 1)
    return {
        "smiles": product,
        "type": "mol",
        "in_stock": False,
        "children": [
            {
                "type": "reaction",
                "metadata": {"smiles": reaction_smiles},
                "children": [
                    {
                        "smiles": reactant,
                        "type": "mol",
                        "in_stock": True,
                        "children": [],
                    }
                    for reactant in reactants.split(".")
                ],
            }
        ],
    }


def fail_if_unwrapped(*_args, **_kwargs):
    raise AssertionError("semantic single-center reaction should not be split")


def multicenter_extraction():
    return ReactionGranularity(
        reaction_smiles=REACTION,
        multicenter_rule_smarts="deprotect$transform",
        single_center_rules=(
            SingleCenterRule(
                "deprotect",
                frozenset({2}),
                forward_change_kind="bond_breaking",
                forward_bonds_broken=((2, 3),),
            ),
            SingleCenterRule(
                "transform",
                frozenset({1}),
                forward_change_kind="bond_forming",
                forward_bonds_formed=((1, 2),),
            ),
        ),
        center_components=(frozenset({2}), frozenset({1})),
    )


def non_protection_multicenter_extraction():
    return ReactionGranularity(
        reaction_smiles=REACTION,
        multicenter_rule_smarts="form$break",
        single_center_rules=(
            SingleCenterRule(
                "form",
                frozenset({1}),
                forward_change_kind="bond_forming",
                forward_bonds_formed=((1, 2),),
            ),
            SingleCenterRule(
                "break",
                frozenset({2}),
                forward_change_kind="bond_breaking",
                forward_bonds_broken=((2, 3),),
            ),
        ),
        center_components=(frozenset({1}), frozenset({2})),
    )


def identity_normalizer(route):
    return copy.deepcopy(route)


def test_route_normalization_is_called():
    calls = []

    def normalizer(route):
        calls.append(route)
        return copy.deepcopy(route)

    cleaned, _resolved, _unresolved, summary = preprocess_routes_json(
        [{"smiles": "CCO", "type": "mol", "in_stock": False, "children": []}],
        extractor=FakeExtractor(
            ReactionGranularity("", "", (), (), skipped=True)
        ),
        protection_rules={},
        protection_config=ProtectionAnalysisConfig(collect_interval_rules=False),
        normalizer=normalizer,
    )

    assert len(calls) == 1
    assert summary["number_of_normalized_routes"] == 1
    assert cleaned[0]["smiles"] == "CCO"


def test_non_protection_multicenter_reaction_is_split_by_forward_bond_order():
    seen_rule_orders = []

    def fake_unwrapper(target_smiles, rule_smarts, **_kwargs):
        seen_rule_orders.append(tuple(rule_smarts))
        return {
            0: {
                "type": "mol",
                "smiles": target_smiles,
                "children": [
                    {
                        "type": "reaction",
                        "smiles": "intermediate>>target",
                        "children": [
                            {
                                "type": "mol",
                                "smiles": "intermediate",
                                "children": [
                                    {
                                        "type": "reaction",
                                        "smiles": "child>>intermediate",
                                        "children": [
                                            {
                                                "type": "mol",
                                                "smiles": CHILD,
                                                "children": [],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        }

    cleaned, resolved, unresolved, summary = preprocess_routes_json(
        [route_with_one_reaction()],
        extractor=FakeExtractor(non_protection_multicenter_extraction()),
        protection_rules={},
        protection_config=ProtectionAnalysisConfig(collect_interval_rules=False),
        normalizer=identity_normalizer,
        protection_detector=lambda *_args: [],
        unwrapper=fake_unwrapper,
    )

    root_reaction = cleaned[0]["children"][0]
    nested_reaction = root_reaction["children"][0]["children"][0]

    assert seen_rule_orders[0] == ("break", "form")
    assert root_reaction["smiles"] == "intermediate>>target"
    assert nested_reaction["smiles"] == "child>>intermediate"
    assert list(resolved) == ["0"]
    assert unresolved == {}
    assert summary["number_of_multicenter_reactions_found"] == 1
    assert summary["number_of_multicenter_reactions_split"] == 1
    assert summary["number_of_protection_related_multicenter_reactions_split"] == 0
    assert summary["number_of_non_protection_multicenter_reactions_split"] == 1


def test_non_protection_mapped_intermediate_fallback_splits_n_oxide_chlorination():
    product = (
        "[c:14]1([cH:31][cH:32][c:6]([Br:69])[c:7]([Cl:42])[n:13]1)"
        "[C:15](=[O:38])[O:68][CH3:67]"
    )
    n_oxide = (
        "[cH:32]1[cH:31][c:14]([n+:13]([cH:7][c:6]1[Br:69])[O-:77])"
        "[C:15](=[O:38])[O:68][CH3:67]"
    )
    reagent = "[Cl:42][P:74]([Cl:75])([Cl:76])=[O:73]"
    reaction = f"{reagent}.{n_oxide}>>{product}"
    route = {
        "smiles": product,
        "type": "mol",
        "in_stock": False,
        "children": [
            {
                "type": "reaction",
                "metadata": {"smiles": reaction},
                "children": [
                    {"smiles": reagent, "type": "mol", "in_stock": True},
                    {"smiles": n_oxide, "type": "mol", "in_stock": True},
                ],
            }
        ],
    }
    extraction = ReactionGranularity(
        reaction_smiles=reaction,
        multicenter_rule_smarts="chlorination$n_oxide_cleavage",
        single_center_rules=(
            SingleCenterRule(
                "chlorination",
                frozenset({7, 42, 74}),
                forward_change_kind="bond_forming_and_breaking",
                forward_bonds_formed=((7, 42),),
                forward_bonds_broken=((42, 74),),
            ),
            SingleCenterRule(
                "n_oxide_cleavage",
                frozenset({13, 77}),
                forward_change_kind="bond_breaking",
                forward_bonds_broken=((13, 77),),
            ),
        ),
        center_components=(frozenset({7, 42, 74}), frozenset({13, 77})),
    )

    def failing_unwrapper(*_args, **_kwargs):
        raise RuntimeError("force mapped fallback")

    cleaned, resolved, unresolved, summary = preprocess_routes_json(
        [route],
        extractor=FakeExtractor(extraction),
        protection_rules={},
        protection_config=ProtectionAnalysisConfig(collect_interval_rules=False),
        normalizer=identity_normalizer,
        protection_detector=lambda *_args: [],
        unwrapper=failing_unwrapper,
        ignore_errors=True,
    )

    first_reaction = cleaned[0]["children"][0]
    intermediate = first_reaction["children"][0]
    second_reaction = intermediate["children"][0]
    change = summary["changes"][0]

    assert "[n+:13]([O-:77])" in first_reaction["smiles"].split(">>", 1)[0]
    assert first_reaction["smiles"].endswith(f">>{product}")
    assert second_reaction["smiles"].startswith(f"{reagent}.{n_oxide}>>")
    assert "[n+:13]([O-:77])" in second_reaction["smiles"].split(">>", 1)[1]
    assert (
        change["single_center_rule_details"][0]["forward_change_kind"]
        == "bond_breaking"
    )
    assert (
        change["single_center_rule_details"][1]["forward_change_kind"]
        == "bond_forming_and_breaking"
    )
    assert change["split_method"] == "mapped_intermediate"
    assert list(resolved) == ["0"]
    assert unresolved == {}
    assert summary["number_of_non_protection_multicenter_reactions_split"] == 1


def test_boronate_workup_is_not_split_as_multicenter():
    reaction = (
        "[CH2:34]1[CH2:35][O:33][CH:12]([O:36]1)[c:11]2[cH:19]"
        "[cH:20][c:8]([n:9][cH:10]2)[F:37].[CH3:43][CH:44]([CH3:45])"
        "[O:46][B:40]([O:41][CH:50]([CH3:51])[CH3:52])[O:39]"
        "[CH:47]([CH3:48])[CH3:49]>>[OH:39][B:40]([OH:41])[c:20]1"
        "[c:8]([n:9][cH:10][c:11]([CH:12]2[O:33][CH2:35][CH2:34]"
        "[O:36]2)[cH:19]1)[F:37]"
    )
    extraction = ReactionGranularity(
        reaction_smiles=reaction,
        multicenter_rule_smarts="boronate_workup",
        single_center_rules=(
            SingleCenterRule(
                "alkoxy_loss",
                frozenset({39, 47}),
                forward_change_kind="bond_breaking",
                forward_bonds_broken=((39, 47), (47, 48), (47, 49)),
            ),
            SingleCenterRule(
                "borylation",
                frozenset({20, 40, 46}),
                forward_change_kind="bond_forming_and_breaking",
                forward_bonds_formed=((20, 40),),
                forward_bonds_broken=((40, 46), (44, 46)),
            ),
        ),
        center_components=(frozenset({39, 47}), frozenset({20, 40, 46})),
    )

    cleaned, resolved, unresolved, summary = preprocess_routes_json(
        [route_from_reaction(reaction)],
        extractor=FakeExtractor(extraction),
        protection_rules={},
        protection_config=ProtectionAnalysisConfig(collect_interval_rules=False),
        normalizer=identity_normalizer,
        protection_detector=lambda *_args: [],
        unwrapper=fail_if_unwrapped,
    )

    assert cleaned[0]["children"][0]["metadata"]["smiles"] == reaction
    assert resolved == {}
    assert unresolved == {}
    assert summary["number_of_multicenter_reactions_found"] == 0
    assert summary["number_of_routes_modified"] == 0


def test_local_cascade_without_pure_breaking_is_not_split():
    reaction = (
        "[CH2:21]=[CH:22][C:17](=[O:43])[CH3:18].[CH3:36][C:37]"
        "([CH3:38])([CH3:39])[O:40][C:41](=[O:34])[N:25]1[CH2:24]"
        "[CH2:23][CH:20]([CH2:32][CH2:31]1)[CH:19]=[O:44]>>"
        "[CH3:36][C:37]([CH3:38])([CH3:39])[O:40][C:41](=[O:34])"
        "[N:25]1[CH2:24][CH2:23][C:20]2([CH2:32][CH2:31]1)"
        "[CH:19]=[CH:18][C:17](=[O:43])[CH2:22][CH2:21]2"
    )
    extraction = ReactionGranularity(
        reaction_smiles=reaction,
        multicenter_rule_smarts="local_cascade",
        single_center_rules=(
            SingleCenterRule(
                "enone",
                frozenset({18, 19, 44}),
                forward_change_kind="bond_forming_and_breaking",
                forward_bonds_formed=((18, 19),),
                forward_bonds_broken=((19, 44),),
            ),
            SingleCenterRule(
                "cyclization",
                frozenset({20, 21, 22}),
                forward_change_kind="bond_forming",
                forward_bonds_formed=((20, 21),),
            ),
        ),
        center_components=(frozenset({18, 19, 44}), frozenset({20, 21, 22})),
    )

    _cleaned, resolved, unresolved, summary = preprocess_routes_json(
        [route_from_reaction(reaction)],
        extractor=FakeExtractor(extraction),
        protection_rules={},
        protection_config=ProtectionAnalysisConfig(collect_interval_rules=False),
        normalizer=identity_normalizer,
        protection_detector=lambda *_args: [],
        unwrapper=fail_if_unwrapped,
    )

    assert resolved == {}
    assert unresolved == {}
    assert summary["number_of_multicenter_reactions_found"] == 0


def test_acetonide_c_n_cascade_is_not_split():
    reaction = (
        "[CH3:33][C:34]([CH3:35])([CH3:36])[O:37][C:38](=[O:39])"
        "[NH:16][c:17]1[c:18]([NH2:32])[cH:19][c:20]([C:21]#[C:22]"
        "[c:23]2[cH:24][cH:25][c:26]([cH:28][cH:29]2)[F:27])[cH:30]"
        "[cH:31]1.[CH3:41][C:42]1([CH3:43])[O:44][C:2]([CH:3]=[C:4]"
        "([c:5]2[cH:15][cH:14][cH:13][c:7](-[n:8]3[cH:12][n:11]"
        "[cH:10][cH:9]3)[cH:6]2)[O:40]1)=[O:1]>>[CH3:33][C:34]"
        "([CH3:35])([CH3:36])[O:37][C:38](=[O:39])[NH:16][c:17]1"
        "[c:18]([NH:32][C:2]([CH2:3][C:4](=[O:40])[c:5]2[cH:15]"
        "[cH:14][cH:13][c:7]([cH:6]2)-[n:8]3[cH:12][n:11][cH:10]"
        "[cH:9]3)=[O:1])[cH:19][c:20]([C:21]#[C:22][c:23]4[cH:24]"
        "[cH:25][c:26]([cH:28][cH:29]4)[F:27])[cH:30][cH:31]1"
    )
    extraction = ReactionGranularity(
        reaction_smiles=reaction,
        multicenter_rule_smarts="acetonide_c_n_cascade",
        single_center_rules=(
            SingleCenterRule(
                "acetonide",
                frozenset({3, 4, 40, 42}),
                forward_change_kind="bond_breaking",
                forward_bonds_broken=((40, 42), (41, 42), (42, 43), (42, 44)),
            ),
            SingleCenterRule(
                "c_n",
                frozenset({2, 32, 44}),
                forward_change_kind="bond_forming_and_breaking",
                forward_bonds_formed=((2, 32),),
                forward_bonds_broken=((2, 44), (42, 44)),
            ),
        ),
        center_components=(frozenset({3, 4, 40, 42}), frozenset({2, 32, 44})),
    )

    _cleaned, resolved, unresolved, summary = preprocess_routes_json(
        [route_from_reaction(reaction)],
        extractor=FakeExtractor(extraction),
        protection_rules={},
        protection_config=ProtectionAnalysisConfig(collect_interval_rules=False),
        normalizer=identity_normalizer,
        protection_detector=lambda *_args: [],
        unwrapper=fail_if_unwrapped,
    )

    assert resolved == {}
    assert unresolved == {}
    assert summary["number_of_multicenter_reactions_found"] == 0


def test_mapped_fallback_restores_full_leaving_fragment():
    reaction = (
        "[CH2:35]1[CH2:34][CH2:33][CH2:32][CH:31]([n:23]2[n:22]"
        "[cH:21][c:20]3[c:19]([n:27][cH:26][n:25][c:24]23)[Cl:30])"
        "[O:36]1.[F:1][C:2]([F:3])([c:4]1[cH:9][cH:8][cH:7][cH:6]"
        "[cH:5]1)[c:10]2[n:29][c:13]([o:12][n:11]2)[C@@H:14]3"
        "[CH2:28][C@@H:17]([NH2:18])[CH2:16][CH2:15]3>>[F:1][C:2]"
        "([F:3])([c:4]1[cH:9][cH:8][cH:7][cH:6][cH:5]1)[c:10]2"
        "[n:29][c:13]([o:12][n:11]2)[C@H:14]3[CH2:15][CH2:16]"
        "[C@H:17]([NH:18][c:19]4[c:20]5[c:24]([nH:23][n:22][cH:21]5)"
        "[n:25][cH:26][n:27]4)[CH2:28]3"
    )
    expected_intermediate = (
        "[F:1][C:2]([F:3])([c:4]1[cH:9][cH:8][cH:7][cH:6][cH:5]1)"
        "[c:10]2[n:29][c:13]([o:12][n:11]2)[C@H:14]3[CH2:15]"
        "[CH2:16][C@@H:17]([CH2:28]3)[NH:18][c:19]4[n:27][cH:26]"
        "[n:25][c:24]5[n:23]([n:22][cH:21][c:20]45)[CH2:35]1"
        "[CH2:34][CH2:33][CH2:32][C:31][O:36]1"
    )
    extraction = ReactionGranularity(
        reaction_smiles=reaction,
        multicenter_rule_smarts="substitution",
        single_center_rules=(
            SingleCenterRule(
                "restore_fragment",
                frozenset({23, 31}),
                forward_change_kind="bond_breaking",
                forward_bonds_broken=((23, 31), (31, 32), (31, 36)),
            ),
            SingleCenterRule(
                "c_n",
                frozenset({18, 19, 30}),
                forward_change_kind="bond_forming_and_breaking",
                forward_bonds_formed=((18, 19),),
                forward_bonds_broken=((19, 30),),
            ),
        ),
        center_components=(frozenset({23, 31}), frozenset({18, 19, 30})),
    )

    cleaned, resolved, unresolved, summary = preprocess_routes_json(
        [route_from_reaction(reaction)],
        extractor=FakeExtractor(extraction),
        protection_rules={},
        protection_config=ProtectionAnalysisConfig(collect_interval_rules=False),
        normalizer=identity_normalizer,
        protection_detector=lambda *_args: [],
        unwrapper=lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError()),
        ignore_errors=True,
    )

    first_reaction = cleaned[0]["children"][0]["smiles"]
    intermediate = first_reaction.split(">>", 1)[0]

    assert same_molecule_smiles(intermediate, expected_intermediate)
    intermediate_molecule = parse_molecule(intermediate)
    assert molecule_has_bond(intermediate_molecule, 23, 35)
    assert not molecule_has_bond(intermediate_molecule, 23, 31)
    assert "([CH3:32])[OH:36]" not in intermediate
    assert list(resolved) == ["0"]
    assert unresolved == {}
    assert summary["number_of_non_protection_multicenter_reactions_split"] == 1


def test_protection_related_multicenter_reaction_is_split():
    seen_rule_orders = []

    def fake_unwrapper(target_smiles, rule_smarts, **_kwargs):
        seen_rule_orders.append(tuple(rule_smarts))
        return {
            0: {
                "type": "mol",
                "smiles": target_smiles,
                "children": [
                    {
                        "type": "reaction",
                        "smiles": "intermediate>>target",
                        "children": [
                            {
                                "type": "mol",
                                "smiles": "intermediate",
                                "children": [
                                    {
                                        "type": "reaction",
                                        "smiles": "child>>intermediate",
                                        "children": [
                                            {
                                                "type": "mol",
                                                "smiles": CHILD,
                                                "children": [],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        }

    cleaned, resolved, unresolved, summary = preprocess_routes_json(
        [route_with_one_reaction()],
        extractor=FakeExtractor(multicenter_extraction()),
        protection_rules={},
        protection_config=ProtectionAnalysisConfig(collect_interval_rules=False),
        normalizer=identity_normalizer,
        protection_detector=lambda *_args: [FakeProtectionMatch()],
        unwrapper=fake_unwrapper,
    )

    root_reaction = cleaned[0]["children"][0]
    nested_reaction = root_reaction["children"][0]["children"][0]

    assert seen_rule_orders[0] == ("deprotect", "transform")
    assert root_reaction["smiles"] == "intermediate>>target"
    assert nested_reaction["smiles"] == "child>>intermediate"
    assert nested_reaction["children"][0]["smiles"] == CHILD
    assert list(resolved) == ["0"]
    assert unresolved == {}
    assert summary["number_of_routes_modified"] == 1
    assert (
        summary["number_of_protection_related_multicenter_reactions_split"] == 1
    )


def test_protection_introduction_multicenter_reaction_is_split_from_trace_event():
    def fake_unwrapper(target_smiles, rule_smarts, **_kwargs):
        assert tuple(rule_smarts) == ("deprotect", "transform")
        return {
            0: {
                "type": "mol",
                "smiles": target_smiles,
                "children": [
                    {
                        "type": "reaction",
                        "smiles": "protected>>target",
                        "children": [
                            {
                                "type": "mol",
                                "smiles": "deprotected",
                                "children": [
                                    {
                                        "type": "reaction",
                                        "smiles": "child>>protected",
                                        "children": [
                                            {
                                                "type": "mol",
                                                "smiles": CHILD,
                                                "children": [],
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        }

    _cleaned, resolved, unresolved, summary = preprocess_routes_json(
        [route_with_one_reaction()],
        extractor=FakeExtractor(multicenter_extraction()),
        protection_rules={},
        protection_config=ProtectionAnalysisConfig(collect_interval_rules=False),
        normalizer=identity_normalizer,
        protection_detector=lambda *_args: [],
        protection_event_analyzer=lambda *_args, **_kwargs: (
            [FakeProtectionEvent()],
            [],
            None,
        ),
        unwrapper=fake_unwrapper,
    )

    assert list(resolved) == ["0"]
    assert unresolved == {}
    assert summary["number_of_routes_modified"] == 1


def test_unresolved_routes_and_summary_are_written(tmp_path):
    def failing_unwrapper(*_args, **_kwargs):
        raise RuntimeError("cannot split")

    input_path = tmp_path / "n1_routes.json"
    input_path.write_text(json.dumps([route_with_one_reaction()]), encoding="utf-8")
    output_paths = dataset_output_paths(tmp_path / "clean", "n1_routes.json")

    summary = preprocess_routes_file(
        input_path,
        output_paths,
        extractor=FakeExtractor(multicenter_extraction()),
        protection_rules={},
        protection_config=ProtectionAnalysisConfig(collect_interval_rules=False),
        ignore_errors=True,
        normalizer=identity_normalizer,
        protection_detector=lambda *_args: [],
        unwrapper=failing_unwrapper,
    )

    cleaned = json.loads(output_paths["cleaned"].read_text())
    unresolved = json.loads(output_paths["unresolved"].read_text())
    resolved = json.loads(output_paths["resolved"].read_text())
    written_summary = json.loads(output_paths["summary"].read_text())

    assert cleaned[0]["metadata"]["route_preprocessing"]["status"] == "unresolved"
    assert list(unresolved) == ["0"]
    assert resolved == {}
    assert summary["number_of_unresolved_multicenter_routes"] == 1
    assert written_summary["total_routes_processed"] == 1
    assert output_paths["summary"].exists()
