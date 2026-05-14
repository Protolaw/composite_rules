from composite_rules.unwrap import split_composite_rule, unwrap_composite_rule


def test_split_composite_rule_strips_empty_parts():
    assert split_composite_rule(" a $ $ b ") == ["a", "b"]


def test_unwrap_composite_rule_builds_route_tree():
    route = unwrap_composite_rule(
        "CCO",
        "[C:1]-[O:2]>>[C:1].[O:2]$[C:1]-[C:2]>>[C:1].[C:2]",
        route_id=7,
    )

    root = route[7]
    assert root["smiles"] == "CCO"
    assert root["children"][0]["rule_key"] == "composite:1"
    first_step_products = root["children"][0]["children"]
    assert [node["smiles"] for node in first_step_products] == ["CC", "O"]
    assert first_step_products[0]["children"][0]["rule_key"] == "composite:2"
    assert first_step_products[1]["in_stock"] is True
