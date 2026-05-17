from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProtectionRule:
    """One curated protecting-group cleavage rule adapted from chython."""

    rule_id: str
    name: str
    query: Any
    product_query: Any
    product_smarts: str
    atoms_to_keep: tuple[int, ...]
    atoms_to_add: tuple[tuple[Any, ...], ...]
    protected_example: str
    cleaved_example: str
    decoys: tuple[str, ...]
    source: str


def _product_smarts_from_atoms_to_keep(atoms_to_keep: tuple[int, ...]) -> str:
    if not atoms_to_keep:
        return ""
    return ".".join(f"[A:{atom_number}]" for atom_number in atoms_to_keep)


def _mapped_atoms_from_product_smarts(product_smarts: str) -> tuple[int, ...]:
    atoms = tuple(int(value) for value in re.findall(r":(\d+)\]", product_smarts))
    return atoms


def _adapt_protective_rules(rules: Any, *, source: str) -> dict[str, ProtectionRule]:
    from chython import smarts

    out: dict[str, ProtectionRule] = {}
    for name, value in rules.items():
        query, atoms_to_keep, atoms_to_add, protected, cleaved, decoys = value
        keep = tuple(int(atom_number) for atom_number in atoms_to_keep)
        product_smarts = _product_smarts_from_atoms_to_keep(keep)
        rule_id = str(name)
        out[rule_id] = ProtectionRule(
            rule_id=rule_id,
            name=str(name),
            query=query,
            product_query=smarts(product_smarts),
            product_smarts=product_smarts,
            atoms_to_keep=keep,
            atoms_to_add=tuple(tuple(item) for item in atoms_to_add),
            protected_example=str(protected),
            cleaved_example=str(cleaved),
            decoys=tuple(str(decoy) for decoy in decoys),
            source=source,
        )
    return out


def _load_from_algorithms_groups() -> dict[str, ProtectionRule]:
    from chython.algorithms.groups._protective import rules

    return _adapt_protective_rules(rules, source="chython.algorithms.groups._protective")


def _load_from_reactor_deprotection() -> dict[str, ProtectionRule]:
    from chython import smarts
    from chython.reactor import deprotection

    out: dict[str, ProtectionRule] = {}
    names = [
        name
        for name in getattr(deprotection, "__all__", ())
        if name != "apply_all" and hasattr(deprotection, f"_{name}")
    ]
    for name in names:
        rule_entries = getattr(deprotection, f"_{name}", ())
        for index, entry in enumerate(rule_entries):
            query_smarts, product_smarts, *examples = entry
            protected = examples[0] if len(examples) >= 1 else ""
            cleaved = examples[1] if len(examples) >= 2 else ""
            decoys = examples[2:] if len(examples) > 2 else []
            keep = _mapped_atoms_from_product_smarts(product_smarts)
            rule_id = name if len(rule_entries) == 1 else f"{name}#{index + 1}"
            out[rule_id] = ProtectionRule(
                rule_id=rule_id,
                name=name,
                query=smarts(query_smarts),
                product_query=smarts(product_smarts),
                product_smarts=product_smarts,
                atoms_to_keep=keep,
                atoms_to_add=(),
                protected_example=str(protected),
                cleaved_example=str(cleaved),
                decoys=tuple(str(decoy) for decoy in decoys),
                source="chython.reactor.deprotection",
            )
    return out


def _load_from_local_protective() -> dict[str, ProtectionRule]:
    from alchems.protection import chython_protective as module

    return _adapt_protective_rules(
        module.rules,
        source="alchems.protection.chython_protective",
    )


def load_chython_protection_rules() -> dict[str, ProtectionRule]:
    """Load the curated protecting-group rules available in this environment.

    The project-local ``alchems.protection.chython_protective`` copy is tried
    first to avoid environment-version differences in chython. Newer chython
    builds can also expose ``chython.algorithms.groups._protective.rules``;
    older SynPlanner environments expose similar definitions through
    ``chython.reactor.deprotection``.
    """

    loaders = (
        _load_from_local_protective,
        _load_from_algorithms_groups,
        _load_from_reactor_deprotection,
    )
    errors: list[Exception] = []
    for loader in loaders:
        try:
            rules = loader()
        except Exception as exc:
            errors.append(exc)
            continue
        if rules:
            return rules

    message = "; ".join(f"{type(exc).__name__}: {exc}" for exc in errors)
    raise RuntimeError(f"could not load chython protection rules: {message}")
