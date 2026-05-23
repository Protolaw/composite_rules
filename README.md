# Route Inspector

<p align="center">
  <img src="docs/assets/route-inspector-demo.svg" alt="Route Inspector workflow: route JSON to composite rules, alchemical rules, SVG unwrapping, scoring, and protection analysis" width="880">
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <img alt="CLI" src="https://img.shields.io/badge/interface-CLI%20%2B%20notebooks-informational.svg">
</p>

**Route Inspector finds reusable multi-step chemistry patterns hidden inside
retrosynthetic route trees.** Feed it PaRoutes/SynPlanner-style route JSON and
get composite route rules, alchemical pseudo-reactions, SVG route depictions,
Retro-BLEU-like overlap scores, and protection/deprotection traces.

Route Inspector is a standalone SynPlanner/chython toolkit for route-level rule
analysis. It works with PaRoutes-style `mol`/`reaction` route trees and uses
SynPlanner rule extraction instead of RDKit/AiZynthFinder templates.

## Why use it?

Most retrosynthesis tooling focuses on individual reaction templates. Route
Inspector asks route-level questions:

- Which **contiguous reaction-center-sharing rule sequences** repeat across a
  route collection?
- Which multi-step rules can be unwrapped back into **route JSON/SVG** so a
  chemist can inspect them visually?
- Which composite-rule applications collapse into **alchemical pseudo-reactions**?
- Which discovered route fragments overlap a reference route vocabulary?
- Where are **protecting groups** introduced, inherited from stock, or removed?

## What it does

The package can:

- extract one-step rules and ordered composite rules from synthesis routes;
- keep only contiguous rule sequences whose adjacent reaction centers are shared
  or chemically connected by the route molecule;
- unwrap composite or alchemical rules back into route JSON/SVG depictions;
- collect alchemical rules by collapsing unwrapped routes into pseudo-reactions;
- classify alchemical rules against default one-step rules by QueryCGR identity;
- score extracted composite-rule overlap against reference route JSON files;
- analyze protection/deprotection strategies in mapped routes.

## Installation

Use an environment where `synplan` and `chython-synplan` are available. From the
repository root:

```bash
conda activate synplan
python -m pip install -e .
route-inspector --help
```

For development without installation, run the module directly:

```bash
python -m route_analysis.cli --help
```

## Quick start

Extract composite rules from a PaRoutes-style route JSON:

```bash
route-inspector extract-composite-rules \
  --routes-json data/n1-routes.json \
  --output comp_output/n1/n1.tsv \
  --config configs/rule_extraction_functional_groups.yaml \
  --ignore-errors
```

The command writes `*_t1_single_rules.tsv`, `*_t2_composite_rules.tsv`, larger
composite-rule TSVs, and a JSON extraction summary.

Apply a composite rule back to a target molecule and write an inspectable route
depiction:

```bash
route-inspector unwrap-composite-rule \
  --smiles 'CCO' \
  --composite-rule-tsv comp_output/n1/n1_t2_composite_rules.tsv \
  --row 0 \
  --output-json demo_out/unwrapped_route.json \
  --output-svg demo_out/unwrapped_route.svg
```

## Outputs at a glance

| Task | Command | Main outputs |
| --- | --- | --- |
| Composite route-rule extraction | `extract-composite-rules` | `*_single_rules.tsv`, `*_composite_rules.tsv`, extraction summary JSON |
| Alchemical rule collection | `extract-alchemical-rules` | alchemical rule TSV, pseudo-reaction `.smi`, summary/error sidecars |
| Rule classification | `classify-alchemical-rules` | positive/negative alchemical-rule TSV splits |
| Route unwrapping | `unwrap-composite-rule`, `unwrap-alchemical-rule` | route JSON and SVG depictions |
| Reference overlap scoring | `score-composite-overlap` | unique-rule, coverage, Jaccard, and popularity-weighted overlap scores |
| Protection analysis | `analyze-protection` | event tables, rule-family summaries, trace failures, network edges, summary JSON |

## Documentation

- `docs/composite_rules.rst`: extraction principles, output schema, large-run
  options, and composite unwrapping.
- `docs/alchemical_rules.rst`: alchemical extraction, pseudo-reaction output,
  classification, and alchemical unwrapping.
- `docs/protection_analysis.rst`: protection/deprotection tracing outputs and
  options.
- `docs/scoring.rst`: Retro-BLEU-style overlap scoring and positive/negative
  classification-aware metrics.
- `docs/tutorials.rst`: notebook map and recommended tutorial order.

Interactive examples live in `tutorials/`.

## Suggested repository topics

`cheminformatics`, `retrosynthesis`, `synthesis-planning`,
`reaction-informatics`, `synplanner`, `chython`, `chemical-reactions`,
`route-analysis`, `reaction-templates`, `python`

## Status

Route Inspector is early research software. The CLI is usable today, while the
packaging and demo assets are being improved to make installation,
reproducibility, and visual inspection smoother for new users.
