# Tutorials

Run examples from `/Users/almazgil/Desktop/projects/Retro-BLEU` unless a block
explicitly changes directory. SynPlanner is expected to be importable from the
active `synplan` environment; the CLIs do not take `--synplanner-root`.

## 1. Extract Composite Rules

```bash
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli extract-composite-rules \
  --routes-json PaRoutes/data/n1-routes.json \
  --output composite_rules/comp_output/n1/n1.tsv \
  --config composite_rules/configs/rule_extraction_functional_groups.yaml \
  --ignore-errors
```

Expected outputs:

```text
composite_rules/comp_output/n1/n1_t2_composite_rules.tsv
composite_rules/comp_output/n1/n1_t3_composite_rules.tsv
composite_rules/comp_output/n1/n1_t4_composite_rules.tsv
composite_rules/comp_output/n1/n1_t5_composite_rules.tsv
composite_rules/comp_output/n1/n1_composite_rule_extraction_summary.json
```

## 2. Unwrap One Composite Rule

```bash
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli unwrap-composite-rule \
  --smiles 'CCO' \
  --composite-rule-tsv composite_rules/comp_output/n1/n1_t2_composite_rules.tsv \
  --row 0 \
  --output-json /private/tmp/composite_unwrapped_route.json \
  --output-svg /private/tmp/composite_unwrapped_route.svg
```

## 3. Collect Alchemical Rules

```bash
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli extract-alchemical-rules \
  --composite-rule-tsv composite_rules/comp_output/n1 \
  --output composite_rules/alchemical_out \
  --config composite_rules/configs/rule_extraction_functional_groups.yaml \
  --ignore-errors
```

When `--output` is a directory, the collector derives all sidecar filenames:

```text
composite_rules/alchemical_out/n1_alchemical_rules.tsv
composite_rules/alchemical_out/n1_alchemical_reactions.smi
composite_rules/alchemical_out/n1_alchemical_rule_collection_summary.json
```

## 4. Classify Positive/Negative Rules

```bash
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli classify-alchemical-rules \
  --alchemical-rules-tsv composite_rules/alchemical_out/n1_alchemical_rules.tsv \
  --default-rules-tsv /path/to/default_reaction_rules.tsv \
  --output /private/tmp/n1_classified_alchemical_rules.tsv
```

## 5. Analyze Protection Strategies

```bash
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli analyze-protection \
  --routes-json PaRoutes/data/n1-routes.json \
  --composite-rule-tsv composite_rules/comp_output/n1 \
  --output-dir composite_rules/protection_out/n1 \
  --config composite_rules/configs/protection_analysis.yaml \
  --include-multicenter \
  --deprotection-first \
  --querycgr-compare \
  --ignore-errors
```

See `analyze_protection.ipynb` for a visual single-route example and the
pool-level summary tables.

## 6. Score Overlap

```bash
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli score-composite-overlap \
  --extracted-tsv composite_rules/comp_output/n1 \
  --reference-routes-json PaRoutes/data/n1-routes.json \
  --classification-tsv /private/tmp/n1_classified_alchemical_rules.tsv \
  --output /private/tmp/composite_rule_overlap_scores
```

The scoring output reports unique-rule overlap, reference coverage, Jaccard,
popularity-weighted extracted overlap, and optional classification-aware
`pos_overlap`/`neg_overlap` reward and penalty ratios.
