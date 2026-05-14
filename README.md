# Composite Rules

This is a small standalone extractor for route-level composite rules. It reads
PaRoutes-style `mol`/`reaction` route trees, extracts individual reaction rules
with SynPlanner/chython, and writes composite rule sequences to TSV.

The extractor only keeps contiguous ordered rule sequences where each adjacent
reaction pair shares at least one reaction-center atom. For a valid ordered
chain `t1, t2, t3`, the output includes `t1$t2`, `t2$t3`, and `t1$t2$t3`, but
does not include the non-contiguous `t1$t3`.

Reaction-center sharing is computed as a non-empty intersection between the
adjacent standardized SynPlanner/chython reaction CGR `center_atoms` sets.

## Output

The extractor writes one TSV per composite-rule size. For an output prefix like
`n1`, filenames are:

```text
n1_t2_composite_rules.tsv
n1_t3_composite_rules.tsv
...
```

The TSV schema is:

```text
Composite_rule	popularity	route_ids_size	Reference	Target_molecules
```

`Composite_rule` is a `$`-separated sequence of SynPlanner rule SMARTS.
`Reference` is a comma-separated list of route IDs where that sequence occurs.
`popularity` and `route_ids_size` are `len(Reference)`. `Target_molecules` is
the comma-separated list of unique molecules where that composite rule starts;
for subsequences this can be an intermediate molecule rather than the full route
target. Rows are sorted by descending popularity.

A JSON summary is also written:

```text
n1_composite_rule_extraction_summary.json
```

## Usage

From `/Users/almazgil/Desktop/projects/Retro-BLEU`:

```bash
PYTHONPATH=composite_rules/src:SynPlanner \
conda run -n synplan python -m composite_rules.extract \
  --routes-json PaRoutes/data/n1-routes.json \
  --output composite_rules/output/n1.tsv \
  --synplanner-root SynPlanner
```

For a quick smoke test:

```bash
PYTHONPATH=composite_rules/src:SynPlanner \
conda run -n synplan python -m composite_rules.extract \
  --routes-json PaRoutes/data/n1-routes.json \
  --output /private/tmp/n1.tsv \
  --limit 100 \
  --synplanner-root SynPlanner
```

By default the extractor uses SynPlanner rule extraction with reactor validation
disabled for speed and to avoid dropping otherwise parseable chython rules. Add
`--reactor-validation` if you want SynPlanner reactor validation during
per-reaction rule extraction.

## Unwrapping

The unwrapping script applies each `$`-separated rule in a composite rule to a
target molecule and writes a route JSON that is compatible with
`get_route_svg_from_json`.

```bash
PYTHONPATH=composite_rules/src:SynPlanner \
conda run -n synplan python -m composite_rules.unwrap \
  --smiles 'CCO' \
  --composite-rule '[C:1]-[O:2]>>[C:1].[O:2]' \
  --synplanner-root SynPlanner \
  --output-json /private/tmp/unwrapped_route.json \
  --output-svg /private/tmp/unwrapped_route.svg
```

You can also read the composite rule from one of the extractor TSV files:

```bash
PYTHONPATH=composite_rules/src:SynPlanner \
conda run -n synplan python -m composite_rules.unwrap \
  --smiles 'CCO' \
  --composite-rule-tsv /private/tmp/n1_t2_composite_rules.tsv \
  --row 0 \
  --synplanner-root SynPlanner \
  --output-svg /private/tmp/unwrapped_route.svg
```

## Alchemical Rules

Alchemical rules collapse an unwrapped composite route into one pseudo-reaction:
final in-stock leaves are the reactants and the starting molecule is the product.
The pseudo-reactions are then passed through SynPlanner rule extraction again.

```bash
PYTHONPATH=composite_rules/src:SynPlanner \
conda run -n synplan python -m composite_rules.alchemical \
  --composite-rule-tsv \
    /private/tmp/n1_t2_composite_rules.tsv \
    /private/tmp/n1_t3_composite_rules.tsv \
    /private/tmp/n1_t4_composite_rules.tsv \
    /private/tmp/n1_t5_composite_rules.tsv \
  --output /private/tmp/alchemical_out \
  --synplanner-root SynPlanner \
  --config composite_rules/configs/rule_extraction_functional_groups.yaml \
  --ignore-errors
```

If you are already inside `/Users/almazgil/Desktop/projects/Retro-BLEU/composite_rules`,
use paths relative to that directory:

```bash
PYTHONPATH=src:../SynPlanner \
conda run -n synplan python -m composite_rules.alchemical \
  --composite-rule-tsv ./comp_output \
  --output ./alchemical_out \
  --synplanner-root ../SynPlanner \
  --config configs/rule_extraction_functional_groups.yaml \
  --ignore-errors
```

The `--composite-rule-tsv` argument accepts either individual TSV files or a
directory containing `*_composite_rules.tsv` files.
The `--output` argument accepts either a TSV file or a directory. When a
directory is given, the collector writes `<prefix>_alchemical_rules.tsv`,
`<prefix>_alchemical_reactions.smi`, and the summary/error sidecars there.
Shell line continuations must be a final `\` character with no trailing space.

The alchemical TSV links each alchemical rule back to the composite rules that
generated it:

```text
Alchemical_rule	popularity	route_ids_size	Reference	Target_molecules	composite_rules_size	Composite_rule_sizes	Composite_rules	Source_composite_rows	pseudo_reactions_size	Pseudo_reaction_ids	Alchemical_cgr
```

The `.smi` file contains mapped pseudo-reactions in the first column, followed
by pseudo-reaction id, alchemical rule id, route references, target molecule,
source composite size, and source TSV row.

To apply one alchemical rule directly to a target:

```bash
PYTHONPATH=composite_rules/src:SynPlanner \
conda run -n synplan python -m composite_rules.unwrap_alchemical \
  --smiles 'CCO' \
  --alchemical-rule-tsv /private/tmp/n1_alchemical_rules.tsv \
  --row 0 \
  --synplanner-root SynPlanner \
  --output-json /private/tmp/alchemical_unwrapped_route.json \
  --output-svg /private/tmp/alchemical_unwrapped_route.svg
```

To classify alchemical rules, provide a default SynPlanner rule TSV. A rule is
negative when its QueryCGR matches any default rule; otherwise it is positive.

```bash
PYTHONPATH=composite_rules/src:SynPlanner \
conda run -n synplan python -m composite_rules.classify_alchemical \
  --alchemical-rules-tsv /private/tmp/n1_alchemical_rules.tsv \
  --default-rules-tsv /path/to/default_reaction_rules.tsv \
  --output /private/tmp/n1_classified_alchemical_rules.tsv \
  --synplanner-root SynPlanner
```
