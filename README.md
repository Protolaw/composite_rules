# Composite Rules

This is a small standalone extractor for route-level composite rules. It reads
PaRoutes-style `mol`/`reaction` route trees, extracts individual reaction rules
with SynPlanner/chython, and writes composite rule sequences to TSV.

The extractor only keeps contiguous ordered rule sequences where each adjacent
reaction pair shares at least one reaction-center atom. For a valid ordered
chain `t1, t2, t3`, the output includes `t1$t2`, `t2$t3`, and `t1$t2$t3`, but
does not include the non-contiguous `t1$t3`.

Reaction-center sharing is computed on the shared route molecule between two
adjacent reactions. The extractor projects the parent reaction center from the
parent reactant side and the child reaction center from the child product side
onto that shared molecule, then checks for overlapping centers or constrained
functional-group center shifts.

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

Routes that are processed successfully but do not yield any composite rules are
written to:

```text
n1_routes_without_composite_rules.json
```

This file is a JSON route lookup, `{route_id: route_tree}`, so it can be used
directly with `get_route_svg_from_json`. Each route root also gets
`metadata.composite_rule_extraction` with the target SMILES, per-route reaction
counts, and a coarse reason such as `no_reactions`,
`fewer_than_min_length_extracted_reactions`, or
`no_reaction_center_sharing_sequence`. Use
`--routes-without-composites-output` to choose a different path.

## Usage

The CLIs no longer take a `--synplanner-root` argument. Run them from an
environment where `synplan`, `chython`, and `chython-synplan` are importable
(`conda activate synplan` in the current setup).

From `/Users/almazgil/Desktop/projects/Retro-BLEU`:

```bash
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli extract-composite-rules \
  --routes-json PaRoutes/data/n1-routes.json \
  --output composite_rules/output/n1.tsv
```

For a quick smoke test:

```bash
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli extract-composite-rules \
  --routes-json PaRoutes/data/n1-routes.json \
  --output /private/tmp/n1.tsv \
  --limit 100
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
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli unwrap-composite-rule \
  --smiles 'CCO' \
  --composite-rule '[C:1]-[O:2]>>[C:1].[O:2]' \
  --output-json /private/tmp/unwrapped_route.json \
  --output-svg /private/tmp/unwrapped_route.svg
```

You can also read the composite rule from one of the extractor TSV files:

```bash
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli unwrap-composite-rule \
  --smiles 'CCO' \
  --composite-rule-tsv /private/tmp/n1_t2_composite_rules.tsv \
  --row 0 \
  --output-svg /private/tmp/unwrapped_route.svg
```

## Alchemical Rules

Alchemical rules collapse an unwrapped composite route into one pseudo-reaction:
final in-stock leaves are the reactants and the starting molecule is the product.
The pseudo-reactions are then passed through SynPlanner rule extraction again.

```bash
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli extract-alchemical-rules \
  --composite-rule-tsv \
    /private/tmp/n1_t2_composite_rules.tsv \
    /private/tmp/n1_t3_composite_rules.tsv \
    /private/tmp/n1_t4_composite_rules.tsv \
    /private/tmp/n1_t5_composite_rules.tsv \
  --output /private/tmp/alchemical_out \
  --config composite_rules/configs/rule_extraction_functional_groups.yaml \
  --ignore-errors
```

If you are already inside `/Users/almazgil/Desktop/projects/Retro-BLEU/composite_rules`,
use paths relative to that directory:

```bash
PYTHONPATH=. \
conda run -n synplan python -m alchems.cli extract-alchemical-rules \
  --composite-rule-tsv ./comp_output \
  --output ./alchemical_out \
  --config configs/rule_extraction_functional_groups.yaml \
  --ignore-errors
```

The `--composite-rule-tsv` argument accepts either individual TSV files or a
directory containing `*_composite_rules.tsv` files.
The `--output` argument accepts either a TSV file or a directory. When a
directory is given, the collector writes `<prefix>_alchemical_rules.tsv`,
`<prefix>_alchemical_reactions.smi`, and the summary/error sidecars there.
Shell line continuations must be a final `\` character with no trailing space.

Alchemical rule duplicates are merged by comparing the composed QueryCGR
objects, not by raw SMARTS strings. This avoids splitting equivalent rules only
because their atom-map numbers render differently. Skipped unwrap applications
are written to the error sidecar with a compact schema: `row_index`,
`Target_smiles`, `Composite_rule`, `source_tsv`, `Composite_size`, and
`Route_ids`.

The alchemical TSV links each alchemical rule back to the composite rules that
generated it:

```text
Alchemical_rule	output_reactants_num	popularity	route_ids_size	Reference	Target_molecules	composite_rules_size	Composite_rule_sizes	Composite_rules	Source_composite_rows	pseudo_reactions_size	Pseudo_reaction_ids
```

The `.smi` file contains mapped pseudo-reactions in the first column, followed
by pseudo-reaction id, alchemical rule id, route references, target molecule,
source composite size, and source TSV row.

To apply one alchemical rule directly to a target:

```bash
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli unwrap-alchemical-rule \
  --smiles 'CCO' \
  --alchemical-rule-tsv /private/tmp/n1_alchemical_rules.tsv \
  --row 0 \
  --output-json /private/tmp/alchemical_unwrapped_route.json \
  --output-svg /private/tmp/alchemical_unwrapped_route.svg
```

## Protection Analysis

Protection analysis detects curated chython deprotection events in route steps,
traces the protected atom/group backward through the mapped route reactions,
and reports whether the protecting group was introduced during the route or
came from an in-stock precursor.

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

The command prints progress every 100 processed routes by default. If you use
`conda run`, add `--no-capture-output` to see progress live, or run
`python -m alchems.cli ...` directly from an activated `synplan` environment.

The command writes route stats, event rows, interval composite-rule rows,
protecting-group summaries, rule-family summaries, network edges, trace
failures, and a JSON summary. See
`tutorials/analyze_protection.ipynb` for a route-level walkthrough with
`get_route_svg_from_json`.

To classify alchemical rules, provide a default SynPlanner rule TSV. A rule is
negative when its QueryCGR matches any default rule; otherwise it is positive.

```bash
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli classify-alchemical-rules \
  --alchemical-rules-tsv /private/tmp/n1_alchemical_rules.tsv \
  --default-rules-tsv /path/to/default_reaction_rules.tsv \
  --output /private/tmp/n1_classified_alchemical_rules.tsv
```

## Scoring

The scoring CLI measures order-sensitive overlap between extracted composite
rule TSV files and composite rules extracted from a reference route JSON.
This follows the same high-level idea as Retro-BLEU: extract ordered adjacent
route fragments, convert reactions into reusable rule/template identifiers, and
score how many extracted fragments occur in a reference vocabulary. The main
differences are that this project uses SynPlanner/chython rule SMARTS instead
of RDKit/AiZynthFinder templates, keeps only reaction-center-sharing fragments,
and reads the reference directly from route JSON instead of a precomputed pickle.

```bash
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli score-composite-overlap \
  --extracted-tsv composite_rules/comp_output/n1 \
  --reference-routes-json PaRoutes/data/n1-routes.json \
  --output /private/tmp/composite_rule_scores
```

If you provide classified alchemical rules, the scorer also reports
classification-aware overlap:

```bash
PYTHONPATH=composite_rules \
conda run -n synplan python -m alchems.cli score-composite-overlap \
  --extracted-tsv composite_rules/comp_output/n1 \
  --reference-routes-json PaRoutes/data/n1-routes.json \
  --classification-tsv composite_rules/reference/n1_classified_alchemical_rules.tsv \
  --output /private/tmp/composite_rule_scores
```

`pos_overlap` is the extracted-popularity-weighted overlap assigned to positive
classified alchemical rules. `neg_overlap` is the same quantity assigned to
negative classified alchemical rules. For a composite rule linked to both
positive and negative alchemical rows, its contribution is split by the
positive/negative classification weights from the classified alchemical TSV.
Use `pos_overlap` as a reward term and `neg_overlap` as a penalty term.
