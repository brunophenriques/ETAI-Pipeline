# Automated checks

Run from the repository root in the virtual environment:

```bash
python -m unittest discover -s tests
```

The suite covers preprocessing, imputation comparison, engineered features,
feature selection, evaluation, tuning and entry-point routing. Adult checks
also verify the expected cleaning counts, feature roles, row preservation,
optional COMPAS fairness baseline and sequential model configuration.

Tests use small synthetic inputs where possible. The Adult cleaning check reads
the supplied local CSVs. Tests do not launch full experiment searches or evaluate
the reserved test set.

The unrelated manual recency draft was preserved under
[`docs/archive/recency/`](../docs/archive/recency/README.md); it is not an integrated
test of this pipeline.
