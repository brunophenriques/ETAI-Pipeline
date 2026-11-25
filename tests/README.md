# Automated checks

Run from the repository root in the virtual environment:

```bash
python -m unittest discover -s tests
```

The suite covers preprocessing, imputation comparison, engineered features,
feature selection, evaluation, tuning and entry-point routing. Adult checks
also verify the expected cleaning counts, feature roles, row preservation,
optional COMPAS fairness baseline and sequential model configuration.

Week 10 checks exercise all four imputers on unseen missing values, serialize and reload a fitted pipeline, preserve all 20 example rows and duplicate people, and reject a missing required column. Optional checks cover independent posterior seeds and probability averaging, category-assisted imputation, proximity donors, and saved library versions with CLI rejection of a mismatch.

Week 11 checks cover LOF's separate training/novelty scores, label-independent detectors, feature-name propagation through selection, and saved prediction pipelines. Optional checks verify that quantile limits come only from training rows and that anomaly removal never drops validation or prediction rows.

Tests use small synthetic inputs where possible. The Adult cleaning check reads
the supplied local CSVs. Tests do not launch full experiment searches or evaluate
the reserved test set.

The unrelated manual recency draft was preserved under
[`docs/archive/recency/`](../docs/archive/recency/README.md); it is not an integrated
test of this pipeline.
