# Pipeline source

Both datasets use these modules. Dataset-specific choices belong in the YAML;
the root `main.py` connects the stages.

| Module | Responsibility |
|---|---|
| `data.py` | Load a CSV without altering source files |
| `cleaning.py` | Configured placeholder handling, numeric validation, category normalisation and training deduplication |
| `preprocess.py` | Row-wise feature preparation, stratified split and unfitted feature transformer |
| `model.py` | Existing classifier registry, fold-local selectors and pipeline construction |
| `evaluate.py` | Scorers, out-of-fold evaluation, fairness and corrected paired tests |
| `tuning.py` | Optuna search and nested cross-validation |
| `inference.py` | Prepare unlabelled rows without dropping observations and export predictions |
| `results.py` | Output layout, config/report snapshots and append-only scoreboard |

Imputation, scaling, encoding and feature selection learn from training folds
inside the sklearn pipeline. New-row inference reuses the fitted pipeline.
The reserved test split stays outside model comparison and tuning.

The recency helper at the end of `preprocess.py` is an existing separate draft;
neither COMPAS nor Adult calls it. Its original manual check is preserved in
[`docs/archive/recency/`](../docs/archive/recency/README.md).
