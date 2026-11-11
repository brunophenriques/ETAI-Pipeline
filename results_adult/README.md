# Adult results

Start with the [Week 9 write-up](../ADULT_README.md) or [full scoreboard](runs.csv).
All reported validation scores use development data; the reserved test set was
not evaluated. Source CSVs live in `../data_adult/`.

## Latest saved run per model

These rows include different tuning budgets; the trial column makes that explicit.
Untuned rows use ordinary CV and tuned rows use nested CV. Earlier runs remain
in the same model folders and the append-only scoreboard.

| Model / report | Trials per search | ROC-AUC | Accuracy | Class-1 F1 |
|---|---:|---:|---:|---:|
| [gradient_boosting](gradient_boosting/gradient_boosting_20261111_121238_224406.txt) | 0 | 0.9102 | 0.8592 | 0.6683 |
| [hist_gradient_boosting](hist_gradient_boosting/hist_gradient_boosting_20261111_122557_080350.txt) | 30 | 0.9096 | 0.8617 | 0.6774 |
| [xgboost](xgboost/xgboost_20261111_122745_824765.txt) | 30 | 0.9093 | 0.8575 | 0.6637 |
| [random_forest](random_forest/random_forest_20261111_121627_545810.txt) | 15 | 0.9040 | 0.8540 | 0.6469 |
| [logistic_regression](logistic_regression/logistic_regression_20261111_121544_460492.txt) | 15 | 0.8989 | 0.8467 | 0.6451 |
| [adaboost](adaboost/adaboost_20261111_121249_903479.txt) | 0 | 0.8959 | 0.8479 | 0.6364 |
| [oblique_random_forest](oblique_random_forest/oblique_random_forest_20261111_121244_751245.txt) | 0 | 0.8862 | 0.8456 | 0.6417 |
| [adaboost_logistic_regression](adaboost_logistic_regression/adaboost_logistic_regression_20261111_121252_475223.txt) | 0 | 0.8839 | 0.8363 | 0.6559 |
| [naive_bayes](naive_bayes/naive_bayes_20261111_121231_474179.txt) | 0 | 0.8685 | 0.7652 | 0.6332 |
| [decision_tree](decision_tree/decision_tree_20261111_121233_381504.txt) | 0 | 0.7476 | 0.8087 | 0.6024 |
| [dummy](dummy/dummy_20261111_121217_395734.txt) | 0 | 0.5000 | 0.7606 | 0.0000 |

## Prediction files

- [HistGradientBoosting predictions](hist_gradient_boosting/predictions.csv): the write-up's preferred 30-trial candidate.
- [XGBoost predictions](xgboost/predictions.csv): the latest 30-trial XGBoost candidate.
- Each other model folder has its own latest `predictions.csv` from the sequential runner.
- The root [predictions.csv](predictions.csv) is a historical single-model export.
  `python main.py config_adult.yaml` replaces this file; the loop runner updates
  each model's separate predictions instead.

Every prediction export has `id`, `probability` (class 1), and `prediction`
(probability >= 0.5). Each contains the 2,000 new people in source order.
Per-model prediction files are replaced on rerun; timestamped reports accumulate.

## File notation

A report stem is `<model>_<YYYYMMDD_HHMMSS_microseconds>`. Files sharing a stem
belong to the same run:

| Suffix | Contents |
|---|---|
| `.txt` | Human-readable evaluation and fairness report |
| `.json` | Exact configuration snapshot for that run |
| `_default_folds.csv` | Untuned baseline on the outer folds |
| `_nested_folds.csv` | Tuned outer-fold metrics and selected parameters |
| `_trials.csv` | Final development-set Optuna study; not all outer-fold studies |
| `_oof.csv` | Out-of-fold probabilities, predictions and true labels |
| `_features.json` | Final refit's input/retained feature names and selector settings |

In `runs.csv`, `mean` is the selected metric's CV mean, `std` is its fold standard
deviation, and `gap` is mean training score minus validation score. `best_trial`
is the final search's selection score, not a held-out evaluation. `n_trials`
is the budget **per study**: five outer searches plus one final study. A
`comparison_id` groups matching data/preprocessing/fold recipes; check model,
selector and budget as well before interpreting a comparison.

`logs/` contains local execution logs and is ignored by Git. Config snapshots
record historical recipes; the only editable Adult YAML is at the repository root.

## Reproduce

Run from the repository root with the virtual environment active:

```bash
python main.py config_adult.yaml
python -m scripts.run_adult --untuned
python -m scripts.run_adult --models hist_gradient_boosting xgboost --trials 30
```

This index describes the saved results at cleanup time. New runs append to
`runs.csv`; use that file for the latest values after running more experiments.
