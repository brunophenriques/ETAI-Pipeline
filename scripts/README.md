# Experiment runners

Run from the repository root, with the virtual environment active.
The normal single-model entry point remains `python main.py`.

| Experiment | Command | Evidence |
|---|---|---|
| Adult: sequential model comparison | `python -m scripts.run_adult` | `results_adult/<model>/`, `results_adult/runs.csv` |
| Week 4: median versus KNN | `python -m scripts.compare_imputation` | `results/week4/imputation_comparison/` |
| Week 6: ensembles | `python -m scripts.run_week6_comparison` | `results/week6/ensembles/` |
| Week 6: engineered features | `python -m scripts.run_feature_comparison` | `results/week6/feature_engineering/` |
| Week 7: all models | `python -m scripts.run_week7` | `results/week7/<model>/`, `results/runs.csv` |
| Week 8: voting selection | `python -m scripts.run_week8` | `results/week8/selection_on/`, `selection_off/` and comparison files |
| Week 8 B1: compare selectors | `python -m scripts.run_selectors` | `results/week8/b1/<method>/` and paired tests |

The weekly runners use in-memory configuration overrides and preserve config.yaml.
Completed reports are reused when resuming. Their budgets remain specific to each
week: Week 6 uses accuracy and 50 trials; Weeks 7-8 use ROC-AUC and 15 trials.
Archive a completed experiment folder before intentionally starting it again.
Reusable implementation belongs in `src/`; these scripts orchestrate experiments.

Adult uses the single `config_adult.yaml` without modifying it. The runner keeps
its preprocessing, selection, folds and tuning settings; Dummy disables tuning.
Each model gets its own predictions CSV. Start with a quick comparison:

```powershell
python -m scripts.run_adult --untuned
python -m scripts.run_adult --models logistic_regression random_forest xgboost
```

Omit `--untuned` to use tuning as configured. `--trials 5` overrides the search
budget for this command. Running again appends new results to the scoreboard.
