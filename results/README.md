# Results layout

`runs.csv` is the append-only scoreboard. Its report column links to each full
report; the associated JSON, folds, OOF predictions and trials share its filename stem.

| Folder | Contents |
|---|---|
| `week2/` to `week5/` | Older local reports, preserved but ignored by Git |
| `week6/ensembles/` | Baseline and tuned comparisons, configurations and summaries |
| `week6/feature_engineering/` | Features on/off reports and comparison files |
| `week6/boosted_lr/` | Optional boosted Logistic Regression experiment |
| `week6/archive/` | Earlier local trial reports, ignored by Git |
| `week7/<model>/` | Report, config, folds, OOF probabilities, trials and console log |
| `week7/metadata/` | Experiment plan, environment and validation record |
| `week8/selection_off/` | XGBoost without voting selection |
| `week8/selection_on/` | XGBoost with selection, including feature names and votes |
| `week8/b1/<method>/` | Comparable nested runs of vote, kbest and from_model, with timings |
| `week8/b1/` comparison files | Timing/score summary and three corrected paired tests with Holm adjustment |
| `week8/` comparison files | Run summary, paired scores, corrected test and validation |

Smoke checks are ignored under week7/smoke/ and week8/smoke/. Selected Week 6-8
evidence is tracked by Git. Reorganization changed paths, not the recorded scores.

Future runs use output.layout: `flat` writes directly to results_dir; `model`
adds the model name; `selection` adds selection_on/ or selection_off/.
