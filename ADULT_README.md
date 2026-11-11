# Adult income data (for the "new dataset" challenge)

**Task:** predict whether a person earns more than 50,000 USD a year (`income_over_50k`: 1 = yes, 0 = no) from census information.

| File | Rows | Target |
|---|---|---|
| [`data_adult/adult_train.csv`](data_adult/adult_train.csv) | 6,000 people | yes (`income_over_50k`) |
| [`data_adult/adult_new.csv`](data_adult/adult_new.csv) | 2,000 new people | no: these are the rows your final model predicts |

Each row has an `id`. About 24% of people in `adult_train.csv` earn over 50K.

## Columns

| Column | Description |
|---|---|
| `id` | row identifier (not a feature) |
| `age` | age in years |
| `workclass` | type of employer (private, self-employed, government, ...) |
| `fnlwgt` | census sampling weight: how many people in the population this row represents |
| `education` | highest education level, as text |
| `education_num` | highest education level, as a number |
| `marital_status` | marital status |
| `occupation` | type of occupation |
| `relationship` | role in the household (husband, wife, own child, not in family, ...) |
| `race` | race |
| `sex` | sex |
| `capital_gain` | capital gains in the year |
| `capital_loss` | capital losses in the year |
| `hours_per_week` | usual working hours per week |
| `native_country` | country of origin |
| `v01` ... `v25` | additional anonymised attributes supplied with the data; their meaning is not documented |
| `income_over_50k` | **target** (train file only) |

Look at the data before configuring the pipeline: some columns have placeholder values, missing values or text that needs cleaning, and some columns may not be useful at all.

**Source:** derived from the UCI Adult dataset (Becker & Kohavi, 1996; 1994 US Census), licensed CC BY 4.0. Subsampled and extended for this course.

__________

**Week 9: Beyond COMPAS — Option A**

This week, I adapted my existing pipeline to the Adult income dataset, predicting whether a person earns more than 50,000 USD per year. I kept the same reusable preprocessing, models, feature selection, evaluation and Optuna tuning code. The new settings are in [config_adult.yaml](config_adult.yaml), while `python main.py` still runs COMPAS with its original config.

### *What broke and how I fixed it*

| Problem | Why it happened | Fix |
|---|---|---|
| The entry point always loaded COMPAS | The config path was fixed | Added an optional config argument. |
| Offence features and age rules did not fit Adult | The old recipe used COMPAS columns and rejected age 17 | Disabled offence feature engineering and configured Adult's own columns and ranges. |
| Fairness reporting required a COMPAS score | Adult has no `score_text` | Audit the configured sensitive attribute and only include the COMPAS comparison when available. |
| New people have no target | Training preparation expects labels and removes duplicates | Added prediction preparation using the same configured cleaning, without splitting or removing rows. |

I converted placeholders into missing values, normalised categorical text and added missingness indicators for workclass and occupation. Cleaning preserves the valid age-17 records and the capital-gain top code 99999. I audit **sex**, exclude both sex and race from model inputs, and also exclude `id`, the census weight `fnlwgt` and redundant `education`. All 25 anonymised columns remain available to fold-local feature selection.

### *Model comparison and hyperparameter tuning*

I first compared all 11 existing model families without tuning, then tuned Logistic Regression, Random Forest, HistGradientBoosting and XGBoost with 15 trials per search. I repeated the two strongest candidates with 30 trials. Each tuned run used five stratified outer folds and five inner folds, ROC-AUC selection and seed 42. Preprocessing and kbest selection were fitted inside the training folds. The same 4,800 development rows were used throughout; the 1,200 reserved test rows were not evaluated.

| Model | Trials per search | Nested ROC-AUC | Accuracy | Class 1 F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 15 | 0.8989 | 0.8467 | 0.6451 |
| Random Forest | 15 | 0.9040 | 0.8540 | 0.6469 |
| HistGradientBoosting | 15 | 0.9089 | 0.8600 | 0.6710 |
| XGBoost | 15 | 0.9092 | 0.8598 | 0.6740 |
| **HistGradientBoosting** | **30** | **0.9096** | **0.8617** | **0.6774** |
| XGBoost | 30 | 0.9093 | 0.8575 | 0.6637 |

Both 30-trial models reached all three targets: ROC-AUC at least 0.90, accuracy at least 0.855 and F1 at least 0.66. HistGradientBoosting is my preferred candidate from this comparison, with nested ROC-AUC **0.9096 ± 0.0160**. Its advantage over XGBoost is small, so these results do not establish a statistically clear winner. Increasing the budget slightly improved HistGradientBoosting, while XGBoost's accuracy and F1 decreased despite a small ROC-AUC increase.

### *Features kept and predictions*

Both 30-trial final refits retained **16 of 137 encoded columns**, with no anonymised `v` columns. They kept age, education number, capital gain, capital loss and hours per week, plus encoded categories from workclass, marital status, occupation and relationship. These are the final refit's features; selection was learned separately inside each fold, so the outer folds did not all use that same list. The exact names are saved in each run's `_features.json` file.

Each final pipeline was refitted on development rows and produced probabilities and 0/1 predictions for all **2,000 new people**, preserving their IDs and order. The predictions for my preferred candidate are in [hist_gradient_boosting/predictions.csv](results_adult/hist_gradient_boosting/predictions.csv), and the full comparisons are in [runs.csv](results_adult/runs.csv).

### *Running the pipeline*

```bash
python main.py config_adult.yaml
python -m scripts.run_adult --untuned
python -m scripts.run_adult --models logistic_regression random_forest hist_gradient_boosting xgboost
python -m scripts.run_adult --models hist_gradient_boosting xgboost --trials 30
```

The hardest assumption to find was that the fairness report always expected a COMPAS score. Moving the dataset-specific choices into the YAML allowed the same pipeline to handle Adult. Comparing several models and keeping selection inside CV helped me reach the targets, while the extra tuning showed that more trials do not improve every metric. These comparisons used the same development data repeatedly, so I treat the differences as exploratory and keep the reserved test set aside.

#### have a nice weekend!
