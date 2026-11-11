# Adult dataset files

| File | Rows | Role |
|---|---:|---|
| `adult_train.csv` | 6,000 | Labelled data; target is `income_over_50k` |
| `adult_new.csv` | 2,000 | Unlabelled rows to predict, retaining every ID |

The data dictionary and Week 9 write-up are in
[ADULT_README.md](../ADULT_README.md) at the repository root.
Settings are in [config_adult.yaml](../config_adult.yaml), and generated
outputs are in [results_adult/](../results_adult/README.md).
Keep these source CSVs unchanged; cleaning happens in the pipeline.
