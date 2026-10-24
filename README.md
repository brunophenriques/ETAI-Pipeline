# Baseline Predictive Pipeline -- ETAI

# 20231608 Bruno Henriques

### WEEKLY CHALLENGES

**Week 2**:

DT:   Train accuracy: 0.829
      Test accuracy:  0.627
      (Slight overfit, without any parameters, wouldn't be a good model to use)

DT with  max_depth: 5 | min_samples_split: 10 | min_samples_leaf: 5 
      Train accuracy: 0.679
      Test accuracy:  0.669

LR with max_iter1000:
    Train accuracy: 0.679
    Test accuracy:  0.679

Our current best model is the simple Linear Regression with max_iter 1000, since it performs better by 0.01 our modified DT and since it doesnt present to us any signs of overfit, or any sign that is way worse than the slight less accurate DT, we can say the LR is the one that wins this bout.

___

## `END OF THIS WEEK CHALLENGES HAVE A NICE WEEKEND`

___

**Week 3**:

This week, I followed the diagnostics and requirements presented in the professor’s GitHub repository, particularly the weekly progress table. I chose not to pull and copy the completed implementation directly. Instead, I tried to understand the expected steps and create the functions myself before integrating them into the main pipeline through main.py. Because I implemented the work independently and avoided looking too closely at the completed code, some details may differ from the professor’s version or may have escaped my attention.

The dataset was cleaned before training the models. Invalid values and missing-value placeholders were converted into missing values, inconsistent categories were standardized, duplicate observations and redundant variables were removed, and missing values were imputed. Missingness indicators were also created for the variables identified as MNAR.

### *Decision Tree with max_depth: 5, min_samples_split: 10 and min_samples_leaf: 5:*

Train accuracy: 0.684 | Test accuracy: 0.665 | Train–test gap: 0.020 | Class 1 recall: 0.51 |Class 1 F1-score: 0.58

The restricted Decision Tree generalizes much better than the original unrestricted tree from Week 2. The original tree had a train accuracy of 0.829 and a test accuracy of only 0.627, showing clear overfitting. After preprocessing and restricting its complexity, its test accuracy increased to 0.665 and its train–test gap decreased from 0.202 to only 0.020.

### *Logistic Regression with max_iter: 1000 and solver: lbfgs:*

Train accuracy: 0.673 | Test accuracy: 0.665 | Train–test gap: 0.008 | Class 1 recall: 0.53 | Class 1 F1-score: 0.59

The new Logistic Regression has a slightly lower test accuracy than the Week 2 version, decreasing from 0.679 to 0.665. However, the new result is based on a cleaner and more reliable dataset. The old pipeline removed every incomplete row using dropna(), while the new pipeline preserves more observations through imputation and evaluates 1,443 test cases instead of 1,252. Therefore, the two accuracy values are not directly comparable, and the new result provides a more trustworthy estimate of performance on unseen data.

### *Current best model*

Considering all models tested so far, **Logistic Regression remains the best overall model**. The new Logistic Regression and restricted Decision Tree have the same test accuracy of 0.665, but Logistic Regression has a smaller train–test gap (0.008 compared with 0.020), slightly higher recall for class 1 (0.53 compared with 0.51), and a slightly higher class 1 F1-score (0.59 compared with 0.58).

The Week 2 Logistic Regression obtained the highest numerical test accuracy at 0.679, but it was evaluated using the older preprocessing approach and a smaller test set produced after dropping incomplete observations. For this reason, the new Logistic Regression is considered the strongest and most reliable model to date: it maintains stable performance while using cleaner variables, more observations, and a safer preprocessing process.

#### have a nice weekend!

__________

### Additional EDA findings and remaining work

During the exploratory data analysis, I found 6 records where the value in `age` did not match the corresponding `age_cat` category: defendants aged between 51 and 66 were labelled `Less than 25`. I also found 10 records where the COMPAS risk score did not agree with its textual category. For example, some records with a low `decile_score` of 1 or 2 were labelled `High`, while some records with a high score of 9 or 10 were labelled `Low`. In addition, 4 records had a negative `juvenile_total`. Since this column represents the total number of juvenile offences, it should never be negative; these cases were linked to an invalid negative value in `juv_fel_count`.

Finally, I found an ambiguity in the age categories. There are 110 records for people aged 45 labelled `Greater than 45`, while another category is named `25 - 45`. Therefore, it is unclear from the category names whether age 45 should belong to the middle category or the upper category, and this should be clarified before applying an automatic correction.

These findings were documented in `notebooks/week3/01_eda_introduction.ipynb`. Due to family issues and the deadline, I was not able to finish creating the new reusable diagnostic functions or fully update the pipeline with the additional validation checks. A future improvement would be to implement these functions and update the preprocessing/configuration so that the identified issues are handled automatically.

## Pipeline progress

| Week | Changes |
|---|---|
| 2 | Tested Logistic Regression and Decision Trees with a single train/test split. |
| 3 | Cleaned the dataset, added imputation and missingness indicators, and compared the new results with Week 2. |
| 4 | Added cross-validation, kept the test set aside, added Dummy and Random Forest, and compared median with KNN imputation. |
| 5 | Added Optuna tuning and nested cross-validation, then tuned Logistic Regression and Random Forest and compared them with the tuned Decision Tree. |
| 6 | Added AdaBoost and a projection-based oblique forest, compared the model families with 50-trial nested tuning, tested three engineered features, and compared boosted LR with plain LR. The final test set stayed aside. |
| 7 | Added Gradient Boosting, HistGradientBoosting and Naive Bayes, eight evaluation metrics, OOF probabilities and confusion matrices, and a ROC-AUC scoreboard with untuned and nested results. |

__________

**Week 4**:

This week, I added cross-validation to the pipeline so the results would not depend only on one train/test split. The preprocessing and the model are now together in the same Pipeline, meaning that each fold learns its own imputation values, encoding and scaling from its training data. This avoids using information from the validation rows when preparing the data.

The 5,771 development cases are used for the comparisons, while the 1,443 test cases are kept aside. This test set was already evaluated in previous weeks, but it was not used for this week's comparison. The classification report and fairness comparison now use out-of-fold predictions, so each person is evaluated by a model that did not train on their record.

### *Median vs KNN imputation:*

For this challenge, I compared the usual median imputation with KNN using 5 neighbors. Instead of filling every missing value with the column median, KNN uses similar records to calculate the replacement. Since it measures distances, I used StandardScaler before imputation for both versions, so a variable with larger values would not dominate just because of its scale.

I kept the same Decision Tree parameters: max_depth: 5, min_samples_split: 10, min_samples_leaf: 5 and random_state: 42. Both versions also kept the same encoding and missingness indicators. The comparison used the same 5 stratified folds with random_state: 42, and three single splits with seeds 0, 42 and 123.

| Imputation | Holdout seed 0 | Holdout seed 42 | Holdout seed 123 | CV accuracy (mean ± std) | CV train–val gap |
|---|---|---|---|---|---|
| Median | 0.655 | 0.678 | 0.676 | 0.678 ± 0.015 | +0.008 |
| KNN (5 neighbors) | 0.663 | 0.677 | 0.676 | 0.671 ± 0.011 | +0.013 |

These holdout results come from 75/25 splits of the development data, so they are different from the Week 3 test results.

### *Did the single split and CV agree?*

The answer depends on which single split we look at: KNN performed better with seeds 0 and 123, while median performed better with seed 42 (seed 123 looks equal in the table because of rounding). With cross-validation, median obtained the higher average accuracy, 0.678 compared with 0.671, and also had a smaller train–validation gap. The corrected paired comparison gave p = 0.1628 and a 95% interval of [-0.01865, +0.00444] for KNN minus median, so the difference is not clear enough to say one is definitely better. For this reason, **median stays in config.yaml**, since it is simpler and KNN did not show a clear improvement in this comparison. This conclusion is for the Decision Tree and 5 neighbors tested here; it does not mean KNN would always perform worse with other settings or models.

The paired comparison corrects for the fact that the CV folds share training data. It follows the [scikit-learn example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html), using `SE = sqrt((1/5 + n_validation/n_train) * variance_of_paired_differences)`; the corrected SE was 0.00416. There are only five paired results, so not finding a clear difference does not prove the methods are equivalent.

### *Running the comparison:*

The comparison can be repeated with:

```bash
python compare_imputation.py
```

The full results are saved in `results/imputation_comparison/`, including the scores for each fold and holdout, the settings used, and the paired comparison. This folder is ignored by Git, so the script generates the files locally and the main results are kept in the table above.

To try KNN in the normal pipeline, change `numerical_imputation` to `"knn"` in config.yaml, keep `scaler` as `"standard"` or `"robust"`, and choose `n_neighbors`. Then run `python main.py`. The current configuration uses median with standard scaling.

### *Current best model*

Considering the four models tested with the same preprocessing and the same 5 CV folds, **the restricted Decision Tree is now my current best model by average accuracy**. With max_depth: 5, min_samples_split: 10 and min_samples_leaf: 5, it obtained a CV accuracy of 0.678 ± 0.015, compared with 0.670 ± 0.017 for Logistic Regression, 0.631 ± 0.011 for Random Forest and 0.549 ± 0.000 for Dummy. Its average train–validation gap was only 0.008, while Random Forest had a much larger gap of 0.171, showing clear overfitting.

In Week 3, I preferred Logistic Regression because of its smaller gap and slightly better recall and F1 on the single test split. With cross-validation, the Decision Tree has the higher average accuracy, although Logistic Regression still has a smaller gap of 0.002 and their accuracy results are close. For now, the Decision Tree with median imputation and standard scaling stays in config.yaml, since it gave the best average accuracy in this comparison. This is my current choice for the settings tested, rather than proof that it will always beat Logistic Regression; the final test set was not evaluated again to make this decision.

#### have a nice weekend!


__________

**Week 5**:

This week, I added hyperparameter tuning with Optuna. I kept my cleaning.py and preprocess.py separate and moved the construction of the full Pipeline into build_pipeline() in src/model.py. The new src/tuning.py tries different parameter combinations, while main.py chooses between the normal CV and tuning depending on tuning.enabled in config.yaml.

To avoid leakage, each trial evaluates the whole Pipeline with 5 stratified inner folds, so imputation, scaling and encoding are learned only from each fold's training rows. I also added 5-fold nested CV: tuning happens inside each outer training set, and the chosen pipeline is then evaluated on the outer validation rows that were left out of the search. The classification and fairness reports use these nested out-of-fold predictions.

All three models below used accuracy, 15 trials per search, the same outer folds and seed 42 for CV and Optuna's TPE sampler. I first tried the Decision Tree with seed 67, which gave 0.672 ± 0.011, then repeated it with seed 42 to match the Week 4 folds. The comparison below uses seed 42 throughout; I did not choose a different seed for each model according to its best result.

### *Challenge 1: Tune Logistic Regression and compare it with Decision Tree*

I added a search space for Logistic Regression's regularisation parameter C, between 0.0001 and 100 on a log scale. I kept max_iter: 1000 and solver: lbfgs. The final search on all development rows selected **C = 0.01205**, which means stronger regularisation than the starting value of 1.

For Decision Tree, I tuned max_depth, min_samples_leaf and criterion, while keeping min_samples_split: 10 and random_state: 42. Its final search selected **max_depth: 4, min_samples_leaf: 1 and criterion: entropy**.

| Model | Nested CV accuracy (mean ± std) | Train–validation gap | Class 1 recall | Class 1 F1 |
|---|---|---|---|---|
| Tuned Logistic Regression | 0.673 ± 0.013 | +0.002 | 0.53 | 0.59 |
| Tuned Decision Tree | 0.679 ± 0.011 | +0.007 | 0.54 | 0.60 |

The tuned Decision Tree had the higher average accuracy and slightly better class 1 recall and F1. Logistic Regression still had the smaller train–validation gap. Compared with Week 4, LR increased from 0.670 to 0.673, while the tree increased from 0.678 to 0.679. These are small differences, so I cannot say tuning gave a clear improvement or that one model is definitely better.

### *Challenge 2: Tune Random Forest*

I searched n_estimators between 100 and 500, max_depth from [None, 3, 5, 8, 12, 20], min_samples_leaf between 1 and 100 on a log scale, and max_features from [sqrt, log2, None]. I used cv.n_jobs: -1 to run folds in parallel and kept the forest's own n_jobs: 1 to avoid both levels trying to use all the CPU cores.

The final search selected **272 trees, max_depth: 12, min_samples_leaf: 36 and max_features: sqrt**.

| Model | Nested CV accuracy (mean ± std) | Train–validation gap | Class 1 recall | Class 1 F1 |
|---|---|---|---|---|
| Tuned Random Forest | 0.677 ± 0.011 | +0.013 | 0.55 | 0.61 |

The forest's selected maximum depth of 12 is higher than the tuned tree's 4, but it also requires at least 36 observations per leaf. This limits how much the trees can split even with a larger depth allowed. In Week 4, Random Forest had an accuracy of 0.631 and a gap of 0.171. With tuning, its recorded accuracy increased to 0.677 and its gap fell to 0.013, so the strong overfitting seen before was reduced substantially.

### *Best trial score vs nested CV*

| Model | Best final Optuna trial score | Nested CV mean | Best trial minus nested mean |
|---|---|---|---|
| Logistic Regression | 0.673 | 0.673 | +0.001 |
| Decision Tree | 0.674 | 0.679 | -0.005 |
| Random Forest | 0.682 | 0.677 | +0.005 |

The differences were calculated before rounding, which is why LR shows +0.001 even though both displayed scores are 0.673. The best trial score is used to choose parameters, so I used the nested CV score for the model comparison. The difference can also be negative, as it was for the tree: it is an optimism check, not a direct measurement of selection bias, because the training sets also differ in size.

### *Fairness and current best model*

The fairness comparison also uses nested OOF predictions. For African-American defendants, the FPR was 0.28 for LR and DT and 0.29 for RF, compared with 0.45 for COMPAS. For Caucasian defendants, it was 0.14 for LR and 0.16 for DT and RF, compared with 0.23 for COMPAS. The group differences are still present, so the tuning results do not mean the models are equally fair across groups.

**Decision Tree remains my leading model by recorded average accuracy**, with 0.679 compared with 0.677 for Random Forest and 0.673 for Logistic Regression. However, Random Forest had the highest class 1 recall and F1, while LR had the smallest training gap. Their results are close, and I have not established a statistically clear winner. The active configuration currently contains Random Forest because it was the last experiment I ran.

The nested scores evaluate the tuning procedure, which can select different parameters in each outer fold. After that evaluation, I ran a separate search on all 5,771 development rows and refitted each final winner on those rows. The 1,443 reserved test cases were not evaluated this week, although this test set had already been used in earlier weeks.

The reports are saved locally as results/optuna42.txt, results/optuna_lr42.txt and results/optuna_rf42.txt. The two main challenges are complete. The optional tasks of tuning preprocessing and repeating the experiments with balanced accuracy or F1 are still to do.

#### have a nice weekend!

__________

**Week 6**:

This week, I added AdaBoost and the notebook's projection-based oblique forest to my existing pipeline. The ensemble table below uses the original features and preprocessing, and compares the models on the same 5,771 development rows with 5 stratified outer folds, accuracy and seed 42. The 1,443 reserved test rows were not evaluated.

## Model evaluation

### Ensembles

The default column uses ordinary CV without tuning. I used the constructor defaults for model complexity, with seed 42 for reproducibility, one worker per forest because CV runs folds in parallel, and max_iter: 1000 for LR to allow convergence. The oblique baseline uses our notebook constructor defaults, including 100 trees and 10 projection components; it is a projection per tree, rather than a projection per node.

For the tuned column, every model used 50 trials per study, 5 inner folds and the same outer folds. I repeated LR, DT and RF with that budget and kept the existing 50-trial AdaBoost and oblique runs. The best-trial score and chosen parameters come from the separate final search on all development rows; the nested score evaluates the tuning procedure across the outer folds. I also included tuned LR because I had already implemented its C search in Week 5.

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | [0.670 ± 0.017 (+0.002)](results/week6_comparison/logistic_regression_default_cv_seed42.txt) | [0.670 ± 0.014 (+0.004)](results/week6_comparison/logistic_regression_tuned_50trials_seed42.txt) | 0.674 | `C=0.01533` |
| Decision tree | [0.627 ± 0.016 (+0.175)](results/week6_comparison/decision_tree_default_cv_seed42.txt) | [0.678 ± 0.011 (+0.008)](results/week6_comparison/decision_tree_tuned_50trials_seed42.txt) | 0.680 | `max_depth=5`, `min_samples_leaf=1`, `criterion=entropy` |
| Random forest | [0.631 ± 0.015 (+0.170)](results/week6_comparison/random_forest_default_cv_seed42.txt) | [0.676 ± 0.013 (+0.014)](results/week6_comparison/random_forest_tuned_50trials_seed42.txt) | 0.684 | `n_estimators=102`, `max_depth=None`, `min_samples_leaf=60`, `max_features=sqrt` |
| Oblique random forest | [0.637 ± 0.011 (+0.164)](results/week6_comparison/oblique_random_forest_default_cv_seed42.txt) | [0.681 ± 0.012 (+0.009)](results/week6_comparison/oblique_random_forest_tuned_50trials_seed42.txt) | 0.683 | `tree.max_depth=10`, `tree.min_samples_leaf=22`, `project.n_components=7` (100 trees) |
| AdaBoost | [0.668 ± 0.013 (+0.003)](results/week6_comparison/adaboost_default_cv_seed42.txt) | [0.680 ± 0.016 (+0.008)](results/week6_comparison/adaboost_tuned_50trials_seed42.txt) | 0.686 | `estimator.max_depth=6`, `n_estimators=301`, `learning_rate=0.01324` |
| Dummy (most frequent, reference) | [0.549 ± 0.000 (-0.000)](results/week6_comparison/dummy_default_cv_seed42.txt) | — | — | No tuning |

The gap is mean training score minus validation score, and std is the sample standard deviation across the five folds. The DT tuning still fixes min_samples_split: 10, as in Week 5; its default baseline uses the library default of 2. This week's default DT therefore differs from the restricted tree reported as my best model in Week 4. The new RF default also uses the library default of 100 trees, whereas the Week 4 forest used 300.

For this comparison, I would choose the tuned oblique forest as my current accuracy candidate, with a nested CV result of **0.681 ± 0.012**. Its lead over AdaBoost is only 0.001, smaller than the fold-to-fold standard deviations, so I have not shown a clear winner. AdaBoost selected depth-6 trees with a small learning rate of 0.01324, allowing more complex learners to contribute in small steps over 301 rounds. The ordinary forest selected no maximum depth but at least 60 observations per leaf, while the oblique forest selected depth 10 and at least 22 per leaf, so both still restrict tree growth. I would report the oblique forest's **nested CV accuracy of 0.681 ± 0.012**, rather than its best-trial score of 0.683, as the estimate for its tuning procedure, with final test performance still unmeasured this week.

I used the planned 50-trial runs for this table even where an earlier 5- or 15-trial run had a higher nested score. More trials improved the opportunity to find good inner-CV settings, but did not guarantee an improvement on outer validation rows.

### Reproducing and keeping the results

The full reports are in results/week6_comparison/, named by model, default or tuned mode, trial count and seed. The new runs also have JSON files with their exact configurations, and summary.csv / summary.json collect the values shown above. The existing AdaBoost and oblique reports were copied into this folder without deleting the originals; existing_run_sources.json records their original paths. The selected Week 6 reports are included in Git so the table links can be opened on GitHub; other generated results remain ignored.

I added a comparison runner that calls the same main.main() workflow with temporary configuration overrides, leaving config.yaml unchanged:

```bash
python run_week6_comparison.py
```

It skips reports already present in results/week6_comparison/. To repeat the experiments from scratch, keep the existing folder as an archive and use an empty results/week6_comparison/ folder. For a single model, set its model.type and params in config.yaml and run python main.py; tuning.enabled: false produces the default CV comparison and true runs nested tuning.

### Challenge 1: Feature engineering

I added add_engineered_features() to src/preprocess.py, keeping my separate cleaning and preprocessing structure. It runs after cleaning and duplicate removal, before separating features and target and making the reserved split. My ColumnTransformer already detects numeric columns automatically, so the three new numeric columns enter its scaling and imputation pipeline without adding a separate numeric_features list.

| Feature | Formula | Why I tried it |
|---|---|---|
| priors_per_adult_year (professor example) | `priors_count / (age - 17).clip(lower=1)` | Antecedents relative to approximate years of adult life. |
| juvenile_offenses_total (my feature) | `juv_fel_count + juv_misd_count + juv_other_count` | Let a tree split directly on the total juvenile count. |
| age_x_priors (my feature) | `age * priors_count` | Let the effect of priors depend on age, including in LR. |

None of these formulas learns a mean, threshold or any other statistic from the dataset, so they are safe before the split. Missing inputs stay missing; for the juvenile total I require all three counts, rather than treating an unknown count as zero. The old juvenile_total column is not reused: the new total comes from the cleaned source columns. Imputation, scaling and encoding are still fitted only inside each training fold.

I compared the three features together against the no-feature runs above for LR, RF and AdaBoost. Both conditions used 50 trials per search, the same search spaces, 5 inner folds, 5 outer folds, accuracy and seed 42. The existing no-feature runs were reused after checking their settings and the dataset fingerprint, and the new runs tuned again from scratch with the features enabled. Tests confirmed that the feature switch keeps exactly the same development and reserved-test row indices.

| Model | Without features: nested CV mean ± std (gap) | With features: nested CV mean ± std (gap) | Accuracy difference (with minus without) |
|---|---|---|---|
| Logistic regression | [0.670 ± 0.014 (+0.004)](results/feature_engineering/without_features/logistic_regression_tuned_50trials_seed42.txt) | [0.674 ± 0.013 (+0.003)](results/feature_engineering/with_features/logistic_regression_tuned_50trials_seed42.txt) | +0.004 |
| AdaBoost | [0.680 ± 0.016 (+0.008)](results/feature_engineering/without_features/adaboost_tuned_50trials_seed42.txt) | [0.678 ± 0.015 (+0.009)](results/feature_engineering/with_features/adaboost_tuned_50trials_seed42.txt) | -0.002 |
| Random forest | [0.676 ± 0.013 (+0.014)](results/feature_engineering/without_features/random_forest_tuned_50trials_seed42.txt) | [0.676 ± 0.013 (+0.016)](results/feature_engineering/with_features/random_forest_tuned_50trials_seed42.txt) | +0.000 |

The values and differences are displayed to three decimal places, so RF having the same displayed mean does not establish exact equality. The fold comparison CSV also uses the precision available in the saved reports.

The linear model had the only positive change in recorded mean accuracy, increasing by 0.004, while RF stayed at 0.676 at the displayed precision and AdaBoost decreased by 0.002. The LR gain is smaller than its fold standard deviations of 0.014 without features and 0.013 with them, so these results do not establish a clear improvement. Its class 1 recall also fell from 0.52 to 0.47 and F1 fell from 0.59 to 0.57, meaning that the higher accuracy came with a tradeoff in identifying positive cases. The ratio and interaction can add relationships that LR could not directly express before, while the juvenile sum is already a linear combination on fully observed rows and trees can already learn interactions through their splits. Since I tested all three features together, I cannot say which individual feature caused the changes, and I would keep the no-feature ensemble comparison as my main reference for now.

The feature switch is `preprocessing.engineered_features.enabled` in config.yaml, currently true for experimenting. To run the original features only, set it to false; run_week6_comparison.py explicitly disables it for the ensemble table above. To repeat this challenge comparison, run:

```bash
python run_feature_comparison.py
```

Reports and exact configurations are saved locally under results/feature_engineering/with_features/ and the baseline copies under without_features/. The folder also contains summary.csv, summary.json, fold_comparison.csv and method.json. Existing completed reports are skipped, and the original baseline files are preserved. The final tuned pipelines were refitted on development rows only; the reserved test set was not evaluated.

### Challenge 2: Boosting Logistic Regression

I also tried AdaBoost with LogisticRegression(max_iter=2000) as its base learner. I tuned the base learner's C from 0.0001 to 100 on a log scale, n_estimators from 25 to 400 and learning_rate from 0.01 to 1.0, using the same 50-trial budget, inner and outer folds, accuracy and seed 42. The engineered features stayed enabled for both boosted LR and its plain LR reference.

| Model | Nested CV accuracy (mean ± std) | Train–validation gap | Class 1 recall | Class 1 F1 |
|---|---|---|---|---|
| Plain LR, with features | 0.674 ± 0.013 | +0.003 | 0.47 | 0.57 |
| Boosted LR, with features | 0.676 ± 0.012 | +0.001 | 0.52 | 0.59 |

The final search selected C = 0.0001062, 46 boosting rounds and learning_rate = 0.77636. Its best-trial score was 0.680, while the nested score was 0.676. Boosting gave a small increase of 0.002 in average accuracy and improved positive-class recall compared with plain LR with features, but the accuracy lead is smaller than the fold standard deviations, so I cannot claim a clear improvement.

A weighted sum of raw linear scores is still linear: `sum(alpha_i * (w_i · x + b_i)) = W · x + B`. However, this scikit-learn AdaBoost implementation combines the learners' predicted classes through weighted votes, rather than adding their raw linear scores, so the resulting voting rule does not have to be a single linear boundary. The linearity statement applies to the inputs received by LR; with our ratio and interaction features, even a plain LR can already describe nonlinear relationships in the original inputs.

The [full boosted LR report](results/adaboost_logistic_regression/adaboost_lr_seed42_50trials_features.txt) includes the outer-fold choices, classification and fairness reports, and final parameters. The final pipeline was refitted on all development rows, and the reserved test set remained unevaluated. Both the main feature-engineering challenge and this optional experiment are now complete.

The active config is currently set to adaboost_logistic_regression with features enabled and 50 trials. Running `python main.py` repeats that experiment and saves its report under results/adaboost_logistic_regression/.

#### have a nice weekend!

__________

**Week 7**:

This week, I added GradientBoostingClassifier, HistGradientBoostingClassifier and GaussianNB to my model registry and their search spaces to config.yaml. I kept my separate cleaning.py and preprocess.py, including the three engineered features, standard scaling and median imputation. All comparisons used the same 5,771 development rows and five stratified outer folds with seed 42; the 1,443 reserved test rows were not evaluated.

### Several metrics and OOF probabilities

I now report accuracy, balanced accuracy, precision, recall, F1, F2, ROC-AUC and average precision for every model. The tuning objective uses only cv.selection_metric, while all eight metrics are evaluated on the untouched validation folds. The pipeline is refitted independently within each inner fold, as before. Classification reports and fairness reports use the final OOF probabilities at a threshold of 0.5, and each full report includes a confusion matrix with rows representing true labels and columns representing predictions.

### Selection metric

I chose **ROC-AUC** because it compares how well the models rank positive cases above negative cases without fixing one classification threshold. Accuracy at 0.5 could hide useful differences in those scores, so I used ROC-AUC for tuning and the scoreboard. I still report recall, F2, average precision and group false-positive rates because a higher AUC does not by itself show that the final decisions are fair or useful.

### Scoreboard

The source is [results/runs.csv](results/runs.csv), with one completed run per model below. Every tuned model used the same budget of **15 Optuna trials per study**, five inner folds and seed 42, including the final search on all development rows. I used a smaller budget than the Week 6 final runs to finish within the available time, and did not choose different budgets according to each model's result. Dummy has no search space and is untuned. Untuned means the starting parameters recorded in the run configuration, not a tuned model evaluated again on its selection folds.

| Model | Untuned ROC-AUC mean ± std (gap) | Tuned nested ROC-AUC mean ± std (gap) | Dummy floor | Best final tuning score |
|---|---|---|---|---|
| [Dummy (prior)](results/week7/dummy_20261024_221211_485737.txt) | 0.5000 ± 0.0000 (+0.0000) | — | 0.5000 | — |
| [Logistic regression](results/week7/logistic_regression_20261024_221223_089263.txt) | 0.7223 ± 0.0156 (+0.0030) | 0.7222 ± 0.0134 (+0.0026) | 0.5000 | 0.7232 |
| [Gaussian Naive Bayes](results/week7/naive_bayes_20261024_221236_258435.txt) | 0.6996 ± 0.0137 (+0.0010) | 0.7001 ± 0.0137 (+0.0008) | 0.5000 | 0.7001 |
| [Decision tree](results/week7/decision_tree_20261024_221246_769637.txt) | 0.6320 ± 0.0096 (+0.2709) | 0.7167 ± 0.0107 (+0.0167) | 0.5000 | 0.7160 |
| [Random forest](results/week7/random_forest_20261024_221443_766044.txt) | 0.6651 ± 0.0137 (+0.2297) | 0.7265 ± 0.0129 (+0.0156) | 0.5000 | 0.7277 |
| [Oblique random forest](results/week7/oblique_random_forest_20261024_221729_817984.txt) | 0.6697 ± 0.0122 (+0.2251) | 0.7257 ± 0.0127 (+0.0207) | 0.5000 | 0.7274 |
| [AdaBoost (trees)](results/week7/adaboost_20261024_221919_983122.txt) | 0.7183 ± 0.0168 (+0.0086) | 0.7257 ± 0.0113 (+0.0135) | 0.5000 | 0.7265 |
| [Gradient Boosting](results/week7/gradient_boosting_20261024_221331_064059.txt) | 0.7237 ± 0.0144 (+0.0359) | 0.7258 ± 0.0128 (+0.0117) | 0.5000 | 0.7276 |
| [HistGradientBoosting](results/week7/hist_gradient_boosting_20261024_221256_638212.txt) | 0.7063 ± 0.0189 (+0.1016) | 0.7244 ± 0.0140 (+0.0160) | 0.5000 | 0.7261 |
| [AdaBoost (Logistic Regression)](results/week7/adaboost_logistic_regression_20261024_222140_559867.txt) | 0.6965 ± 0.0183 (+0.0046) | 0.7177 ± 0.0112 (+0.0024) | 0.5000 | 0.7197 |

The gap is training ROC-AUC minus validation ROC-AUC, and std is the sample standard deviation across the five outer folds. The no-model floor is a DummyClassifier(strategy="prior") pipeline evaluated on exactly the same folds. Its constant within-fold score gives ROC-AUC 0.5000. Best final tuning scores are selection scores, not the numbers I use to rank the tuned procedures.

Random Forest has the highest recorded nested mean, **0.7265 ± 0.0129**, but its lead over Gradient Boosting is only about 0.0007, much smaller than their fold-to-fold standard deviations. Gradient Boosting, the oblique forest and tree-based AdaBoost are also close, so I have not established a clear winner. The unrestricted DT and forests had large baseline gaps, and tuning reduced those gaps substantially. Tuning did not improve every baseline: for example, LR remained around 0.722, and boosted LR finished below plain LR on this selection metric. If I report one result from this comparison, I would use Random Forest's **nested ROC-AUC**, alongside that uncertainty and the other metrics, rather than its best Optuna trial score.

These are new ROC-AUC-selected experiments, so the earlier Week 6 accuracy table is kept as history rather than treated as the same ranking. The final chosen parameters came from a separate search on all development rows after nested evaluation, and each final winner was refitted on development rows only.

### Saved evidence and reproduction

Each model link above opens its full report. results/week7/ also contains the exact configuration, default and nested per-fold CSVs, final Optuna trials, and OOF probabilities with their original row indices. The experiment plan, environment versions and validation checks are recorded there too. The ROC-AUC scorer can use decision scores, while the exported OOF files contain probabilities; probability saturation introduced only a tiny last-decimal AUC difference (below 0.000002) in one LR fold.

The CSV scoreboard appends runs instead of replacing older ones. It records the untuned result, nested result, standard deviations, gaps, no-model floor, chosen parameters and all eight metrics, and the printed board filters to matching data, preprocessing, folds and selection metric. I tested that a second run is appended without losing the first. The untuned Gradient Boosting integration check was stored separately under results/week7/smoke/, outside the final scoreboard.

For one model, select model.type and its params in config.yaml and run:

```bash
python main.py
```

Use tuning.enabled: false for an untuned-only check, and true for nested tuning when the model has a search space. To reproduce or resume the complete comparison with the fixed 15-trial budget:

```bash
python run_week7.py
```

The runner skips completed models whose reports are already present. Use five trials while testing implementation changes; the results in this table are the completed 15-trial runs. No final test performance is reported because that set remains reserved.

#### have a nice weekend!
