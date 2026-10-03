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

