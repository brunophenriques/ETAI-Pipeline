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

