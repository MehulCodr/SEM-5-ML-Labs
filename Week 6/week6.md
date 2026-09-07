# Experiment No. 06: Logistic Regression for Heart Disease Classification

## Aim

To build and evaluate a leakage-safe logistic regression model for binary heart-disease classification, compare it with a naive baseline, interpret its coefficients, and study the trade-off between sensitivity and specificity at different probability thresholds.

## Notebook

`week6_LogisticRegression.ipynb`

The notebook is fully executed and contains data-quality checks, exploratory plots, a preprocessing pipeline, baseline comparison, classification metrics, confusion matrices, ROC and precision-recall curves, cross-validation, coefficient interpretation, threshold analysis, automated checks, a conclusion, and viva answers.

> This experiment is for machine-learning education only. The model is not clinically validated and must not be used for diagnosis or patient-care decisions.

## Objectives

- Inspect and validate the supplied heart-disease dataset.
- Remove the exported row-index column and audit exact duplicate records without changing the source CSV.
- Distinguish continuous measurements from integer-coded categorical variables.
- Handle missing values, encoding, and feature scaling inside a model pipeline.
- Use a reproducible stratified train/test split.
- Compare logistic regression with a majority-class dummy baseline.
- Measure accuracy, precision, sensitivity, specificity, F1-score, ROC-AUC, and average precision.
- Estimate training-set stability with five-fold stratified cross-validation.
- Interpret logistic-regression coefficients and odds ratios.
- Examine how the decision threshold changes the false-positive/false-negative balance.

## Dataset

The supplied `heart disease classification dataset.csv` initially contains 303 rows and 15 columns. Its first unnamed column is a saved row index rather than a predictor. After that column and one exact duplicate are removed from the analysis copy, 302 rows and 14 variables remain. The original CSV is not modified.

The binary target is mapped as follows:

- `no` = 0 (no heart disease)
- `yes` = 1 (heart disease)

The analysis data contains 164 positive rows (54.30%) and 138 negative rows (45.70%). This mild imbalance makes accuracy usable but insufficient by itself, so class-specific and ranking metrics are also reported.

### Missing values

| Feature | Missing rows |
| --- | ---: |
| `trestbps` | 4 |
| `chol` | 1 |
| `thalach` | 5 |
| All other analysis columns | 0 |

The missing measurements are median-imputed inside the pipeline. Consequently, replacement values are learned only from the relevant training data or cross-validation fold.

### Feature treatment

Continuous variables:

- `age` (years)
- `trestbps` (resting blood pressure, mm Hg)
- `chol` (serum cholesterol, mg/dL)
- `thalach` (maximum heart rate achieved, bpm)
- `oldpeak` (exercise-induced ST depression)

Categorical variables:

- `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, and `thal`

The integer codes in the second group represent categories rather than equal-interval measurements. They are therefore one-hot encoded instead of being treated as continuous numbers.

## Method

1. Load the CSV from either the repository root or the `Week 6` directory.
2. Drop the saved index from an analysis copy and remove one exact duplicate.
3. Convert `target` from `yes`/`no` to 1/0.
4. Separate continuous and categorical predictors.
5. Create an 80:20 stratified split with `random_state=42`.
6. Median-impute and standardize continuous predictors.
7. Most-frequent-impute and one-hot encode categorical predictors.
8. Fit preprocessing and logistic regression as one pipeline.
9. Fit a majority-class dummy classifier for comparison.
10. Evaluate both models on the untouched 61-row test set.
11. Run five-fold stratified cross-validation on the 241-row training set.
12. Inspect coefficients, odds ratios, and probability-threshold trade-offs.

The split preserves the positive-class proportion: 131 of 241 training rows (54.36%) and 33 of 61 testing rows (54.10%) are positive.

## Held-out Test Results

| Model | Accuracy | Precision | Sensitivity | Specificity | F1-score | ROC-AUC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority-class baseline | 0.5410 | 0.5410 | 1.0000 | 0.0000 | 0.7021 | 0.5000 | 0.5410 |
| Logistic regression | 0.8361 | 0.8286 | 0.8788 | 0.7857 | 0.8529 | 0.8853 | 0.9023 |

The dummy classifier labels every test row as positive. Its 54.10% accuracy simply equals the positive-class prevalence, and its zero specificity shows why accuracy alone can be misleading.

Logistic regression improves accuracy by 29.51 percentage points and obtains an ROC-AUC of 0.8853. Its average precision of 0.9023 is also substantially above the positive prevalence of 0.5410.

### Logistic-regression confusion matrix

| Actual class | Predicted no disease | Predicted disease |
| --- | ---: | ---: |
| No disease | 22 | 6 |
| Disease | 4 | 29 |

At the default threshold of 0.50, the model correctly classifies 51 of 61 test rows. It produces 6 false positives and 4 false negatives.

### Per-class results

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| No disease | 0.8462 | 0.7857 | 0.8148 | 28 |
| Disease | 0.8286 | 0.8788 | 0.8529 | 33 |

## Five-Fold Cross-Validation

Cross-validation is performed only on the training portion. Each validation fold receives transformations learned from its corresponding training folds.

| Metric | Mean | Standard deviation |
| --- | ---: | ---: |
| Accuracy | 0.8509 | 0.0568 |
| Precision | 0.8606 | 0.0623 |
| Recall | 0.8701 | 0.0586 |
| F1-score | 0.8641 | 0.0496 |
| ROC-AUC | 0.9088 | 0.0425 |
| Average precision | 0.9196 | 0.0295 |

The held-out values are consistent with these cross-validation estimates. However, the variation between folds is non-trivial because the dataset is small.

## Coefficient Interpretation

The largest fitted coefficients by absolute magnitude are shown below. Odds ratios are `exp(coefficient)`.

| Transformed feature | Coefficient | Odds ratio |
| --- | ---: | ---: |
| `cp_2` | 1.4737 | 4.3653 |
| `cp_3` | 1.3751 | 3.9555 |
| `ca_2` | -1.3196 | 0.2673 |
| `ca_1` | -1.2280 | 0.2929 |
| `sex_male` | -0.9670 | 0.3802 |
| `thal_2` | 0.9105 | 2.4856 |
| `thal_3` | -0.8069 | 0.4462 |
| `exang_1` | -0.7620 | 0.4667 |
| `cp_1` | 0.7374 | 2.0905 |
| `ca_3` | -0.6383 | 0.5282 |
| Standardized `chol` | -0.4996 | 0.6068 |
| Standardized `oldpeak` | -0.4939 | 0.6103 |

For a one-hot feature, the coefficient compares that category with the encoder's dropped reference category while other modeled features stay fixed. For a standardized continuous feature, it represents a one-standard-deviation increase. These are fitted associations, not causal or clinical effects; coded meanings and reference categories must be interpreted using the dataset's original codebook.

## Threshold Analysis

| Threshold | Accuracy | Precision | Sensitivity | Specificity | F1-score | Predicted-positive rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 | 0.7541 | 0.7143 | 0.9091 | 0.5714 | 0.8000 | 0.6885 |
| 0.40 | 0.8197 | 0.7895 | 0.9091 | 0.7143 | 0.8451 | 0.6230 |
| 0.50 | 0.8361 | 0.8286 | 0.8788 | 0.7857 | 0.8529 | 0.5738 |
| 0.60 | 0.8361 | 0.8710 | 0.8182 | 0.8571 | 0.8438 | 0.5082 |
| 0.70 | 0.7377 | 0.8400 | 0.6364 | 0.8571 | 0.7241 | 0.4098 |

Lowering the threshold from 0.50 to 0.40 increases sensitivity from 0.8788 to 0.9091 but lowers specificity from 0.7857 to 0.7143. Raising it to 0.60 increases specificity to 0.8571 while lowering sensitivity to 0.8182. The table illustrates the trade-off; it does not select a clinical operating point. A real threshold requires separate validation data, domain costs, and external clinical evaluation.

## Observations

- The data is only mildly imbalanced, but the dummy result demonstrates that class-specific metrics remain necessary.
- Leakage-safe preprocessing is important even though only ten measurement values are missing.
- Logistic regression substantially outperforms the majority-class baseline on every informative test metric.
- Sensitivity is higher than specificity at the default 0.50 threshold.
- ROC-AUC and average precision indicate useful ranking ability on this split.
- Cross-validation results are broadly consistent with the held-out results, though uncertainty remains because there are only 302 unique records.
- One-hot encoding prevents arbitrary category codes from being interpreted as continuous clinical measurements.
- Coefficients provide model transparency, but correlated features, category references, sample size, and regularization affect their magnitudes.
- Changing the threshold redistributes false positives and false negatives; it does not improve every metric simultaneously.
- External validation, calibration assessment, subgroup evaluation, and clinical review would be required before any real-world use.

## Conclusion

The completed pipeline demonstrates a sound logistic-regression workflow for binary classification. With a fixed stratified split, the model achieved 83.61% test accuracy, 87.88% sensitivity, 78.57% specificity, an F1-score of 0.8529, and an ROC-AUC of 0.8853. These values are markedly better than the majority-class baseline and agree reasonably with the five-fold training-set estimates.

The key methodological result is not a clinical claim: it is that correct feature typing, training-only preprocessing, baseline comparison, multiple evaluation metrics, and explicit threshold analysis produce a more trustworthy classification experiment than reporting accuracy alone.

## Viva Questions - Short Answers

1. **Why is logistic regression used for classification?** It models the probability of a class through a sigmoid transformation and converts the probability into a label using a threshold.
2. **What does the sigmoid function do?** It maps any real-valued linear score to a value between 0 and 1.
3. **What is log-odds?** It is $\log(p/(1-p))$. Logistic regression expresses log-odds as a linear combination of predictors.
4. **Why is a pipeline used?** It makes preprocessing reproducible and ensures imputation, encoding, and scaling are learned from training data only.
5. **Why are coded variables one-hot encoded?** Their numbers denote categories, not equally spaced measurements; one-hot encoding avoids imposing a false numeric relationship.
6. **What is sensitivity?** It is the proportion of actual positive cases found by the model: $TP/(TP+FN)$.
7. **What is specificity?** It is the proportion of actual negative cases correctly rejected: $TN/(TN+FP)$.
8. **What is precision?** It is the proportion of predicted positive cases that are actually positive: $TP/(TP+FP)$.
9. **What is the F1-score?** It is the harmonic mean of precision and recall.
10. **What does ROC-AUC measure?** It measures how well prediction scores rank positive cases above negative cases across thresholds.
11. **What does an odds ratio above 1 mean?** Holding modeled variables fixed, the feature value is associated with higher fitted odds of the positive class relative to its reference.
12. **Why use stratified splitting?** It preserves approximately the same class balance in training and testing data.
13. **Why compare with a dummy classifier?** It tests whether the model learns useful patterns beyond always predicting the majority class.
14. **Why does lowering the threshold increase sensitivity?** More rows are labeled positive, so fewer positive cases are missed, usually at the cost of more false positives.
15. **Can this model diagnose heart disease?** No. It is a classroom experiment on a small dataset without clinical or external validation.

## Requirements and Execution

- Python 3.x
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn

Run the notebook from the repository root or the `Week 6` directory:

```bash
jupyter notebook week6_LogisticRegression.ipynb
```

Then run all cells in order. The notebook loads the supplied local CSV and does not require internet access.
