# Week 2 — Experiment No. 02: Dataset Inspection and Baseline Modelling

This experiment formulates Titanic survival prediction as binary classification, audits data quality, evaluates naive baselines, and compares them with a leakage-free logistic-regression pipeline.

## Aim

To formulate a machine-learning problem, inspect and assess dataset health, separate features and target variables, split the data into training and testing sets, and establish a dummy baseline for evaluating whether a machine-learning solution is useful.

## Objectives

1. Identify classification, regression, and clustering problems.
2. Understand dataset structure and characteristics.
3. Inspect missing values, duplicates, constant columns, mixed types, and invalid values.
4. Separate features (`X`) and target (`y`).
5. Create training and testing subsets.
6. Build a simple dummy baseline model.
7. Evaluate the baseline using an appropriate performance metric.
8. Compare most-frequent and stratified strategies.
9. Determine whether a real ML model is justified relative to the baseline.

## Files

- `Experiment_1_Baseline_Modelling.ipynb` — complete experiment with executed code and conclusions
- `Titanic-Dataset.csv` — input dataset

**Dataset source:** [Titanic Dataset on Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

## Pseudo-algorithm

1. Start.
2. Read and understand the problem statement.
3. Identify the problem as classification, regression, or clustering.
4. Predict approximate naive-model performance.
5. Load the dataset with pandas.
6. Display dimensions, information, statistical summary, and first rows.
7. Check duplicates, missing values, categorical columns, unique values, constant columns, mixed types, and invalid values.
8. Separate features `X` and target `y`.
9. Make a 70:30 training/testing split.
10. Create a dummy baseline.
11. Train it on the training data.
12. Generate test predictions.
13. Calculate baseline performance.
14. Change the baseline strategy and repeat evaluation.
15. Compare results with the initial prediction.
16. Determine whether a real ML model could improve meaningfully on the baseline.
17. Record observations and conclusion.
18. Stop.

## Workflow

1. Define the task, features, target, success criterion, and expected naive performance.
2. Inspect the schema and descriptive statistics.
3. Check duplicates, missing values, constant columns, mixed types, and impossible values.
4. Fill missing `Cabin` values with the mode and drop the two rows with missing `Embarked`.
5. Create a reproducible stratified 70/30 train/test split from the cleaned data.
6. Evaluate most-frequent and stratified dummy classifiers.
7. Compare the baseline with a logistic-regression pipeline using only complete columns.
8. Interpret the result and decide whether ML is justified.

## Run

```powershell
jupyter notebook "Week 2/Experiment_1_Baseline_Modelling.ipynb"
```

Run all cells in order. The notebook works when launched from the repository root or from `Week 2`.

## Requirements

```powershell
python -m pip install jupyter numpy pandas matplotlib seaborn scikit-learn
```

- Hardware: computer/laptop with at least 4 GB RAM
- Software: Python 3.x, Jupyter Notebook or Google Colab, Anaconda (optional)

## Recorded results

- The most-frequent baseline achieved **0.618 test accuracy**.
- The logistic-regression pipeline achieved **0.798 test accuracy**, an absolute improvement of **0.180**.
- Non-survival is the majority class, making it a meaningful but uninformative baseline.
- Missing values and high-cardinality text columns are the main data-quality concerns.
- Missing `Cabin` values are filled with the mode, and the two records missing `Embarked` are removed. The optional logistic comparison uses complete columns and excludes high-cardinality `Cabin`.
- The improvement over baseline shows that the selected passenger features contain useful predictive signal.

The stratified baseline achieved **0.513 test accuracy**. The cleaned dataset contains **889 rows**, split into **622 training rows** and **267 testing rows**.

## Observation and conclusion

The most-frequent classifier predicts the majority training class for every test sample, so its accuracy follows the class distribution and it detects no survivors. The stratified classifier samples labels according to training proportions, giving a different result. The optional logistic model's 0.180 improvement indicates useful predictive information in the selected features; therefore, applying ML is justified.

## Viva questions and answers

1. **Three major ML problem types?** Classification, regression, and clustering.
2. **Purpose of dataset inspection?** To understand structure, distributions, quality issues, and modelling risks.
3. **Why split train and test data?** To train on one subset and estimate generalization independently on unseen data.
4. **What is a baseline model?** A simple reference that establishes minimum expected performance.
5. **What does `DummyClassifier(strategy='most_frequent')` do?** It predicts the most common training label for every record.
6. **Why is a baseline important?** It shows whether model complexity produces meaningful improvement.
7. **Purpose of `random_state=42`?** It makes randomized operations reproducible.
8. **What is data leakage?** Training with information that would be unavailable at prediction time, including held-out test information.
9. **Why check missing values?** They can break estimators or cause biased, lossy, or leaky preprocessing.
10. **What does `df.describe(include='all')` do?** It summarizes numeric and categorical columns, including counts, unique values, common values, and numeric statistics.
