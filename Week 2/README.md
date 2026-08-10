# Week 2 — Experiment 1: Dataset Inspection and Baseline Modelling

This experiment formulates Titanic survival prediction as binary classification, audits data quality, evaluates naive baselines, and compares them with a leakage-free logistic-regression pipeline.

## Files

- `Experiment_1_Baseline_Modelling.ipynb` — complete experiment with executed code and conclusions
- `Titanic-Dataset.csv` — input dataset

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
python -m pip install jupyter pandas matplotlib scikit-learn
```

## Recorded results

- The most-frequent baseline achieved **0.618 test accuracy**.
- The logistic-regression pipeline achieved **0.798 test accuracy**, an absolute improvement of **0.180**.
- Non-survival is the majority class, making it a meaningful but uninformative baseline.
- Missing values and high-cardinality text columns are the main data-quality concerns.
- Missing `Cabin` values are filled with the mode, and the two records missing `Embarked` are removed. The optional logistic comparison uses complete columns and excludes high-cardinality `Cabin`.
- The improvement over baseline shows that the selected passenger features contain useful predictive signal.
