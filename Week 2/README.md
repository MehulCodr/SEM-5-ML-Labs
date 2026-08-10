# Week 2 — Experiment 1: Dataset Inspection and Baseline Modelling

This experiment formulates Titanic survival prediction as binary classification, audits data quality, evaluates naive baselines, and compares them with a leakage-free logistic-regression pipeline.

## Files

- `Experiment_1_Baseline_Modelling.ipynb` — complete experiment with executed code and conclusions
- `Titanic-Dataset.csv` — input dataset

## Workflow

1. Define the task, features, target, success criterion, and expected naive performance.
2. Inspect the schema and descriptive statistics.
3. Check duplicates, missing values, constant columns, mixed types, and impossible values.
4. Create a reproducible stratified 70/30 train/test split.
5. Evaluate most-frequent and stratified dummy classifiers.
6. Compare the baseline with a training-only preprocessing and logistic-regression pipeline.
7. Interpret the result and decide whether ML is justified.

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

- The most-frequent baseline achieved **0.616 test accuracy**.
- The leakage-free logistic-regression pipeline achieved **0.799 test accuracy**, an absolute improvement of **0.183**.
- Non-survival is the majority class, making it a meaningful but uninformative baseline.
- Missing values and high-cardinality text columns are the main data-quality concerns.
- The improvement over baseline shows that the selected passenger features contain useful predictive signal.
