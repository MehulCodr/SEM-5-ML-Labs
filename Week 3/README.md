# Week 3 — Experiment 2: Preprocessing Pipelines and Leakage Detection

This experiment demonstrates why fitting preprocessing before a train/test split leaks held-out information. It compares deliberately flawed workflows with a correct scikit-learn pipeline on the Titanic dataset.

## Files

- `Experiment_2_Preprocessing_Leakage.ipynb` — annotated leakage experiment and metric comparison
- `Titanic-Dataset.csv` — input dataset

## Leakage cases

1. Imputation, scaling, and one-hot encoding fitted on the full dataset before splitting.
2. Median/mode imputation values computed from the full dataset before splitting.
3. Category vocabulary learned from held-out rows.

The corrected workflow splits raw data first and fits preprocessing plus logistic regression inside one `Pipeline` using only training rows.

## Run

```powershell
jupyter notebook "Week 3/Experiment_2_Preprocessing_Leakage.ipynb"
```

Run all cells in order. The notebook works when launched from the repository root or from `Week 3`.

## Requirements

```powershell
python -m pip install jupyter pandas matplotlib scikit-learn
```

## Key lesson

Leakage is invalid even if it does not improve accuracy in a particular split. Every data-dependent operation—imputation, scaling, encoding, feature selection, or resampling—must be fitted using training data only. A pipeline makes that boundary explicit and repeatable.

## Recorded result

Both the fully leaky and corrected workflows produced **0.799 test accuracy** on this fixed split. The equality is useful evidence for the central lesson: leakage is defined by an invalid information boundary, not by whether a single observed score happens to increase.
