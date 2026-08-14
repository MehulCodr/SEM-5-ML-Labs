# Week 3: Preprocessing and Data Leakage on the Titanic Dataset

This project demonstrates how data leakage can enter a machine-learning workflow through incorrect preprocessing and target-based feature engineering. Using the Titanic dataset and logistic regression, the notebook compares deliberately flawed workflows with a leakage-free scikit-learn pipeline.

## Notebook

`week3_DataLeakage.ipynb`

## Objectives

- Explain how test-set information can leak into model training.
- Compare preprocessing before and after the train-test split.
- Build a pipeline for imputation, scaling, encoding, and classification.
- Demonstrate leakage caused by full-dataset imputation.
- Show severe target leakage through ticket-based survival rates.
- Evaluate each workflow using classification metrics and visualizations.

## Dataset

The notebook downloads the Titanic dataset from Kaggle using:

```python
kagglehub.dataset_download("yasserh/titanic-dataset")
```

It loads `Titanic-Dataset.csv`, which contains 891 passenger records. The target is `Survived`, and the model uses seven features:

1. `Pclass`
2. `Age`
3. `SibSp`
4. `Parch`
5. `Fare`
6. `Sex`
7. `Embarked`

High-cardinality columns such as `Name` and `Ticket` are excluded from the baseline model for simplicity. `Ticket` is used later only to construct an intentionally leaky feature.

The dataset is divided into 70% training data and 30% testing data using `random_state=42`.

## Workflow

### 1. Preprocessing setup

Numeric and categorical features are processed separately with a `ColumnTransformer`:

- Numeric values are filled with the median and standardized with `StandardScaler`.
- Categorical values are filled with the most frequent category and encoded with `OneHotEncoder(handle_unknown="ignore")`.
- Logistic regression is used for binary classification.

The notebook evaluates each model using accuracy, precision, recall, F1 score, and ROC AUC. It also plots a confusion matrix and ROC curve.

### 2. Flawed preprocessing before splitting

The first flawed workflow calls `fit_transform()` on the entire feature matrix before creating the training and testing partitions. The imputers, scaler, and encoder therefore learn from test rows, contaminating the held-out evaluation.

### 3. Correct leakage-free pipeline

The corrected workflow splits the raw data first. It then places the preprocessor and logistic regression model in a single scikit-learn `Pipeline` fitted only on the training partition.

The notebook shows that the mean passenger age differs between the complete dataset and the training partition:

| Statistic | Value |
| --- | ---: |
| Full-dataset age mean | 29.70 |
| Training-only age mean | 29.26 |

This difference illustrates that preprocessing on the complete dataset can expose information about the held-out feature distribution.

### 4. Full-data imputation leakage

The second flawed workflow fills missing numeric values with full-dataset medians and categorical values with full-dataset modes before splitting. Scaling and encoding happen later in a pipeline, but the imputation values have already been influenced by the test rows.

### 5. Ticket-based target leakage

The most severe flawed workflow creates `Ticket_Surv_Rate` by grouping the complete dataset by `Ticket` and calculating the mean `Survived` value before splitting. This feature directly incorporates target labels, including labels belonging to held-out passengers, and produces an unrealistically strong model.

## Results

| Workflow | Accuracy | Precision | Recall | F1 score | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preprocessing before split | 0.8097 | 0.7941 | 0.7297 | 0.7606 | 0.8800 |
| Correct leakage-free pipeline | 0.8097 | 0.7941 | 0.7297 | 0.7606 | 0.8798 |
| Full-data imputation | 0.8097 | 0.7941 | 0.7297 | 0.7606 | 0.8798 |
| Full-data ticket target encoding | 0.9888 | 1.0000 | 0.9730 | 0.9863 | 0.9999 |

The preprocessing and imputation leaks do not change most recorded metrics for this particular split. Leakage is still present because held-out data influences model development. The ticket-based target feature makes the problem more visible by increasing accuracy to 98.88% and ROC AUC to 0.9999.

## Requirements

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- kagglehub
- pandas
- scikit-learn
- matplotlib
- seaborn

Install the dependencies with:

```bash
python -m pip install jupyter kagglehub pandas scikit-learn matplotlib seaborn
```

## Running the Project

From this directory, start Jupyter:

```bash
jupyter notebook week3_DataLeakage.ipynb
```

Then run the notebook cells in order. An internet connection is required the first time `kagglehub` downloads the dataset.

## Key Observations

- Split raw observations before fitting any data-dependent preprocessing step.
- Fit imputers, scalers, and encoders only on training data by placing them inside a pipeline.
- Similar scores do not prove that a leaky workflow is valid; they only mean the leakage had little visible effect on those metrics for this split.
- Features derived from the complete target column can directly reveal test labels and produce misleadingly high performance.
- The corrected pipeline provides the trustworthy comparison because the test features remain unseen until prediction and evaluation.
