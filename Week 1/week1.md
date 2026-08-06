# Week 1: K-Nearest Neighbors on the Iris Dataset

This project demonstrates multiclass classification with the K-Nearest Neighbors (KNN) algorithm using scikit-learn's Iris dataset. The notebook explores the data, trains KNN classifiers with and without feature scaling, compares several values of `k`, calculates class probabilities, and visualizes decision boundaries.

## Notebook

`MLLabAssignment1_MehulGupta_24CS282.ipynb`

## Objectives

- Load and inspect the Iris dataset.
- Visualize feature relationships and class distribution.
- Split the dataset into training and testing sets.
- Train KNN classifiers for different neighbor counts.
- Compare performance before and after standardization.
- Examine predicted class probabilities.
- Plot decision boundaries for uniform and distance-based voting.

## Dataset

The Iris dataset contains 150 samples divided equally among three flower species:

- Iris setosa (`0`)
- Iris versicolor (`1`)
- Iris virginica (`2`)

Each sample has four measurements in centimeters:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

The dataset has no missing values. It is divided into 70% training data and 30% testing data using `random_state=0`, producing 105 training samples and 45 testing samples.

## Workflow

### 1. Data exploration

The notebook loads the built-in dataset with `load_iris()`, prints its description, converts the feature matrix to a pandas DataFrame, and creates:

- A Seaborn pair plot for feature relationships
- A distribution plot for the target classes

### 2. KNN without scaling

Models are trained using `KNeighborsClassifier` with `k` values of 1, 3, 5, and 7.

| Neighbors | Training accuracy | Testing accuracy |
| ---: | ---: | ---: |
| 1 | 1.0000 | 0.9778 |
| 3 | 0.9619 | 0.9778 |
| 5 | 0.9714 | 0.9778 |
| 7 | 0.9714 | 0.9778 |

### 3. KNN with scaling

`StandardScaler` is fitted only on the training data and then applied to both training and testing features. This avoids leaking information from the test set into preprocessing.

| Neighbors | Training accuracy | Testing accuracy |
| ---: | ---: | ---: |
| 1 | 1.0000 | 0.9333 |
| 3 | 0.9714 | 0.9778 |
| 5 | 0.9714 | 0.9778 |
| 7 | 0.9714 | 0.9778 |

For this split, scaling reduces the test accuracy of the 1-neighbor model, while the models using 3, 5, and 7 neighbors retain a test accuracy of approximately 97.78%.

### 4. Class probabilities and decision boundaries

The notebook uses `predict_proba()` to show the proportion of neighboring samples belonging to each class. Because probabilities are generated after each training loop, the displayed probabilities come from the final model, where `k=7`.

Decision boundaries are plotted using the first two features, sepal length and sepal width. For every tested value of `k`, the notebook compares:

- `weights="uniform"`: every neighbor contributes equally
- `weights="distance"`: closer neighbors have greater influence

The decision-boundary models use a scikit-learn `Pipeline` that standardizes the two selected features before classification.

## Requirements

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- scikit-learn
- pandas
- seaborn
- matplotlib

Install the dependencies with:

```bash
python -m pip install jupyter scikit-learn pandas seaborn matplotlib
```

## Running the Project

From this directory, start Jupyter:

```bash
jupyter notebook MLLabAssignment1_MehulGupta_24CS282.ipynb
```

Then run the notebook cells in order.

> **Execution-order note:** In the current notebook, the train-test split cell appears before the cell that imports `train_test_split`. In a fresh kernel, move or run `from sklearn.model_selection import train_test_split` before creating `X_train` and `X_test`.

## Key Observations

- All tested unscaled models achieve approximately 97.78% test accuracy.
- With scaling, `k=3`, `k=5`, and `k=7` achieve the same 97.78% test accuracy.
- The perfect training accuracy at `k=1` is expected because each training sample is its own closest neighbor, but this can indicate overfitting.
- Petal measurements are generally more informative for separating Iris species than sepal measurements, so the two-feature decision plots are illustrative rather than the best possible classifiers.