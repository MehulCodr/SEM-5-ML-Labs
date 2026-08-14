# Week 2: SVM and Naive Bayes on the Titanic Dataset

This project demonstrates binary classification using Gaussian Naive Bayes and Support Vector Machine (SVM) algorithms on the Titanic dataset. The notebook explores the data, performs feature engineering and preprocessing, trains both classifiers, and compares their performance using accuracy scores and confusion matrices.

## Notebook

`week2_SVM_and_NaiveBayes.ipynb`

## Objectives

- Load and inspect the Titanic dataset.
- Handle missing values and drop irrelevant columns.
- Encode categorical features and scale numerical features.
- Split the dataset into training and testing sets.
- Train a Gaussian Naive Bayes classifier.
- Train a Support Vector Machine (SVM) with an RBF kernel.
- Compare the performance of both models using accuracy, confusion matrix, and classification report.

## Dataset

The Titanic dataset contains passenger information and whether they survived the shipwreck. The target variable is `Survived` (0 = No, 1 = Yes).

The features used for prediction after dropping non-informative columns (`PassengerId`, `Name`, `Ticket`, `Cabin`) include:

- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Sex`: Sex
- `Age`: Age in years
- `SibSp`: Number of siblings/spouses aboard the Titanic
- `Parch`: Number of parents/children aboard the Titanic
- `Fare`: Passenger fare
- `Embarked`: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

The dataset is divided into 70% training data and 30% testing data using `random_state=67`.

## Workflow

### 1. Data Loading and Exploration

The notebook downloads the `yasserh/titanic-dataset` from Kaggle using `kagglehub`, loads it into a pandas DataFrame, and inspects its structure and checks for missing values.

### 2. Feature Engineering & Preprocessing

The preprocessing steps include:
- Dropping irrelevant columns that don't directly contribute to the prediction model: `PassengerId`, `Name`, `Ticket`, `Cabin`.
- Label encoding categorical variables: `Sex` and `Embarked`.
- Imputing missing values for the `Age` column.
- Scaling numerical features `Age` and `Fare` using `StandardScaler` (fitted only on the training data to prevent data leakage).

### 3. Naive Bayes Classification

A `GaussianNB` model is trained on the preprocessed training set and evaluated on the test set.
- **Testing Accuracy**: ~79%

### 4. Support Vector Machine (SVM) Classification

An `SVC` model with an RBF kernel (`kernel='rbf'`, `C=1.0`, `gamma='scale'`) is trained on the same preprocessed data and evaluated.
- **Testing Accuracy**: ~83%

## Requirements

- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- scikit-learn
- pandas
- kagglehub

Install the dependencies with:

```bash
python -m pip install jupyter scikit-learn pandas kagglehub
```

## Running the Project

From this directory, start Jupyter:

```bash
jupyter notebook week2_SVM_and_NaiveBayes.ipynb
```

Then run the notebook cells in order.

## Key Observations

- The SVM model using an RBF kernel (approx. 83% accuracy) outperformed the Gaussian Naive Bayes model (approx. 79% accuracy) on this specific test split.
- Proper feature scaling (applied to `Age` and `Fare`) and handling of missing values are crucial steps for models like SVM which are sensitive to the magnitude of features.
- Label encoding was successfully applied to convert categorical features (`Sex`, `Embarked`) into a numerical format suitable for these algorithms.
