# Experiment No. 05: Polynomial Regression, Overfitting, Ridge and Lasso Regularisation

## Aim

To study polynomial regression and the effect of increasing model complexity on training and testing errors, identify overfitting, and apply Ridge (L2) and Lasso (L1) regularisation to control model complexity.

## Notebook

`week5_polynomial.ipynb`

The notebook is fully executed and contains the tables, plots, coefficient analysis, model selection, automated checks, conclusion, and viva answers for this experiment.

## Objectives

- Train polynomial regression models of degrees 1, 3, 5, 7, and 10.
- Compare training and testing mean squared error (MSE).
- Identify underfitting and overfitting from the generalisation gap.
- apply Ridge and Lasso regularisation to the degree-10 model.
- Examine Ridge coefficient shrinkage and Lasso sparsity.
- Test regularised intermediate-degree models.
- Select a final model using test performance and model simplicity.

## Dataset

The experiment uses the supplied `Housing.csv` file. It contains 545 observations and 13 columns. Following the lab instructions, one numerical predictor and one target are selected:

- Predictor: `area`
- Target: `price`

Both selected columns have zero missing values. Their Pearson correlation is 0.5360, indicating a moderate positive relationship. The data is split into 436 training rows and 109 testing rows using an 80:20 split with `random_state=42`.

## Method

1. Load and validate `Housing.csv`.
2. Select `area` as the predictor and `price` as the target.
3. Split the observations into training and testing sets.
4. Generate polynomial features for degrees 1, 3, 5, 7, and 10.
5. Standardize polynomial terms using statistics learned only from the training data.
6. Fit an unregularised linear regression model for each degree.
7. Calculate training MSE, testing MSE, training R-squared, and testing R-squared.
8. Track how overfitting develops at degrees 5, 7, and 10 as training error falls while test error rises relative to degree 3.
9. Apply Ridge and Lasso to degree 10 using alpha values 0.001, 0.01, 0.1, 1, 10, and 100.
10. Examine coefficient norms and the number of zero coefficients.
11. Repeat the regularisation search for intermediate degrees 3, 5, and 7.
12. Select the final model primarily by testing MSE, using lower degree and fewer active coefficients as tie-breakers.

Price is divided by 1,000,000 during model fitting so that the shared alpha grid is meaningful. All predictions are converted back to the original price unit before MSE is calculated.

## Results: Polynomial Degree

| Degree | Training MSE | Testing MSE | Training R-squared | Testing R-squared |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2,204,738,681,379.34 | 3,675,286,604,768.19 | 0.2850 | 0.2729 |
| 3 | 2,063,400,239,338.49 | 3,563,793,367,396.84 | 0.3308 | 0.2949 |
| 5 | 1,939,099,209,220.33 | 3,717,573,668,645.70 | 0.3711 | 0.2645 |
| 7 | 1,920,333,820,246.52 | 3,721,799,880,877.72 | 0.3772 | 0.2637 |
| 10 | 1,909,703,967,362.37 | 3,813,251,778,147.18 | 0.3806 | 0.2456 |

Training MSE decreases at every tested degree. Testing MSE improves from degree 1 to degree 3, then worsens at degrees 5, 7, and 10. This growing separation between training and testing performance shows that overfitting begins after degree 3. Degree 10 is the strongest overfit model, with a generalisation gap of 1,903,547,810,784.81.

## Results: Degree-10 Regularisation

The best alpha for each degree-10 regularisation method was selected by its testing MSE.

| Method | Best alpha | Training MSE | Testing MSE | Testing R-squared | Coefficient L2 norm | Zero coefficients |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lasso | 0.01 | 1,984,937,327,786.37 | 3,661,607,743,858.74 | 0.2756 | 2.2223 | 7 of 10 |
| Ridge | 10 | 1,990,361,094,542.95 | 3,675,903,806,206.87 | 0.2728 | 1.7271 | 0 of 10 |

The unregularised degree-10 coefficient norm was approximately 721,448.59. Ridge with alpha 10 reduced it to 1.7271, demonstrating strong coefficient shrinkage. Lasso with alpha 0.01 set 7 of the 10 polynomial coefficients exactly to zero. Both methods reduced the degree-10 testing MSE, although neither outperformed the degree-3 models.

## Results: Intermediate Degrees

| Degree | Method | Best alpha | Training MSE | Testing MSE | Testing R-squared | Non-zero coefficients |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | Lasso | 0.001 | 2,065,411,427,391.94 | 3,562,224,275,170.40 | 0.2952 | 3 |
| 3 | Ridge | 0.1 | 2,063,972,987,534.45 | 3,562,657,068,538.66 | 0.2952 | 3 |
| 5 | Lasso | 0.01 | 2,044,545,295,003.91 | 3,578,810,529,894.14 | 0.2920 | 3 |
| 5 | Ridge | 1 | 2,012,938,957,513.53 | 3,592,684,268,020.23 | 0.2892 | 5 |
| 7 | Ridge | 10 | 2,023,243,813,105.39 | 3,620,993,698,067.50 | 0.2836 | 7 |
| 7 | Lasso | 0.01 | 1,996,762,029,437.00 | 3,622,782,620,129.39 | 0.2833 | 3 |

## Final Model

The selected model is degree-3 Lasso regression with alpha 0.001. It achieved the lowest observed testing MSE of 3,562,224,275,170.40 and a testing R-squared of 0.2952. It is preferred to degree 10 because it has substantially lower complexity and better test performance.

The improvement over unregularised degree 3 is small, so the important practical conclusion is that an intermediate-degree model generalises better than the highly flexible degree-10 model on this dataset.

## Observations

- Degree 1 has the highest training MSE and may underfit some curvature in the area-price relationship.
- Degree 3 gives the best unregularised test result.
- Degrees 5 and 7 lower training error but produce worse test errors than degree 3, showing that overfitting begins after degree 3.
- Degree 10 has the lowest training MSE and highest testing MSE among the unregularised models, making it the clearest overfit model.
- Ridge shrinks every coefficient toward zero without producing exact zeros.
- Lasso produces sparse models by setting polynomial coefficients exactly to zero.
- Very large alpha values cause excessive shrinkage and underfitting.
- Regularisation improves the overfit degree-10 model, but an intermediate degree remains better.
- Model selection should emphasize held-out performance while also considering complexity.

## Conclusion

Polynomial degree controls the bias-variance trade-off. Increasing degree from 1 to 3 captured useful nonlinearity, but degrees 5, 7, and 10 fitted the training data more closely at the cost of poorer test performance. Ridge and Lasso successfully controlled degree-10 complexity through coefficient shrinkage, and Lasso also performed feature selection. Degree-3 Lasso with alpha 0.001 produced the best test MSE and was selected as the final model.

## Viva Questions - Short Answers

1. **What is polynomial regression?** It is linear regression applied to polynomial transformations such as $x$, $x^2$, and $x^3$, allowing curved relationships to be modeled.
2. **What happens when polynomial degree is increased?** Flexibility increases, training error usually decreases, and the risk of high variance and overfitting increases.
3. **What is the difference between Ridge and Lasso?** Ridge uses an L2 penalty and generally shrinks all coefficients. Lasso uses an L1 penalty and can set coefficients exactly to zero.
4. **What is the role of alpha?** Alpha controls the strength of regularisation. Larger alpha values create stronger shrinkage.
5. **Why is a degree-10 polynomial more likely to overfit?** It has many flexible terms that can fit noise and create extreme curves.
6. **What is coefficient shrinkage?** It is the reduction of coefficient magnitudes caused by a regularisation penalty.
7. **What is sparsity in Lasso regression?** Sparsity means some coefficients are exactly zero, removing their polynomial terms from the model.
8. **Why compare train and test MSE?** The comparison shows whether a model generalises to unseen data or only fits its training data.
9. **Why might an intermediate degree be preferred over degree 10?** It can capture useful curvature with less variance, better test performance, and easier interpretation.
10. **What is the bias-variance trade-off?** Simple models tend to have higher bias and lower variance, while complex models tend to have lower bias and higher variance. A suitable model balances the two.

## Requirements and Execution

- Python 3.x
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn

Run the notebook from the repository root or the `Week 5` directory:

```bash
jupyter notebook week5_polynomial.ipynb
```

Then run all cells in order. The notebook loads the supplied local dataset and does not require an internet connection.
