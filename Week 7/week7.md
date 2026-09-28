# Experiment No. 07: Decision Trees, Pruning, and Overfitting

## Aim

To implement a Decision Tree Classifier on the Titanic dataset, study how tree depth affects model performance, identify overfitting, and apply cost-complexity pruning to obtain a suitable balance between accuracy and model complexity.

## Notebook

`week7_DecisionTrees.ipynb`

The notebook is fully executed. It contains data auditing, exploratory visualisations, leakage-safe preprocessing, unrestricted and shallow trees, Gini-versus-entropy comparison, a depth sweep from 1 to 14, a cross-validated cost-complexity pruning search, tree visualisations, feature importance, automated checks, conclusions, and viva answers.

## Objectives

- Understand recursive partitioning in a Decision Tree Classifier.
- Study Gini impurity and entropy as splitting criteria.
- Train an unrestricted tree and a shallow tree with `max_depth=3`.
- Measure training and held-out testing performance.
- Diagnose high variance through the train-test accuracy gap.
- Study the effect of `max_depth` from 1 through 14.
- Generate a cost-complexity pruning path.
- Select a suitable `ccp_alpha` without tuning on the test set.
- Compare accuracy, tree depth, node count, leaf count, and interpretability.
- Answer the supplied viva questions.

## Dataset

The supplied `Titanic-Dataset.csv` contains 891 passenger records and 12 columns. It has no exact duplicate rows. The target is `Survived`:

- 0 = did not survive
- 1 = survived

| Target class | Rows | Percentage |
| --- | ---: | ---: |
| Did not survive | 549 | 61.62% |
| Survived | 342 | 38.38% |

### Missing values

| Column | Missing rows | Missing percentage |
| --- | ---: | ---: |
| `Age` | 177 | 19.87% |
| `Cabin` | 687 | 77.10% |
| `Embarked` | 2 | 0.22% |
| All other columns | 0 | 0.00% |

The source CSV is never changed. Missing predictor values are handled inside the model pipeline, so the imputers learn only from training data in each fit.

### Feature treatment

Numeric predictors:

- `Age`
- `SibSp`
- `Parch`
- `Fare`

Categorical predictors:

- `Pclass`
- `Sex`
- `Embarked`

Excluded columns:

- `PassengerId`: exported row identifier
- `Name` and `Ticket`: high-cardinality identifiers/text
- `Cabin`: 77.10% missing and sparse as a raw string

Numeric features are median-imputed. Categorical features are most-frequent-imputed and one-hot encoded. Feature scaling is unnecessary because decision-tree splits depend on ordering rather than distances or coefficient magnitudes.

## Concepts

For class proportions $p_k$ in a node, Gini impurity is

$$
G = 1 - \sum_k p_k^2.
$$

Entropy is

$$
H = -\sum_k p_k \log_2(p_k).
$$

Both equal zero in a pure node. A tree chooses splits with a large weighted impurity decrease.

An unrestricted tree can create highly specific branches and fit noise. Limiting `max_depth` is pre-pruning. Cost-complexity post-pruning minimises

$$
R_\alpha(T)=R(T)+\alpha|T_{\text{leaves}}|,
$$

where `ccp_alpha` controls the penalty on additional leaves. Larger values generally create smaller trees.

## Method

1. Load and audit the local Titanic CSV.
2. Select interpretable predictors and separate `Survived` as the target.
3. Create a stratified 80:20 train/test split using `random_state=42`.
4. Fit imputers and one-hot encoding inside each model pipeline.
5. Train an unrestricted Gini tree.
6. Train a shallow Gini tree with `max_depth=3`.
7. Compare training accuracy, test metrics, and structural complexity.
8. Compare depth-3 trees using Gini and entropy.
9. Train separate trees for `max_depth` values 1 through 14.
10. Generate candidate `ccp_alpha` values from the training set's pruning path.
11. Evaluate 60 candidate alphas with five-fold stratified cross-validation on the training set.
12. Use the one-standard-error rule to favour a simpler candidate with validation performance close to the best mean.
13. Fit that pruned tree on all training rows and evaluate it once on the held-out test set.
14. Inspect the pruned tree and its impurity-based feature importances.
15. Run automated data, splitting, pruning, and result checks.

The split contains 712 training rows and 179 testing rows. The observed survival rates are 38.34% and 38.55%, respectively.

## Unrestricted vs. Shallow Tree

| Model | Train accuracy | Test accuracy | Train-test gap | Balanced accuracy | Precision | Recall | Specificity | F1-score | Depth | Nodes | Leaves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unrestricted tree | 0.9831 | 0.8156 | 0.1675 | 0.7987 | 0.7812 | 0.7246 | 0.8727 | 0.7519 | 23 | 303 | 152 |
| Shallow tree (`max_depth=3`) | 0.8329 | 0.7933 | 0.0396 | 0.7481 | 0.8636 | 0.5507 | 0.9455 | 0.6726 | 3 | 15 | 8 |

### Held-out confusion matrices

| Model | True negatives | False positives | False negatives | True positives |
| --- | ---: | ---: | ---: | ---: |
| Unrestricted tree | 96 | 14 | 19 | 50 |
| Shallow tree | 104 | 6 | 31 | 38 |
| Pruned tree | 101 | 9 | 31 | 38 |

The unrestricted tree almost memorises the training set and has a 16.75-percentage-point train-test gap. Although its test accuracy on this particular split is higher than the shallow tree's, the much larger gap and 303-node structure indicate higher variance and weaker interpretability. The depth-3 tree has a 3.96-point gap and only 15 nodes, but it misses more survivors at the default class decision.

## Gini vs. Entropy at Depth 3

| Criterion | Train accuracy | Test accuracy | Train-test gap | Balanced accuracy | Precision | Recall | Specificity | F1-score | Nodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gini | 0.8329 | 0.7933 | 0.0396 | 0.7481 | 0.8636 | 0.5507 | 0.9455 | 0.6726 | 15 |
| Entropy | 0.8258 | 0.8045 | 0.0214 | 0.7761 | 0.8036 | 0.6522 | 0.9000 | 0.7200 | 15 |

Entropy performs slightly better on this fixed test split and has a smaller gap. This is an empirical result for one dataset split, not a universal advantage of entropy over Gini.

## Effect of Maximum Depth

| `max_depth` | Train accuracy | Test accuracy | Train-test gap | Nodes | Leaves |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7893 | 0.7765 | 0.0128 | 3 | 2 |
| 2 | 0.8048 | 0.7598 | 0.0450 | 7 | 4 |
| 3 | 0.8329 | 0.7933 | 0.0396 | 15 | 8 |
| 4 | 0.8413 | 0.7877 | 0.0536 | 29 | 15 |
| 5 | 0.8652 | 0.7598 | 0.1054 | 45 | 23 |
| 6 | 0.8862 | 0.7989 | 0.0874 | 69 | 35 |
| 7 | 0.8933 | 0.7989 | 0.0944 | 97 | 49 |
| 8 | 0.9157 | 0.7989 | 0.1168 | 123 | 62 |
| 9 | 0.9298 | 0.8212 | 0.1085 | 151 | 76 |
| 10 | 0.9452 | 0.7933 | 0.1519 | 177 | 89 |
| 11 | 0.9537 | 0.7989 | 0.1548 | 187 | 94 |
| 12 | 0.9551 | 0.8156 | 0.1394 | 195 | 98 |
| 13 | 0.9579 | 0.8156 | 0.1422 | 201 | 101 |
| 14 | 0.9607 | 0.8212 | 0.1394 | 213 | 107 |

Training accuracy generally rises with depth, while test accuracy fluctuates and does not improve systematically. Depths 9 and 14 tie for the highest observed diagnostic test accuracy of 0.8212; the notebook reports depth 9 because it is the first maximum. This curve is diagnostic only and is not used to tune the final model.

## Cost-Complexity Pruning

The notebook evaluates 60 candidate `ccp_alpha` values ranging from 0 to 0.035556. Five-fold stratified cross-validation on the training set gives:

- Best mean validation accuracy: 0.8090
- One-standard-error cutoff: 0.8028
- Selected `ccp_alpha`: 0.009053

The one-standard-error rule chooses the largest alpha whose mean validation accuracy remains at or above the cutoff. This produces a smaller model while keeping validation performance statistically close to the best candidate. The test labels do not influence alpha selection.

### Final comparison

| Model | Train accuracy | Test accuracy | Train-test gap | Balanced accuracy | Precision | Recall | Specificity | F1-score | Depth | Nodes | Leaves |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unrestricted tree | 0.9831 | 0.8156 | 0.1675 | 0.7987 | 0.7812 | 0.7246 | 0.8727 | 0.7519 | 23 | 303 | 152 |
| Shallow tree (`max_depth=3`) | 0.8329 | 0.7933 | 0.0396 | 0.7481 | 0.8636 | 0.5507 | 0.9455 | 0.6726 | 3 | 15 | 8 |
| Pruned tree (CV-selected alpha) | 0.8315 | 0.7765 | 0.0549 | 0.7345 | 0.8085 | 0.5507 | 0.9182 | 0.6552 | 3 | 11 | 6 |

Pruning reduces the unrestricted tree from 303 nodes and 152 leaves to 11 nodes and 6 leaves. The result sacrifices 3.91 percentage points of held-out accuracy on this split in exchange for a 96.4% reduction in node count and a much smaller train-test gap. The shallow depth-3 tree provides a middle point: 15 nodes and slightly better held-out accuracy than the pruned tree.

## Pruned-Tree Feature Importance

| Transformed feature | Importance |
| --- | ---: |
| `Sex_female` | 0.6467 |
| `Pclass_3` | 0.1612 |
| `Age` | 0.0917 |
| `Pclass_1` | 0.0586 |
| `Embarked_S` | 0.0417 |

All other transformed features have zero importance in the selected pruned tree. These are impurity-based fitted importances, not causal effects. They can also favour continuous or high-cardinality features because those features offer more possible split points.

## Observations

- The target is moderately imbalanced, so balanced accuracy, class-specific recall, precision, specificity, and F1-score supplement ordinary accuracy.
- Leakage-safe pipelines keep imputation and encoding inside training and cross-validation.
- The unrestricted tree attains 98.31% training accuracy but has the largest train-test gap and by far the greatest complexity.
- Increasing depth beyond 3 raises training accuracy consistently, while testing accuracy oscillates. This is the expected high-variance pattern.
- The entropy depth-3 tree performs slightly better than the equivalent Gini tree on this split.
- Restricting depth and increasing `ccp_alpha` both reduce complexity, but they need not improve accuracy on every single test split.
- The CV-selected pruned tree is dramatically easier to inspect than the unrestricted tree.
- The selected pruned model has higher specificity than recall, so it identifies non-survivors more reliably than survivors at the default decision rule.
- Test data is used only for final evaluation and diagnostic reporting, not for choosing `ccp_alpha`.
- Tree structure and importance values are associations learned from this historical dataset, not causal statements.

## Conclusion

The experiment successfully demonstrates classification trees, overfitting, pre-pruning, and cost-complexity post-pruning. The unrestricted tree obtains the strongest held-out accuracy in the final three-model comparison, but its 16.75-point train-test gap, depth of 23, and 303 nodes reveal substantial variance and poor interpretability. The depth-3 tree reduces the gap to 3.96 points with only 15 nodes. Five-fold training-set cross-validation selects `ccp_alpha=0.009053`; the resulting 11-node tree further simplifies the model and limits the gap, although its held-out accuracy falls to 77.65% on this split.

The central result is the bias-variance trade-off: added depth improves training fit but does not guarantee better generalisation. A suitable configuration depends on the cost of errors, stability across validation samples, and the need for interpretability rather than on training accuracy alone.

## Viva Questions - Short Answers

1. **What is a Decision Tree?** A supervised model that predicts by following feature-based decisions from a root node through branches to a leaf.
2. **What is the difference between a classification tree and a regression tree?** A classification tree predicts classes or class probabilities; a regression tree predicts continuous numeric values.
3. **What is Gini impurity?** It is $1-\sum_k p_k^2$, a measure of class mixing in a node. It is zero when all samples belong to one class.
4. **What is entropy in a decision tree?** It is $-\sum_k p_k\log_2(p_k)$, a measure of uncertainty in the node's class distribution.
5. **Why does an unrestricted decision tree tend to overfit?** It can create branches for noise and individual training cases, making it highly sensitive to the training sample.
6. **What does `max_depth` control?** It limits the maximum number of edges from the root to a leaf and therefore constrains tree complexity.
7. **What happens if `max_depth` is too small?** The tree may underfit because it cannot represent important structure in the data.
8. **What happens if `max_depth` is too large?** Training accuracy usually rises, but variance and the train-test gap can increase as the tree captures noise.
9. **What is pruning?** Pruning removes or prevents weak branches to reduce model complexity and improve stability and interpretability.
10. **What is `ccp_alpha`?** It is scikit-learn's cost-complexity penalty. A larger value penalises extra leaves more strongly and usually produces a smaller tree.

## Requirements and Execution

- Python 3.x
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- Matplotlib
- seaborn
- scikit-learn

From the repository root, run:

```powershell
jupyter notebook "Week 7/week7_DecisionTrees.ipynb"
```

Alternatively, open the notebook in JupyterLab or VS Code and run all cells in order. The notebook uses only the supplied local CSV and requires no internet access.
