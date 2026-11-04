# Machine Learning Classifiers from Scratch

Five classification algorithms implemented directly in Python and NumPy to explore optimization, probabilistic modelling, distance-based learning and recursive tree construction. Scikit-learn is used for supporting utilities, not for the estimators themselves.

## Implementations

| Algorithm | Core method | Source | Detailed README |
| --- | --- | --- | --- |
| Logistic regression | Sigmoid and batch gradient descent | [Code](logistic_regression.py) | [README](docs/logistic-regression.md) |
| k-nearest neighbours | Euclidean distance, heap and majority vote | [Code](KNearest_Classifier.py) | [README](docs/knn.md) |
| Gaussian Naive Bayes | Class priors and Gaussian log-likelihoods | [Code](gaussian_NB.py) | [README](docs/gaussian-naive-bayes.md) |
| Decision tree | Recursive splits minimizing weighted entropy | [Code](decision_tree.py) | [README](docs/decision-tree.md) |
| Linear soft-margin SVM | Lagrangian dual formulation, SMO and KKT checks | [Code](SVM.py) | [README](docs/svm.md) |

## Setup

Run commands from this repository's root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the self-contained examples

```bash
python gaussian_NB.py
python decision_tree.py
python SVM.py
```

Naive Bayes and the decision tree use scikit-learn's bundled breast cancer dataset. SVM uses a small embedded synthetic dataset and displays its separating hyperplane and margins.

## Run the CSV examples

Place `Social_Network_Ads.csv` in the repository root, then run:

```bash
python logistic_regression.py
python KNearest_Classifier.py
```

The CSV is not included. These scripts interpret every column except the last as a numeric feature, and the final column as the class label. Use 0/1 labels for the supplied demonstrations. Remove or encode text columns before running. The k-NN plot requires at least two feature columns. Training-fitted standardization is applied to both train and test features.

## Evaluation and limitations

Accuracy and confusion matrices are printed where implemented; the k-NN demo prints a confusion matrix, and the SVM demo visualizes its training predictions. No unverified performance scores are claimed.

These are educational implementations rather than a packaged estimator API. Demo code executes at import time. The decision-tree and Naive Bayes splits have no fixed random seed, dependencies are unpinned, and numerical robustness and input validation can be extended. See the individual READMEs for algorithm-specific limitations.

## Related regression projects

- [Linear regression](https://github.com/RazvanSas/linear_regression_fromscratch)
- [Polynomial regression](https://github.com/RazvanSas/polynomial_regression_fromscratch)
- [Decision tree regression](https://github.com/RazvanSas/decision_tree_regression_fromscratch)

The regression implementations remain in their existing repositories; they are not duplicated here.

