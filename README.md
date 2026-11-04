# Machine Learning Classifiers from Scratch

Five classification algorithms implemented directly in Python and NumPy to explore optimization, probabilistic modelling, distance-based learning and recursive tree construction. Scikit-learn is used for supporting utilities, not for the estimators themselves.

## Implementations

| Algorithm | Core method | Source | Detailed README |
| --- | --- | --- | --- |
| Logistic regression | Sigmoid and batch gradient descent | [Code](logistic_regression.py) | [README](logistic-regression/README.md) |
| k-nearest neighbours | Euclidean distance, heap and majority vote | [Code](KNearest_Classifier.py) | [README](docs/knn.md) |
| Gaussian Naive Bayes | Class priors and Gaussian log-likelihoods | [Code](gaussian_NB.py) | [README](docs/gaussian-naive-bayes.md) |
| Decision tree | Recursive splits minimizing weighted entropy | [Code](decision_tree.py) | [README](docs/decision-tree.md) |
| Linear soft-margin SVM | Lagrangian dual formulation, SMO and KKT checks | [Code](SVM.py) | [README](docs/svm.md) |



