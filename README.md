# Machine Learning Classifiers from Scratch

Five classification algorithms implemented directly in Python and NumPy to explore optimization, probabilistic modelling, distance-based learning and recursive tree construction. Scikit-learn is used for supporting utilities, not for the estimators themselves.

## Implementations

| Algorithm | Core method | Source | Detailed README |
| --- | --- | --- | --- |
| Logistic regression | Sigmoid and batch gradient descent | [Code](logistic-regression/logistic_regression.py) | [README](logistic-regression/README.md) |
| k-nearest neighbours | Euclidean distance, heap and majority vote | [Code](knearest-classifier/KNearest_Classifier.py) | [README](knearest-classifier/README.md) |
| Gaussian Naive Bayes | Class priors and Gaussian log-likelihoods | [Code](naive-bayes/gaussian_NB.py) | [README](naive-bayes/README.md) |
| Decision tree | Recursive splits minimizing weighted entropy | [Code](decision-tree/decision_tree.py) | [README](decision-tree/README.md) |
| Linear soft-margin SVM | Lagrangian dual formulation, SMO and KKT checks | [Code](svm/SVM.py) | [README](svm/README.md) |



