import numpy as np
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value



class DecisionTreeClassifier:
        def __init__(self, max_depth=5, min_samples_split=2):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.root = None


        def fit(self, X, y):
            self.root = self._build_tree(X, y)


        def _most_common_class(self, y):
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]

        def _build_tree(self, X, y, depth=0):
            samples, features = X.shape

            if (depth >= self.max_depth or samples < self.min_samples_split or len(np.unique(y)) == 1):
                leaf_value = self._most_common_class(y)
                return Node(value=leaf_value)

            best_feature, best_threshold = self._best_split(X, y)

            if best_feature is None:
                return Node(value=self._most_common_class(y))

            left_idx = X[:, best_feature] <= best_threshold
            right_idx = X[:, best_feature] > best_threshold

            left = self._build_tree(X[left_idx], y[left_idx], depth + 1)

            right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

            return Node(best_feature, best_threshold, left, right)

        def _entropy(self, y):
            classes, counts = np.unique(y, return_counts=True)

            probabilities = counts / len(y)

            entropy = 0

            for p in probabilities:
                entropy -= p * np.log2(p)

            return entropy

        def _compute_entropy(self, left, right):
            n = len(left) + len(right)

            return (
                    len(left) / n * self._entropy(left)
                    + len(right) / n * self._entropy(right)
            )

        def _best_split(self, X, y):
            best_feature = None
            best_threshold = None
            best_entropy = float('inf')

            samples, features = X.shape

            for feature in range(features):
                thresholds = np.unique(X[:, feature])

                for t in thresholds:
                    left = X[:, feature] <= t
                    right = X[:, feature] > t

                    if len(y[left]) == 0 or len(y[right]) == 0:
                        continue

                    entropy = self._compute_entropy(y[left], y[right])

                    if entropy < best_entropy:
                        best_feature = feature
                        best_threshold = t
                        best_entropy = entropy

            return best_feature, best_threshold


        def _traverse_tree(self, x, node):

            if node.feature is None:
                return node.value

            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)

            else:
                return self._traverse_tree(x, node.right)


        def predict(self, X):
            return np.array([
                self._traverse_tree(x, self.root)
                for x in X
            ])


X, y = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))