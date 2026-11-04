import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class GaussianNB:

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)

        n_classes = len(self.classes_)
        n_fetures = X.shape[1]

        self.means_ = np.zeros((n_classes, n_fetures))
        self.variances_ = np.zeros((n_classes, n_fetures))
        self.priors_ = np.zeros(n_classes)

        for idx, k in enumerate(self.classes_):
            Xk = X[y == k]

            self.means_[idx] = Xk.mean(axis=0)
            self.variances_[idx] = Xk.var(axis=0)
            self.priors_[idx] = Xk.shape[0] / X.shape[0]

        return self

    def _log_gaussina(self, X):
        num = -0.5 * (X[:,None,:] - self.means_) ** 2 / self.variances_
        log_prob = np.sum(
            num - 0.5 * np.log(2 * np.pi * self.variances_),
            axis=2
        )

        return log_prob

    def predict(self, X):
        X = np.array(X)

        log_likelihood = self._log_gaussina(X)
        log_prior = np.log(self.priors_)

        return self.classes_[np.argmax(log_likelihood + log_prior, axis=1)]




X, y = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))








