import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def calculate_gradient(X, y, theta):
    m = y.size
    return (1 / m) * (X.T @ (sigmoid(X @ theta) - y))


def gradient_descent(X, y, num_iter = 100, L = 0.1, tol = 1e-7):

    X_b = np.concatenate((np.ones((X.shape[0],1)), X),1)
    theta = np.zeros((X_b.shape[1],1))

    for i in range(num_iter):
        gradient = calculate_gradient(X_b, y, theta)
        theta = theta - L * gradient

        if np.linalg.norm(gradient) < tol:
            break

    return theta


def predict_proba(X, theta):
    X_b = np.concatenate((np.ones((X.shape[0],1)), X),1)
    return sigmoid(X_b @ theta)


def predict(X, theta, threshold = 0.5):

    pred = predict_proba(X, theta)
    return (pred >= threshold).astype(int)


data = pd.read_csv('Social_Network_Ads.csv')

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

y = y.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

theta = gradient_descent(X_train, y_train)

y_pred = predict(X_test, theta)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))






