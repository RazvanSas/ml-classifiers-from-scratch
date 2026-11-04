import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SVM:

    def __init__(self, C, tolerance, epsilon):
        self.C = C
        self.tolerance = tolerance
        self.epsilon = epsilon

        self.X = None
        self.y = None

        self.alphas = None
        self.b = 0.0


    def kernel(self, x1, x2):

        result = np.dot(x1, x2)

        return result


    def decision_function_single(self, x):
        result = 0.0

        number_of_samples = len(self.X)

        for k in range(number_of_samples):
            kernel_value = self.kernel(self.X[k], x)

            contribution = (self.alphas[k] * self.y[k] * kernel_value)

            result = result + contribution

        result = result + self.b

        return result


    def calculate_error(self, i):
        prediction = self.decision_function_single(self.X[i])

        error = prediction - self.y[i]

        return error


    def violates_kkt(self, i):

        error_i = self.calculate_error(i)

        r_i = self.y[i] * error_i

        alpha_i = self.alphas[i]

        if r_i < -self.tolerance and alpha_i < self.C:
            return True

        if r_i > self.tolerance and alpha_i > 0:
            return True

        return False


    def select_second_alpha(self, i):

        error_i = self.calculate_error(i)

        best_j = -1
        best_difference = -1.0

        number_of_samples = len(self.X)

        for j in range(number_of_samples):

            if j == i:
                continue

            error_j = self.calculate_error(j)

            difference = abs(error_i - error_j)

            if difference > best_difference:
                best_difference = difference
                best_j = j

        return best_j


    def calculate_bounds(self, i, j):

        alpha_i = self.alphas[i]
        alpha_j = self.alphas[j]

        y_i = self.y[i]
        y_j = self.y[j]

        if y_i != y_j:

            L = max(0.0, alpha_j - alpha_i)

            H = min(self.C, self.C + alpha_j - alpha_i)

        else:

            L = max(0.0, alpha_i + alpha_j - self.C)

            H = min(self.C, alpha_i + alpha_j)

        return L, H


    def calculate_eta(self, i, j):

        K_ii = self.kernel(self.X[i], self.X[i])
        K_jj = self.kernel(self.X[j], self.X[j])
        K_ij = self.kernel(self.X[i], self.X[j])

        eta = K_ii + K_jj - 2.0 * K_ij

        return eta


    def optimize_pair(self, i, j):

        alpha_i_old = self.alphas[i]
        alpha_j_old = self.alphas[j]

        error_i = self.calculate_error(i)
        error_j = self.calculate_error(j)

        L, H = self.calculate_bounds(i, j)

        if L == H:
            return False

        eta = self.calculate_eta(i, j)

        if eta <= 0:
            return False

        alpha_j_new = alpha_j_old + (self.y[j] * (error_i - error_j)) / eta

        if alpha_j_new < L:
            alpha_j_new = L

        elif alpha_j_new > H:
            alpha_j_new = H

        if abs(alpha_j_new - alpha_j_old) < self.epsilon:
            return False

        alpha_i_new = alpha_i_old + (self.y[i] * self.y[j] * (alpha_j_old - alpha_j_new))

        self.alphas[i] = alpha_i_new
        self.alphas[j] = alpha_j_new

        delta_alpha_i = alpha_i_new - alpha_i_old
        delta_alpha_j = alpha_j_new - alpha_j_old

        K_ii = self.kernel(self.X[i], self.X[i])
        K_jj = self.kernel(self.X[j], self.X[j])
        K_ij = self.kernel(self.X[i], self.X[j])

        b_i = (self.b - error_i - self.y[i] * delta_alpha_i * K_ii - self.y[j] * delta_alpha_j * K_ij)
        b_j = (self.b - error_j - self.y[i] * delta_alpha_i * K_ij - self.y[j] * delta_alpha_j * K_jj)

        if 0.0 < alpha_i_new and alpha_i_new < self.C:

            self.b = b_i

        elif 0.0 < alpha_j_new and alpha_j_new < self.C:

            self.b = b_j

        else:

            self.b = (b_i + b_j) / 2.0

        return True


    def fit(self, X, y, max_passes=10):

        self.X = np.array(X, dtype=float)
        self.y = np.array(y, dtype=float)



        number_of_samples = len(self.X)

        self.alphas = np.zeros(number_of_samples)
        self.b = 0.0

        passes_without_change = 0

        while passes_without_change < max_passes:

            successful_pair_updates = 0

            for i in range(number_of_samples):

                if not self.violates_kkt(i):
                    continue

                j = self.select_second_alpha(i)

                if j == -1:
                    continue

                changed = self.optimize_pair(i, j)

                if changed:
                    successful_pair_updates += 1

            if successful_pair_updates == 0:
                passes_without_change += 1

            else:
                passes_without_change = 0


    def predict_single(self, x):

        score = self.decision_function_single(x)

        if score >= 0:
            return 1

        return -1


    def predict(self, X):

        X = np.array(X, dtype=float)

        predictions = []

        for i in range(len(X)):
            prediction = self.predict_single(X[i])

            predictions.append(prediction)

        return np.array(predictions)


    def calculate_w(self):

        number_of_features = self.X.shape[1]

        w = np.zeros(number_of_features)

        for i in range(len(self.X)):
            contribution = (
                    self.alphas[i]
                    * self.y[i]
                    * self.X[i]
            )

            w = w + contribution

        return w


X_train = np.array([
    [1.0, 2.0],
    [2.0, 2.0],
    [2.0, 3.0],
    [3.0, 3.0],

    [2.5, 2.5],
    [5.0, 5.0],
    [5.0, 6.0],
    [6.0, 5.0],
    [6.0, 6.0]
])

y_train = np.array([
    -1,
    -1,
    -1,
    -1,

     1,
     1,
     1,
     1,
     1
])


"""df = pd.read_csv('Social_Network_Ads.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

y = np.array(y, dtype=float, copy=True)

for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.75, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)"""



svm = SVM(
    C=1.0,
    tolerance=0.001,
    epsilon=0.00001
)

svm.fit(X_train, y_train)

print("Alphas:")
print(svm.alphas)

print("\nb:")
print(svm.b)

w = svm.calculate_w()
print("w =", w)
print("||w|| =", np.linalg.norm(w))
print("margin =", 2.0 / np.linalg.norm(w))

print("\nPredictions:")
print(svm.predict(X_train))

print("\nTrue labels:")
print(y_train)


w = svm.calculate_w()
b = svm.b


# -------------------------
# Plot points
# -------------------------

for i in range(len(X_train)):

    if y_train[i] == 1:

        plt.scatter(
            X_train[i][0],
            X_train[i][1],
            color = 'red'
        )

    else:

        plt.scatter(
            X_train[i][0],
            X_train[i][1],
            color = 'blue'
        )


# -------------------------
# Create x1 coordinates
# -------------------------

x1_values = np.linspace(
    np.min(X_train[:, 0]) - 1,
    np.max(X_train[:, 0]) + 1,
    100
)


decision_boundary = []
positive_margin = []
negative_margin = []


for x1 in x1_values:

    # w1*x1 + w2*x2 + b = 0

    decision_x2 = (
        -w[0] * x1 - b
    ) / w[1]


    # w1*x1 + w2*x2 + b = +1

    positive_x2 = (
        1 - w[0] * x1 - b
    ) / w[1]


    # w1*x1 + w2*x2 + b = -1

    negative_x2 = (
        -1 - w[0] * x1 - b
    ) / w[1]


    decision_boundary.append(decision_x2)

    positive_margin.append(positive_x2)

    negative_margin.append(negative_x2)


# -------------------------
# Plot hyperplanes
# -------------------------

plt.plot(
    x1_values,
    decision_boundary,
    label="Decision boundary"
)

plt.plot(
    x1_values,
    positive_margin,
    linestyle="--",
    label="+1 margin"
)

plt.plot(
    x1_values,
    negative_margin,
    linestyle="--",
    label="-1 margin"
)

for i in range(len(X_train)):

    if svm.alphas[i] > svm.epsilon:

        plt.scatter(
            X_train[i][0],
            X_train[i][1],
            s=200,
            facecolors="none",
            edgecolors="black"
        )


plt.xlabel("x1")
plt.ylabel("x2")

plt.legend()

plt.show()
