import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import heapq
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Social_Network_Ads.csv')

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

red = df[df.iloc[:,-1] == 1].values
blue = df[df.iloc[:,-1] == 0].values

red = red[:,:-1]
blue = blue[:,:-1]

plt.scatter(red[:,0],red[:,1], color = 'red')
plt.scatter(blue[:,0],blue[:,1], color = 'blue')
plt.show()


def euclidian_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))



class KNearestNeighbors:

    def __init__(self, k = 3):
        self.k = k
        self.points = None
        self.categories = None

    def fit(self, points, categories):
        self.points = points
        self.categories = categories

    def predict(self, new_point):
        pq = []

        for point in range(0, len(self.points)):
            distance = euclidian_distance(self.points[point], new_point)
            d = (distance, self.categories[point])
            heapq.heappush(pq, d)

        distances = []
        for i in range(0, self.k):
            distances.append(heapq.heappop(pq)[1])

        counter = Counter(distances)
        result = counter.most_common(1)[0][0]
        return result


classifier = KNearestNeighbors(5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
scaller = StandardScaler()
X_train = scaller.fit_transform(X_train)
X_test = scaller.transform(X_test)

classifier.fit(X_train, y_train)
y_pred = np.array([classifier.predict(point) for point in X_test])
y_pred = y_pred.reshape(-1,1)

print(confusion_matrix(y_test, y_pred))

