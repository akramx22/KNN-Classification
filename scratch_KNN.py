import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("heart.csv")

print("First rows of dataset:")
print(df.head())

df = df.drop_duplicates()

df.fillna(df.mean(numeric_only=True), inplace=True)

df = pd.get_dummies(df, drop_first=True)

print("\nDataset after encoding:")
print(df.head())

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.values
y_test = y_test.values


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))



class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):

        predictions = []

        for x in X:
            prediction = self._predict(x)
            predictions.append(prediction)

        return np.array(predictions)

    def _predict(self, x):

        distances = []

        for x_train in self.X_train:
            distance = euclidean_distance(x, x_train)
            distances.append(distance)

        k_indices = np.argsort(distances)[:self.k]

        k_labels = [self.y_train[i] for i in k_indices]

        most_common = max(set(k_labels), key=k_labels.count)

        return most_common




knn = KNN(k=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)


print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))