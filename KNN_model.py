import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

df = pd.read_csv("heart.csv")

#Data_Quality
print(df.head())
df.info()
print(df.describe())
print(df.duplicated().sum())

df = df.drop_duplicates()

print(df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

#Categorical_Values
df = pd.get_dummies(df, drop_first=True)

#Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

#Model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#Evaluation
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()