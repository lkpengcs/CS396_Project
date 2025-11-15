import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

preprocessor = joblib.load("preprocessor.pkl")

X_train = pd.read_csv("X_train_raw.csv")
y_train = pd.read_csv("y_train.csv")

X_train = preprocessor.transform(X_train)

# Example baseline model pipeline
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

print("Train accuracy:", clf.score(X_train, y_train))

X_test = pd.read_csv("X_test_raw.csv")
y_test = pd.read_csv("y_test.csv")
X_test = preprocessor.transform(X_test)
print("Test accuracy:", clf.score(X_test, y_test))
