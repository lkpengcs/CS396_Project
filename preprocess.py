import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

X_train = pd.read_csv("X_train_raw.csv")

# Identify numeric/categorical
num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

preprocessor.fit(X_train)

joblib.dump(preprocessor, "preprocessor.pkl")
print("preprocessor.pkl saved.")
