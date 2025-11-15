import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("adult.csv").replace("?", pd.NA).dropna()

X = df.drop("income", axis=1)
y = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

# stratify income + sex
strat = y.astype(str) + "_" + X["sex"].astype(str)

# First split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=strat
)

# Second split
strat_temp = strat.loc[X_temp.index]

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=strat_temp
)

# Save raw splits
X_train.to_csv("X_train_raw.csv", index=False)
X_val.to_csv("X_val_raw.csv", index=False)
X_test.to_csv("X_test_raw.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_val.to_csv("y_val.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("✔️ Raw splits saved.")
