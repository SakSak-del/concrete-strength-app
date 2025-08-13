"""Train model for Concrete Strength prediction and save artifacts."""
import json, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

df = pd.read_csv("concrete.csv")
feature_cols = [c for c in df.columns if c != "strength"]
X = df[feature_cols]
y = df["strength"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

metrics = {
    "r2": float(r2_score(y_test, y_pred)),
    "rmse": float(sqrt(mean_squared_error(y_test, y_pred))),
    "n_rows": int(df.shape[0]),
    "n_features": int(len(feature_cols)),
    "features": feature_cols,
}

joblib.dump(model, "model.joblib")
with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("Saved model.joblib and metadata.json")
