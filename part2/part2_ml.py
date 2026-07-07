"""
Part 2 — Supervised Machine Learning
====================================
Predicts annual medical `charges` (regression) from the cleaned dataset
produced in Part 1 (`cleaned_data.csv`).

Pipeline
--------
1. Load cleaned data, one-hot encode categoricals.
2. 80/20 train/test split (fixed seed).
3. Train four models: baseline (mean predictor), Linear Regression,
   Linear Regression on log(charges), Random Forest, Gradient Boosting.
4. Evaluate with MAE, RMSE, R^2 on the held-out test set + 5-fold CV.
5. Plot feature importances and predicted-vs-actual; save metrics table.

Outlier handling (decision documented in Part 1, Task 6): the 135 IQR
outliers in `charges` are RETAINED. Tree ensembles are robust to them, and
for the linear model we compare a log-target variant instead of capping.

Run:  python part2_ml.py
Requires: pandas, numpy, matplotlib, scikit-learn
(expects ../part1/cleaned_data.csv or a local copy of cleaned_data.csv)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

SEED = 42
os.makedirs("plots", exist_ok=True)

# ------------------------------------------------------------------ load
path = "cleaned_data.csv" if os.path.exists("cleaned_data.csv") \
    else os.path.join("..", "part1", "cleaned_data.csv")
df = pd.read_csv(path)
print(f"Loaded {path}: {df.shape}")

TARGET = "charges"
X_raw = df.drop(columns=[TARGET])
y = df[TARGET]

# one-hot encode categoricals (drop_first avoids redundant dummy columns)
X = pd.get_dummies(X_raw, columns=["sex", "smoker", "region"], drop_first=True)
print("Feature matrix after one-hot encoding:", X.shape)
print("Features:", list(X.columns))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ------------------------------------------------------------------ models
models = {
    "Baseline (mean)": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=SEED),
}

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"model": name, "MAE": mae, "RMSE": rmse, "R2": r2}

rows = []
fitted = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted[name] = model
    rows.append(evaluate(name, y_test, model.predict(X_test)))

# linear model with log-transformed target (outlier-robust alternative
# to capping, per the Part 1 decision)
log_lin = LinearRegression().fit(X_train, np.log1p(y_train))
pred_log = np.expm1(log_lin.predict(X_test))
rows.append(evaluate("Linear Regression (log target)", y_test, pred_log))

results = pd.DataFrame(rows).set_index("model").round(2)
print("\n=== Held-out test-set results (20%) ===")
print(results.to_string())
results.to_csv("model_results.csv")

# ------------------------------------------------------------------ 5-fold CV
print("\n=== 5-fold cross-validated R^2 (mean +/- std) ===")
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
for name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
    scores = cross_val_score(models[name], X, y, cv=cv, scoring="r2")
    print(f"{name:>20}: {scores.mean():.4f} +/- {scores.std():.4f}")

# ------------------------------------------------------------------ importances
best_name = results["R2"].idxmax()
print(f"\nBest model on test R^2: {best_name}")

gb = fitted["Gradient Boosting"]
imp = pd.Series(gb.feature_importances_, index=X.columns).sort_values()
print("\nGradient Boosting feature importances:")
print(imp.round(4).to_string())

plt.figure(figsize=(7, 4.5))
imp.plot.barh(color="tab:purple")
plt.title("Feature importance — Gradient Boosting")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150)
plt.close()

# predicted vs actual for the best tree model
best_for_plot = fitted["Gradient Boosting"]
pred = best_for_plot.predict(X_test)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, pred, alpha=0.5, s=20)
lims = [0, max(y_test.max(), pred.max()) * 1.05]
plt.plot(lims, lims, "r--", linewidth=1)
plt.title("Gradient Boosting — predicted vs actual charges (test set)")
plt.xlabel("Actual charges (USD)")
plt.ylabel("Predicted charges (USD)")
plt.tight_layout()
plt.savefig("plots/pred_vs_actual.png", dpi=150)
plt.close()
print("\nSaved plots/feature_importance.png and plots/pred_vs_actual.png")
print("Saved model_results.csv")
print("Done.")
