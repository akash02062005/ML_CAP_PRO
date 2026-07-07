# Part 2 — Supervised Machine Learning

## What this does

Predicts annual medical **`charges`** (regression) from the cleaned dataset produced in Part 1 (`cleaned_data.csv`, committed here for convenience — identical to `../part1/cleaned_data.csv`). The script one-hot encodes the categoricals, holds out 20% of the data, trains four models plus a baseline, evaluates them with MAE / RMSE / R², cross-validates the top candidates, and saves a feature-importance plot, a predicted-vs-actual plot, and a `model_results.csv` metrics table.

## How to run

```bash
pip install pandas numpy matplotlib scikit-learn
python part2_ml.py
```

The script looks for `cleaned_data.csv` locally, then falls back to `../part1/cleaned_data.csv`. Everything is seeded (`random_state=42`) and reproducible.

## Design decisions

**Target and features.** `charges` is the numeric target identified in Part 1. Features: `age`, `bmi`, `children`, `annual_income`, plus one-hot encodings of `sex`, `smoker`, `region` (`drop_first=True` to avoid redundant dummies). Feature-selection guidance came from Part 1's Spearman analysis, which flagged `age` (ρ = 0.51 with charges) and the grouped-aggregation result for `smoker` (3.56× mean ratio) as the strongest signals.

**Outlier handling (follow-through from Part 1, Task 6).** The 135 IQR outliers in `charges` were **retained**, exactly as documented in Part 1: they are real high-cost patients (almost all smokers), not errors, and are the cases an insurer most needs to predict. Two mitigations were used instead of capping: tree ensembles (Random Forest, Gradient Boosting), which are robust to target outliers, and a log-target variant of Linear Regression as the linear-model alternative to capping.

**Evaluation.** MAE (interpretable in dollars), RMSE (penalises large errors — important given the retained outliers), and R². A `DummyRegressor` mean-predictor baseline confirms the models add real value. 5-fold cross-validation guards against a lucky split.

## Results (held-out 20% test set, seed 42)

| model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean) | 8,924.31 | 11,822.80 | −0.00 |
| Linear Regression | 4,432.58 | 6,246.38 | 0.72 |
| Linear Regression (log target) | 4,040.65 | 7,496.20 | 0.60 |
| Gradient Boosting | 3,150.49 | 5,467.26 | 0.79 |
| **Random Forest** | **3,109.23** | **5,308.27** | **0.80** |

5-fold cross-validated R²: Linear 0.670 ± 0.037, Gradient Boosting 0.736 ± 0.037, **Random Forest 0.748 ± 0.038** — the test-set ranking holds under CV, so it is not an artifact of one split.

## Key findings

**Random Forest is the best model** (test R² = 0.80, MAE ≈ $3,109 — roughly a third of the baseline's error). The tree ensembles beat Linear Regression by ~7–8 R² points because the dominant structure is interactive and non-linear: smoker status shifts the whole cost curve, and its effect compounds with `bmi` and `age` — exactly the banded pattern seen in Part 1's scatter plot. The log-target linear model reduces MAE versus plain linear regression (better for typical patients) but worsens RMSE/R², because compressing the target de-emphasises the expensive tail — evidence that capping/compressing outliers would have hurt, validating the Part 1 retention decision.

**Feature importance (Gradient Boosting)** confirms Part 1's EDA: `smoker_yes` dominates with **0.63** of total importance, followed by `bmi` (0.18) and `age` (0.12). `annual_income` contributes little (0.05) — consistent with its near-zero correlation with charges found in Part 1 — and `sex`/`region` are negligible (< 0.01), matching the modest regional differences in the Part 1 bar chart. The predicted-vs-actual plot shows tight fit for non-smokers and moderate spread in the high-cost smoker band, where within-group variance (Part 1, Task 9c) limits what any model can do.

## Repository contents

| file | purpose |
|---|---|
| `part2_ml.py` | full training / evaluation code |
| `cleaned_data.csv` | input data (from Part 1) |
| `model_results.csv` | generated metrics table |
| `plots/feature_importance.png` | generated feature-importance chart |
| `plots/pred_vs_actual.png` | generated predicted-vs-actual scatter |
