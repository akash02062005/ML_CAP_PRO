"""
make_raw_data.py
----------------
Creates `raw_insurance.csv` — the "as delivered by the client" file — from the
public Medical Cost Personal (insurance) dataset.

The original insurance.csv is a well-known public dataset (1,338 rows) from
https://github.com/stedy/Machine-Learning-with-R-datasets (also on Kaggle as
"Medical Cost Personal Datasets"). It is already clean, so this script injects
REPRODUCIBLE, SEEDED data-quality problems that mirror what raw client files
typically contain. Part 1's EDA script then detects and fixes every one of them.

Injected problems (seed = 42 throughout):
  1. `annual_income`  — new positively-skewed numeric column (log-normal,
                        loosely tied to age), with ~8% missing values.
  2. `referral_code`  — new string column with ~30% missing values
                        (exceeds the 20% null-rate threshold in Task 2).
  3. `bmi`            — ~5% of values blanked out (below 20% threshold).
  4. `charges`        — ~6% of values blanked out (below 20% threshold).
  5. Duplicates       — 40 randomly chosen rows are appended again.
  6. Wrong dtype      — ~1.5% of `age` entries carry a " yrs" suffix
                        (e.g. "45 yrs") so pandas infers `object` instead
                        of int; pd.to_numeric(errors='coerce') fixes it.

Run:  python make_raw_data.py
Requires: pandas, numpy  (and insurance.csv in the same folder)
"""

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

df = pd.read_csv("insurance.csv")
n = len(df)
print(f"Loaded clean source: {df.shape}")

# 1. annual_income: log-normal (strong positive skew), loosely tied to age
income = np.exp(rng.normal(10.4, 0.55, n)) * (1 + 0.012 * (df["age"] - 18))
df["annual_income"] = income.round(2)

# 2. referral_code: short string code, ~30% missing (> 20% null threshold)
codes = np.array([f"REF-{c}{i:03d}" for c, i in
                  zip(rng.choice(list("ABCDE"), n), rng.integers(0, 1000, n))])
df["referral_code"] = codes
mask_ref = rng.random(n) < 0.30
df.loc[mask_ref, "referral_code"] = np.nan

# 3-4. blank out ~8% of annual_income, ~5% of bmi, ~6% of charges
for col, rate in [("annual_income", 0.08), ("bmi", 0.05), ("charges", 0.06)]:
    mask = rng.random(n) < rate
    df.loc[mask, col] = np.nan
    print(f"  nulled {mask.sum():4d} values in {col}")

# 5. corrupt age dtype: a handful of entries carry a " yrs" suffix, which
#    forces pandas to infer `object` for the whole column
df["age"] = df["age"].astype(int).astype(str)
mask_sfx = rng.random(n) < 0.015
df.loc[mask_sfx, "age"] = df.loc[mask_sfx, "age"] + " yrs"
print(f"  corrupted {mask_sfx.sum()} `age` entries with ' yrs' suffix")

# 6. append 40 duplicate rows (after corruption, so copies match exactly)
dup_idx = rng.choice(df.index, size=40, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)
print(f"  appended 40 duplicate rows -> {df.shape}")

df.to_csv("raw_insurance.csv", index=False)
print(f"Wrote raw_insurance.csv: {df.shape}")
