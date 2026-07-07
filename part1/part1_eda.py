"""
Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis
=============================================================
Runs top-to-bottom on `raw_insurance.csv` (create it first with
`python make_raw_data.py` if it is missing) and produces:

  * printed answers for Tasks 1-9 (nulls, duplicates, dtypes, skewness,
    IQR outliers, correlations, Spearman-vs-Pearson, grouped aggregation)
  * six .png plots in ./plots/  (five required types + correlation heat map)
  * cleaned_data.csv  (used again in Parts 2 and 3)

Run:  python part1_eda.py
Requires: pandas, numpy, matplotlib, seaborn
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless-safe; plt.savefig still works everywhere
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
os.makedirs("plots", exist_ok=True)

def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# ---------------------------------------------------------------- Task 1
banner("TASK 1 — Load and first look")
df = pd.read_csv("raw_insurance.csv")
print("First five rows:")
print(df.head())
print("\nColumn dtypes:")
print(df.dtypes)
print("\nShape:", df.shape)

# ---------------------------------------------------------------- Task 4 (part A)
# Fix the corrupted dtype FIRST so that the null analysis below counts every
# real gap. `age` arrives as object (stray whitespace around the digits).
banner("TASK 4 — Data type correction")
mem_before = df.memory_usage(deep=True).sum()
print(f"Memory before conversions: {mem_before:,} bytes")

print("\n`age` inferred dtype:", df["age"].dtype, "(should be numeric)")
df["age"] = pd.to_numeric(df["age"], errors="coerce")
print("`age` dtype after pd.to_numeric(errors='coerce'):", df["age"].dtype)
print("NaNs introduced by coercion:", int(df['age'].isnull().sum()))

# repetitive string columns -> category dtype
for col in ["sex", "smoker", "region"]:
    df[col] = df[col].astype("category")
print("\n`sex`, `smoker`, `region` converted to category dtype.")

mem_after = df.memory_usage(deep=True).sum()
print(f"Memory after conversions:  {mem_after:,} bytes "
      f"({100 * (mem_before - mem_after) / mem_before:.1f}% saved)")

# ---------------------------------------------------------------- Task 2
banner("TASK 2 — Null value analysis")
null_count = df.isnull().sum()
null_pct = (df.isnull().sum() / df.shape[0]) * 100
null_table = pd.DataFrame({"null_count": null_count,
                           "null_pct": null_pct.round(2)})
print(null_table)

over20 = null_pct[null_pct > 20].index.tolist()
print(f"\nColumns exceeding 20% null rate: {over20}")

# Median-fill numeric columns below the 20% threshold.
# NOTE: the two most-skewed numeric columns (charges, annual_income) are
# deliberately deferred to Task 9a, where mean vs median is compared BEFORE
# imputation, as the brief requires.
DEFERRED = ["charges", "annual_income"]
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    if 0 < null_pct[col] <= 20 and col not in DEFERRED:
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f"Filled `{col}` nulls with median = {med:.3f}")
print(f"Deferred to Task 9a (skewness-based choice): {DEFERRED}")

# ---------------------------------------------------------------- Task 3
banner("TASK 3 — Duplicate detection and removal")
n_dup = df.duplicated().sum()
rows_before = df.shape[0]
print("Duplicate rows found:", n_dup)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Rows removed: {rows_before - df.shape[0]}  (shape now {df.shape})")

null_pct_after = (df.isnull().sum() / df.shape[0]) * 100
delta = (null_pct_after - null_pct).round(3)
print("\nChange in null percentage after de-duplication (pct points):")
print(delta[delta != 0].to_string() if (delta != 0).any()
      else "  (no column's null percentage changed)")

# ---------------------------------------------------------------- Task 5
banner("TASK 5 — Descriptive statistics and skewness")
print(df.describe())

skews = {col: df[col].skew() for col in df.select_dtypes(include=np.number).columns}
skew_s = pd.Series(skews).sort_values(key=abs, ascending=False)
print("\nSkewness per numeric column (sorted by |skew|):")
print(skew_s.round(4).to_string())
most_skewed = skew_s.index[0]
second_skewed = skew_s.index[1]
print(f"\nHighest |skewness|: `{most_skewed}` (skew = {skew_s.iloc[0]:.4f})")

# ---------------------------------------------------------------- Task 6
banner("TASK 6 — Outlier detection with IQR (documented, NOT dropped)")
for col in ["bmi", "charges"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"{col:>8}: Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}  "
          f"bounds=[{lo:.2f}, {hi:.2f}]  outliers={n_out} "
          f"({100 * n_out / df[col].notna().sum():.1f}% of non-null rows)")

# ---------------------------------------------------------------- Task 9a
banner("TASK 9a — Imputation strategy: mean vs median for the two most-skewed")
for col in [most_skewed, second_skewed]:
    print(f"{col:>14}: mean = {df[col].mean():>12.2f} | "
          f"median = {df[col].median():>12.2f} | skew = {skews[col]:+.3f}")

print("\nBoth columns are positively skewed -> the mean is pulled upward by")
print("extreme high values, so the MEDIAN is used for imputation.")
for col in [most_skewed, second_skewed]:
    df[col] = df[col].fillna(df[col].median())

print("\nisnull().sum() after imputation (numeric columns):")
print(df.select_dtypes(include=np.number).isnull().sum().to_string())

# ---------------------------------------------------------------- Task 7
banner("TASK 7 — Visualizations (five types)")

# 7.1 line plot
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["charges"], linewidth=0.6, color="tab:blue")
plt.title("Medical charges across the dataset (by row index)")
plt.xlabel("Row index")
plt.ylabel("Charges (USD)")
plt.tight_layout()
plt.savefig("plots/line_charges_by_index.png", dpi=150)
plt.close()

# 7.2 bar chart: mean charges per region
plt.figure(figsize=(7, 4.5))
means = df.groupby("region", observed=True)["charges"].mean().sort_values()
plt.bar(means.index.astype(str), means.values, color="tab:orange")
plt.title("Mean medical charges by region")
plt.xlabel("Region")
plt.ylabel("Mean charges (USD)")
plt.tight_layout()
plt.savefig("plots/bar_mean_charges_by_region.png", dpi=150)
plt.close()

# 7.3 histogram of the most-skewed column
plt.figure(figsize=(7, 4.5))
sns.histplot(df[most_skewed], bins=20, kde=True, color="tab:green")
plt.title(f"Distribution of `{most_skewed}` (most skewed, "
          f"skew = {skews[most_skewed]:.2f})")
plt.xlabel(most_skewed)
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/hist_most_skewed.png", dpi=150)
plt.close()

# 7.4 scatter: age vs charges, colored by smoker
plt.figure(figsize=(7.5, 5))
sns.scatterplot(data=df, x="age", y="charges", hue="smoker",
                alpha=0.6, s=25)
plt.title("Age vs medical charges (colored by smoker status)")
plt.xlabel("Age (years)")
plt.ylabel("Charges (USD)")
plt.tight_layout()
plt.savefig("plots/scatter_age_vs_charges.png", dpi=150)
plt.close()

# 7.5 box plot: charges split by smoker
plt.figure(figsize=(6.5, 5))
sns.boxplot(data=df, x="smoker", y="charges", hue="smoker",
            palette="Set2", legend=False)
plt.title("Medical charges by smoker status")
plt.xlabel("Smoker")
plt.ylabel("Charges (USD)")
plt.tight_layout()
plt.savefig("plots/box_charges_by_smoker.png", dpi=150)
plt.close()
print("Saved 5 plots to ./plots/")

# ---------------------------------------------------------------- Task 8
banner("TASK 8 — Pearson correlation heat map")
num_df = df.select_dtypes(include=np.number)
pearson = num_df.corr()
print(pearson.round(4))

plt.figure(figsize=(7, 5.5))
sns.heatmap(pearson, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, square=True)
plt.title("Pearson correlation matrix (numeric columns)")
plt.tight_layout()
plt.savefig("plots/heatmap_pearson.png", dpi=150)
plt.close()

corr_pairs = pearson.where(~np.eye(len(pearson), dtype=bool)).abs().stack()
top_pair = corr_pairs.idxmax()
print(f"\nHighest |Pearson| pair: {top_pair} "
      f"(r = {pearson.loc[top_pair]:.4f})")

# ---------------------------------------------------------------- Task 9b
banner("TASK 9b — Spearman rank correlation vs Pearson")
spearman = num_df.corr(method="spearman")
print("Spearman matrix:")
print(spearman.round(4))
print("\nPearson matrix (again, for comparison):")
print(pearson.round(4))

diff = (spearman - pearson).abs()
pairs = []
cols = diff.columns
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        pairs.append((cols[i], cols[j], spearman.iloc[i, j],
                      pearson.iloc[i, j], diff.iloc[i, j]))
diff_table = (pd.DataFrame(pairs, columns=["col_a", "col_b", "spearman",
                                           "pearson", "abs_diff"])
              .sort_values("abs_diff", ascending=False)
              .reset_index(drop=True))
print("\n|Spearman - Pearson| for every pair (largest first):")
print(diff_table.round(4).to_string())
print("\nTop-3 pairs with the largest |Spearman - Pearson| difference:")
print(diff_table.head(3).round(4).to_string())

# ---------------------------------------------------------------- Task 9c
banner("TASK 9c — Grouped aggregation: charges by smoker")
agg = df.groupby("smoker", observed=True)["charges"].agg(["mean", "std", "count"])
print(agg.round(2))
ratio = agg["mean"].max() / agg["mean"].min()
print(f"\nHighest-mean group: {agg['mean'].idxmax()}  "
      f"| highest-std group: {agg['std'].idxmax()}")
print(f"Ratio of highest to lowest group mean: {ratio:.2f}x")

# ---------------------------------------------------------------- Task 10
banner("TASK 10 — Save cleaned dataset")
# referral_code exceeded the 20% null threshold and carries no analytical
# signal (a random operational code), so it is dropped rather than imputed.
if "referral_code" in df.columns:
    df = df.drop(columns=["referral_code"])
    print("Dropped `referral_code` (>20% nulls, no analytical value).")

df["age"] = df["age"].round().astype(int)
df.to_csv("cleaned_data.csv", index=False)
print(f"Saved cleaned_data.csv  {df.shape}")
print("\nFinal null check:")
print(df.isnull().sum().to_string())
print("\nDone. All tasks complete.")
