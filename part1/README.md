# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis

## Dataset description and justification

The base dataset is the public **Medical Cost Personal (insurance) dataset** (1,338 rows, 7 columns), available at
<https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv> and on Kaggle as "Medical Cost Personal Datasets". Each row is one insured person: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, and the numeric target `charges` (annual medical costs billed by the insurer). It satisfies the brief: 500+ rows, 5+ columns, a numeric target (`charges`) and several categorical columns (`sex`, `smoker`, `region`).

The published file is already clean, which would make a cleaning exercise trivial. `make_raw_data.py` therefore produces `raw_insurance.csv` — the "as delivered by the client" file — by injecting **seeded, reproducible** data-quality problems of the kind raw client files really contain: a positively skewed `annual_income` column with ~8% nulls, a `referral_code` column with ~31% nulls, ~5% nulls in `bmi`, ~6% nulls in `charges`, 40 duplicated rows, and a corrupted `age` column (19 entries like `"45 yrs"`, forcing pandas to infer `object`). Every injected problem is detected and repaired by the EDA script, and the whole pipeline is deterministic (seed 42).

## How to run

```bash
pip install pandas numpy matplotlib seaborn
python make_raw_data.py   # builds raw_insurance.csv from insurance.csv (seeded)
python part1_eda.py       # runs Tasks 1-10, saves 6 plots + cleaned_data.csv
```

Files produced: `plots/*.png` (6 images, committed) and `cleaned_data.csv` (used by Parts 2 and 3).

## Step-by-step findings

### Task 1 — Load and first look
`raw_insurance.csv` loads as **1,378 rows × 9 columns**. `.dtypes` shows `age` inferred as `object` (a corruption fixed in Task 4), `sex`/`smoker`/`region`/`referral_code` as `object`, and `bmi`/`charges`/`annual_income` as `float64`.

### Task 4 (run early) — Data type correction
The dtype fix runs before the null audit so that coercion-created gaps are counted with all the others. `age` contained 19 values with a `" yrs"` suffix; `pd.to_numeric(errors='coerce')` converts the column to `float64` and turns those 19 corrupt entries into NaN. `sex`, `smoker`, and `region` are highly repetitive strings and were converted to `category` dtype. Memory use (via `df.memory_usage(deep=True).sum()`) fell from **458,773 bytes to 135,492 bytes — a 70.5% reduction** — because category codes replace repeated Python string objects.

### Task 2 — Null value analysis
Null count and percentage per column (full table printed by the script):

| column | nulls | % |
|---|---|---|
| age | 19 | 1.38 |
| bmi | 72 | 5.22 |
| charges | 89 | 6.46 |
| annual_income | 111 | 8.06 |
| referral_code | 435 | **31.57** |

Only `referral_code` exceeds the 20% threshold; it is documented here and dropped in Task 10 (imputing a third of an operational ID string is meaningless and it carries no analytical signal). `age` and `bmi` (below 20%, low skew) were filled with their **medians** (39.0 and 30.38). **Why median rather than mean:** the median is robust to outliers and skew — `bmi` has a long right tail and `charges`/`annual_income` are strongly right-skewed, so their means are dragged upward by a minority of extreme values and would systematically overstate a "typical" person if used as fill values. The two most-skewed columns (`charges`, `annual_income`) are deliberately deferred to Task 9a, where mean and median are compared *before* imputation as required.

### Task 3 — Duplicate detection and removal
`df.duplicated().sum()` found **40 duplicates**; `df.drop_duplicates()` removed exactly **40 rows** (1,378 → 1,338). De-duplication *did* change null percentages — e.g. `bmi` dropped by 5.23 points (to 0, because its nulls had already been filled and the duplicate copies removed) and `annual_income` rose slightly by 0.09 points, since removed duplicates were not proportionally null. This is why the script re-prints the null table delta after de-duplication.

### Task 5 — Descriptive statistics and skewness
`df.describe()` is printed for all numeric columns. Skewness (`df[col].skew()`):
`annual_income` **+1.81**, `charges` +1.53, `children` +0.94, `bmi` +0.30, `age` +0.05.

The most-skewed column is **`annual_income`** (+1.81). **Positive skew** means the right tail is long: most incomes cluster between roughly $28k and $58k but a small number of very high earners (max $263k) stretch the distribution rightward, pulling the mean above the median. **Negative skew** would be the mirror image — a long left tail pulling the mean below the median. **Consequence for mean imputation:** with positive skew the mean ($46,835) sits well above the median ($40,975), so filling missing values with the mean would inject values that are "too high" for a typical person, biasing the centre of the distribution upward and distorting downstream models. Hence median imputation.

### Task 6 — Outlier detection with IQR
Two numeric columns analysed; outliers were **counted and documented, not dropped**:

* **`bmi`**: Q1 = 26.51, Q3 = 34.32, IQR = 7.81, bounds [14.80, 46.03] → **16 outliers (1.2%)**. These are clinically plausible extreme-obesity values (up to 53.1), not data errors. **Decision: retain.** They carry real signal about health risk.
* **`charges`**: Q1 = 4,710.60, Q3 = 16,249.10, IQR = 11,538.49, bounds [−12,597.14, 33,556.84] → **135 outliers (10.8%)**. Nearly all are smokers with high charges — a real, systematic subpopulation, not noise. **Decision for Part 2: retain them and use tree-based models (random forest / gradient boosting), which are robust to target outliers; if a linear model is fitted, a log-transform of `charges` will be used instead of capping.** Capping would erase exactly the high-cost cases an insurer most needs to predict.

### Task 7 — Visualizations (all saved to `plots/`)
1. **`line_charges_by_index.png`** — charges over row index. No trend by index (the data are not time-ordered); the plot shows a dense band under ~$15k with frequent spikes up to $60k+, previewing the heavy right tail.
2. **`bar_mean_charges_by_region.png`** — mean charges per region. The southeast has the highest mean (~$14.7k), the southwest the lowest (~$12.3k); regional differences are modest compared with smoker differences.
3. **`hist_most_skewed.png`** — histogram (20 bins) of `annual_income`. **Shape:** unimodal with a sharp peak around $30–45k and a long right tail stretching past $250k — the classic log-normal income shape and the reason for its +1.81 skew.
4. **`scatter_age_vs_charges.png`** — age vs charges. **Direction: positive; strength: moderate overall (Pearson r ≈ 0.27) but clearly structured**: the points form three upward-sloping bands (non-smokers at the bottom, smokers in the two upper bands). Within each band the relationship is strong and nearly linear; pooling the bands dilutes the overall correlation, which is precisely why the Spearman coefficient (0.51) is higher.
5. **`box_charges_by_smoker.png`** — charges by smoker. The smoker median (~$34k) is roughly **four times** the non-smoker median (~$7.4k), and the smoker box (IQR) is far wider, showing both a much higher centre and a much larger spread; non-smokers show many high outliers, smokers almost none because their whole distribution is shifted upward.

### Task 8 — Correlation heat map
`plots/heatmap_pearson.png` (with `annot=True`). The strongest absolute pair among the numeric columns is **`age` and `charges` (r = 0.27)**. **Causality caveat:** the correlation is consistent with a causal story (bodies accumulate health problems with age), but correlation alone cannot establish it. **Plausible alternative (third-variable) explanations:** (1) *chronic-condition prevalence* — age is a proxy for accumulated diagnoses (diabetes, hypertension), and it may be the conditions, not age itself, driving cost; (2) insurers also price plans differently by age band, so part of the association can be a *pricing-policy* artifact rather than a health effect. Also visible: `annual_income` correlates mildly with `age` (r = 0.19) — by construction income was tied to seniority — but barely with `charges` (r = 0.03), so income is a weak direct predictor.

### Task 9a — Imputation strategy comparison
For the two most-skewed columns, mean and median **before imputation**:

| column | mean | median | skew |
|---|---|---|---|
| `annual_income` | 46,835.04 | 40,974.69 | +1.81 |
| `charges` | 13,114.56 | 9,289.08 | +1.53 |

Both columns are **positively skewed**, so the mean is pulled upward by extreme high values (the mean sits 14% and 41% above the median respectively) and the **median is the more representative central tendency — it was chosen for imputation**. (Had either column been negatively skewed, the mean would instead be dragged *below* the typical value by extreme low values, and the median would again be preferable.) After `fillna()` with the medians, `isnull().sum()` confirms **zero remaining nulls** in every retained column.

### Task 9b — Spearman vs Pearson
Both matrices and the full |Spearman − Pearson| difference table are printed. The three pairs with the largest absolute difference:

| pair | Spearman | Pearson | \|diff\| | interpretation |
|---|---|---|---|---|
| `age` – `charges` | 0.508 | 0.273 | 0.235 | \|Spearman\| > \|Pearson\|: **monotonic but non-linear** — charges rise consistently with age, but the smoker/non-smoker cost bands and the skewed target break proportionality, so rank correlation captures the relationship far better. |
| `charges` – `annual_income` | 0.105 | 0.031 | 0.075 | \|Spearman\| > \|Pearson\|: a weak but consistent monotonic association that Pearson almost misses because both variables are heavily right-skewed; still too weak to matter much. |
| `bmi` – `charges` | 0.111 | 0.176 | 0.065 | \|Pearson\| ≥ \|Spearman\|: **approximately linear** — the association is driven by a subgroup (high-BMI smokers) whose extreme charge values inflate the linear coefficient; in rank terms the overall relationship is weaker. |

**Measure chosen for Part 2 feature selection: Spearman.** The target is heavily skewed and the strongest predictor–target relationships are monotonic rather than proportional (age–charges being the clearest case), so Pearson systematically understates the useful signal. Spearman is also insensitive to the outliers we deliberately retained.

### Task 9c — Grouped aggregation
`df.groupby('smoker')['charges'].agg(['mean','std','count'])`:

| smoker | mean | std | count |
|---|---|---|---|
| no | 8,446.88 | 5,761.71 | 1,064 |
| yes | 30,039.44 | 12,530.67 | 274 |

(a) Highest mean: **smokers**; highest standard deviation: **smokers**. (b) High within-group std **is a concern but a manageable one**: a std of $12.5k around a $30k mean means smoker status alone cannot pin down an individual smoker's cost — the feature must be combined with `age` and `bmi` (which drive the within-band slopes seen in the scatter plot) rather than used alone. (c) Mean ratio: **30,039 / 8,447 ≈ 3.56×**. A 3.6-fold difference in group means is very large relative to the within-group spread of the non-smoker group, so `smoker` clearly **carries strong predictive signal** and is expected to be the single most important feature in Part 2.

### Task 10 — Output
`cleaned_data.csv` (1,338 rows × 8 columns) is saved with `df.to_csv(..., index=False)`: all nulls resolved, duplicates removed, dtypes corrected, `referral_code` dropped. Parts 2 and 3 load this file.

## Repository contents

| file | purpose |
|---|---|
| `insurance.csv` | original public source dataset |
| `make_raw_data.py` | builds the seeded "raw client file" `raw_insurance.csv` |
| `raw_insurance.csv` | the raw input the EDA script cleans |
| `part1_eda.py` | all cleaning, EDA, and visualization code (Tasks 1-10) |
| `plots/` | six generated `.png` images (saved via `plt.savefig()`) |
| `cleaned_data.csv` | final cleaned dataset, consumed by Parts 2 and 3 |
