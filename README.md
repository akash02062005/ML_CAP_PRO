<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Anthropic-LLM-6B4FBB?style=for-the-badge&logo=anthropic&logoColor=white" alt="Anthropic"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="MIT License"/>
</p>

<h1 align="center">🧠 Data-to-AI  Project</h1>

<p align="center">
  <strong>End-to-end machine learning pipeline — from raw data to production-grade LLM system</strong>
</p>

<p align="center">
  <em>Medical Insurance Cost Prediction & Intelligent Q&A System with 10 Production Guardrails</em>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
- [Tech Stack](#-tech-stack)
- [Parts Overview](#-parts-overview)
- [Quick Start](#-quick-start)
- [Detailed Part Descriptions](#-detailed-part-descriptions)
  - [Part 1 — Data Acquisition & EDA](#part-1--data-acquisition-cleaning--exploratory-analysis)
  - [Part 2 — Supervised Machine Learning](#part-2--supervised-machine-learning)
  - [Part 3 — LLM Integration](#part-3--llm-integration-with-structured-outputs)
  - [Part 4 — Production System](#part-4--llm-powered-intelligent-system-with-guardrails)
- [Key Results](#-key-results)
- [Project Structure](#-project-structure)
- [Environment Setup](#-environment-setup)
- [License](#-license)

---

## 🎯 Overview

This capstone project demonstrates the **complete data-to-AI lifecycle** across four independently deliverable parts. Starting from a raw medical insurance dataset with real-world data quality issues, the pipeline progresses through:

1. **Data cleaning & exploratory analysis** — handling nulls, duplicates, type corruptions, and generating actionable insights
2. **Supervised ML modeling** — training and evaluating multiple regression models to predict medical charges
3. **LLM integration** — extracting structured analytical insights via raw HTTP API calls with strict schema validation
4. **Production deployment** — building a guarded LLM Q&A system with 10 enterprise-grade guardrails

> **Dataset:** The [Medical Cost Personal Dataset](https://github.com/stedy/Machine-Learning-with-R-datasets) — 1,338 insured individuals with features like age, BMI, smoker status, region, and annual medical charges.

---

## 🏗 Project Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA-TO-AI PIPELINE                             │
├─────────────┬──────────────┬──────────────────┬────────────────────────┤
│   PART 1    │   PART 2     │     PART 3       │       PART 4           │
│  Data &     │  Machine     │  LLM Structured  │  Production LLM       │
│  EDA        │  Learning    │  Outputs         │  System               │
├─────────────┼──────────────┼──────────────────┼────────────────────────┤
│             │              │                  │                        │
│ ▪ Acquire   │ ▪ One-hot    │ ▪ Compute stats  │ ▪ 10 guardrails       │
│ ▪ Clean     │   encode     │ ▪ Raw HTTP POST  │ ▪ PII redaction       │
│ ▪ Impute    │ ▪ Train/test │ ▪ Strict JSON    │ ▪ Injection defense   │
│ ▪ Visualize │   split      │   schema         │ ▪ Rate limiting       │
│ ▪ Correlate │ ▪ 4 models   │ ▪ 3 validated    │ ▪ Backoff retries     │
│             │ ▪ 5-fold CV  │   insights       │ ▪ Audit logging       │
│             │ ▪ Feature    │                  │ ▪ Graceful fallback   │
│             │   importance │                  │                        │
├─────────────┼──────────────┼──────────────────┼────────────────────────┤
│    OUTPUT   │    OUTPUT    │     OUTPUT       │       OUTPUT           │
│ cleaned_    │ model_       │ insights.json    │ Interactive Q&A +      │
│ data.csv    │ results.csv  │                  │ guardrail_log.jsonl    │
│ + 6 plots   │ + 2 plots    │                  │                        │
└─────────────┴──────────────┴──────────────────┴────────────────────────┘
```

---

## 🛠 Tech Stack

| Category | Technologies |
|:---------|:------------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn (Linear Regression, Random Forest, Gradient Boosting) |
| **LLM Integration** | Anthropic Claude API (raw HTTP POST via `requests`) |
| **Environment** | python-dotenv (`.env` for secrets management) |

---

## 📦 Parts Overview

| Part | Folder | Topic | Runnable Offline? |
|:----:|:------:|:------|:-----------------:|
| **1** | [`part1/`](part1/) | Data acquisition, cleaning & exploratory analysis → `cleaned_data.csv` + 6 plots | ✅ Yes |
| **2** | [`part2/`](part2/) | Supervised ML: predicting medical charges (4 models + baseline, CV, feature importance) | ✅ Yes |
| **3** | [`part3/`](part3/) | LLM integration with structured outputs (raw HTTP POST + strict JSON schema validation) | ✅ `--mock` |
| **4** | [`part4/`](part4/) | LLM-powered Q&A system with 10 production guardrails + offline self-tests | ✅ `--selftest` |

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn requests python-dotenv
```

### Run All Parts

```bash
# Part 1 — Generate raw data & run full EDA pipeline
cd part1 && python make_raw_data.py && python part1_eda.py && cd ..

# Part 2 — Train & evaluate ML models
cd part2 && python part2_ml.py && cd ..

# Part 3 — LLM structured outputs (offline demo)
cd part3 && python part3_llm.py --mock && cd ..

# Part 4 — Production guardrail self-tests (offline)
cd part4 && python part4_system.py --selftest && cd ..
```

> **Note:** Parts 3 & 4 need an `ANTHROPIC_API_KEY` in a `.env` file for real API calls. The `--mock` and `--selftest` modes run the complete pipelines offline without any API key.

---

## 📖 Detailed Part Descriptions

### Part 1 — Data Acquisition, Cleaning & Exploratory Analysis

**Script:** [`part1/part1_eda.py`](part1/part1_eda.py) &nbsp;|&nbsp; **Data Generator:** [`part1/make_raw_data.py`](part1/make_raw_data.py)

The raw dataset is created by injecting **seeded, reproducible** data-quality problems into the clean insurance CSV — mimicking real-world client data:

| Problem Injected | Details |
|:-----------------|:--------|
| Corrupted `age` column | 19 entries like `"45 yrs"` → forces `object` dtype |
| Missing values | ~5–8% nulls across `bmi`, `charges`, `annual_income` |
| High-null column | `referral_code` with ~31% nulls |
| Duplicate rows | 40 exact duplicates |

**Tasks completed:**

| # | Task | Key Finding |
|:-:|:-----|:------------|
| 1 | Load & inspect | 1,378 rows × 9 columns; `age` incorrectly inferred as `object` |
| 2 | Null analysis | `referral_code` exceeds 20% threshold → dropped; others median-imputed |
| 3 | Duplicate removal | 40 duplicates removed (1,378 → 1,338 rows) |
| 4 | Dtype correction | `age` coerced to numeric; categoricals → `category` dtype (70.5% memory savings) |
| 5 | Descriptive stats | `annual_income` most skewed (+1.81); `charges` second (+1.53) |
| 6 | Outlier detection | `bmi`: 16 outliers (1.2%); `charges`: 135 outliers (10.8%) — **retained** |
| 7 | Visualizations | 5 plot types: line, bar, histogram, scatter, box |
| 8 | Correlation heatmap | Strongest pair: `age`–`charges` (r = 0.27) |
| 9 | Advanced analysis | Spearman vs Pearson comparison; mean vs median imputation; grouped aggregation |
| 10 | Export | `cleaned_data.csv` — zero nulls, clean dtypes |

**Generated Visualizations (6 plots):**

| Plot | Description |
|:-----|:------------|
| `line_charges_by_index.png` | Charges distribution across dataset rows |
| `bar_mean_charges_by_region.png` | Regional comparison of mean medical charges |
| `hist_most_skewed.png` | Distribution of the most skewed variable |
| `scatter_age_vs_charges.png` | Age vs charges colored by smoker status |
| `box_charges_by_smoker.png` | Charge distribution by smoking status |
| `heatmap_pearson.png` | Pearson correlation matrix heatmap |

---

### Part 2 — Supervised Machine Learning

**Script:** [`part2/part2_ml.py`](part2/part2_ml.py)

Predicts annual medical `charges` using four regression models plus a baseline:

**Model Performance (Held-Out 20% Test Set):**

| Model | MAE ($) | RMSE ($) | R² |
|:------|--------:|---------:|---:|
| Baseline (mean) | 8,924 | 11,823 | −0.00 |
| Linear Regression | 4,433 | 6,246 | 0.72 |
| Linear Regression (log target) | 4,041 | 7,496 | 0.60 |
| Gradient Boosting | 3,150 | 5,467 | 0.79 |
| **🏆 Random Forest** | **3,109** | **5,308** | **0.80** |

**5-Fold Cross-Validated R²:**

| Model | CV R² |
|:------|:------|
| Linear Regression | 0.670 ± 0.037 |
| Gradient Boosting | 0.736 ± 0.037 |
| **Random Forest** | **0.748 ± 0.038** |

**Feature Importance (Gradient Boosting):**

| Feature | Importance |
|:--------|:---------:|
| `smoker_yes` | **0.63** |
| `bmi` | 0.18 |
| `age` | 0.12 |
| `annual_income` | 0.05 |
| Others | < 0.01 |

> **Key Insight:** Smoker status is the dominant cost driver (63% importance), followed by BMI and age. Tree ensembles outperform linear models by ~8 R² points due to the interactive, non-linear nature of the relationships.

---

### Part 3 — LLM Integration with Structured Outputs

**Script:** [`part3/part3_llm.py`](part3/part3_llm.py)

Integrates with the **Anthropic Claude API** via raw HTTP POST (no SDK) to generate structured analytical insights:

- Computes summary statistics from `cleaned_data.csv`
- Sends statistics as context with a strict JSON schema prompt
- **Validates** the response field-by-field before acceptance
- Outputs exactly **3 structured insights** with `finding`, `evidence`, `confidence`, and `recommendation`

**Validated Insights Example:**

| # | Finding | Confidence |
|:-:|:--------|:----------:|
| 1 | Smokers cost ~3.6× more than non-smokers ($30,039 vs $8,447) | 🟢 High |
| 2 | Age–charges relationship is monotonic but non-linear (Spearman 0.508 vs Pearson 0.273) | 🟢 High |
| 3 | Charges are heavily right-skewed (mean $12,939 well above median $9,289) | 🟡 Medium |

---

### Part 4 — LLM-Powered Intelligent System with Guardrails

**Script:** [`part4/part4_system.py`](part4/part4_system.py)

**InsureBot** — a production-grade Q&A assistant with **10 enterprise guardrails**:

| # | Guardrail | Category | Description |
|:-:|:----------|:--------:|:------------|
| G1 | Input validation | 🛡 Input | Reject empty or >500-char inputs |
| G2 | Injection screening | 🛡 Input | Pattern-match for prompt injection attempts |
| G3 | PII redaction | 🛡 Input | Scrub emails & phone numbers before API call |
| G4 | Topic guard | 🛡 Input | Only dataset-related questions allowed |
| G5 | Schema validation | 📤 Output | Enforce strict JSON output schema |
| G6 | Repair retry | 📤 Output | Auto re-prompt on schema failure |
| G7 | Backoff retries | ⚙️ Reliability | Exponential backoff on 429/5xx/timeouts |
| G8 | Rate limiting | ⚙️ Reliability | Client-side minimum call interval |
| G9 | Graceful fallback | ⚙️ Reliability | Cached answer if all retries fail |
| G10 | Audit logging | 📊 Observability | JSONL log with timestamp, latency, cost |

**Self-Test Results:** `10/10 PASS` ✅

```
PASS  G1 empty input rejected
PASS  G1 over-length input rejected
PASS  G2 injection pattern rejected
PASS  G3 email redacted
PASS  G3 phone redacted
PASS  G4 off-topic question refused
PASS  G5 on-topic question answered
PASS  G5 schema keys present
PASS  G5 validator rejects bad schema
PASS  G10 audit log written
```

---

## 📊 Key Results

<table>
  <tr>
    <td align="center"><strong>🔍 1,338</strong><br/>Cleaned Records</td>
    <td align="center"><strong>📈 0.80 R²</strong><br/>Best Model Score</td>
    <td align="center"><strong>💰 $3,109</strong><br/>Best MAE</td>
    <td align="center"><strong>🛡 10/10</strong><br/>Guardrails Passing</td>
  </tr>
  <tr>
    <td align="center"><strong>📊 8</strong><br/>Visualizations</td>
    <td align="center"><strong>🤖 5</strong><br/>Models Trained</td>
    <td align="center"><strong>🧬 3</strong><br/>LLM Insights</td>
    <td align="center"><strong>⚡ 3.56×</strong><br/>Smoker Cost Multiplier</td>
  </tr>
</table>

---

## 📂 Project Structure

```
ML_CAP_PRO/
│
├── 📄 README.md                    # This file
├── 📄 .gitignore                   # Excludes .env, __pycache__, logs
│
├── 📁 part1/                       # Data Acquisition & EDA
│   ├── insurance.csv               # Original public dataset
│   ├── make_raw_data.py            # Injects seeded data-quality problems
│   ├── raw_insurance.csv           # Generated "raw client file"
│   ├── part1_eda.py                # Full EDA pipeline (Tasks 1–10)
│   ├── cleaned_data.csv            # Output: cleaned dataset
│   ├── README.md                   # Detailed Part 1 documentation
│   └── 📁 plots/                   # 6 generated visualizations
│       ├── line_charges_by_index.png
│       ├── bar_mean_charges_by_region.png
│       ├── hist_most_skewed.png
│       ├── scatter_age_vs_charges.png
│       ├── box_charges_by_smoker.png
│       └── heatmap_pearson.png
│
├── 📁 part2/                       # Supervised Machine Learning
│   ├── part2_ml.py                 # Model training & evaluation
│   ├── cleaned_data.csv            # Input data (from Part 1)
│   ├── model_results.csv           # Generated metrics table
│   ├── README.md                   # Detailed Part 2 documentation
│   └── 📁 plots/                   # 2 generated visualizations
│       ├── feature_importance.png
│       └── pred_vs_actual.png
│
├── 📁 part3/                       # LLM Structured Outputs
│   ├── part3_llm.py                # Stats → HTTP POST → validated JSON
│   ├── cleaned_data.csv            # Input data (from Part 1)
│   ├── insights.json               # Generated validated insights
│   ├── .env.example                # Documents required env variable
│   └── README.md                   # Detailed Part 3 documentation
│
└── 📁 part4/                       # Production LLM System
    ├── part4_system.py             # InsureBot + 10 guardrails
    ├── cleaned_data.csv            # Input data (from Part 1)
    ├── .env.example                # Documents required env variable
    ├── guardrail_log.jsonl         # Generated audit log
    └── README.md                   # Detailed Part 4 documentation
```

---

## ⚙ Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/akash02062005/ML_CAP_PRO.git
cd ML_CAP_PRO
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn requests python-dotenv
```

### 3. Configure API Key (Parts 3 & 4 only)

```bash
# Create .env file in part3/ and/or part4/
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > part3/.env
echo "ANTHROPIC_API_KEY=sk-ant-your-key-here" > part4/.env
```

> ⚠️ **Security:** The `.env` file is git-ignored. No API keys are committed anywhere in this repository. See `.env.example` in each directory for the expected format.

### 4. Run Everything

```bash
# Full pipeline (offline-safe)
cd part1 && python make_raw_data.py && python part1_eda.py && cd ..
cd part2 && python part2_ml.py && cd ..
cd part3 && python part3_llm.py --mock && cd ..
cd part4 && python part4_system.py --selftest && cd ..
```

> 💡 Everything is **seeded** (`random_state=42`) and **fully reproducible**. The `--mock` and `--selftest` modes allow the complete pipeline to run without an API key.

---

## 📝 License

This project is developed as part of an academic capstone. All code is original and the dataset is publicly available.

---
