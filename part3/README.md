# Part 3 — LLM Integration with Structured Outputs

## What this does

`part3_llm.py` computes summary statistics from Part 1's `cleaned_data.csv` (row count, mean/median/skew of charges, group means by smoker and region, Pearson & Spearman age–charges correlations), sends them to a publicly accessible LLM API via a **raw HTTP POST with a JSON body** (the `requests` library — no vendor SDK), and requires the model to reply in a **strict JSON schema**: exactly three insights, each with `finding`, `evidence`, `confidence` (`low|medium|high`), and `recommendation`. The reply is parsed and **validated field-by-field** before it is accepted; any schema violation aborts with a clear error showing the raw reply. Validated output is printed and saved to `insights.json`.

## How to run

```bash
pip install pandas requests python-dotenv

# 1) real API call — put your key in a .env file (never committed):
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
python part3_llm.py

# 2) offline demo — full pipeline incl. schema validation, no key needed:
python part3_llm.py --mock
```

The script looks for `cleaned_data.csv` locally, then falls back to `../part1/cleaned_data.csv` (or pass `--data path/to/cleaned_data.csv`).

## Environment variables

| variable | purpose |
|---|---|
| `ANTHROPIC_API_KEY` | API key for the Anthropic Messages API. Loaded from `.env` via `python-dotenv`. `.env` is excluded by the repository's `.gitignore`; `.env.example` documents the expected name. No secret appears anywhere in the code or repo. |

The endpoint used is `https://api.anthropic.com/v1/messages` (model `claude-haiku-4-5-20251001`) — an HTTP POST with a JSON body returning JSON, as the brief requires. Because `call_llm()` is a single isolated function, any other JSON-in/JSON-out LLM endpoint can be substituted by editing ~10 lines.

## Design decisions

**Structured output by contract, not by trust.** The prompt embeds the exact JSON schema and forbids markdown/prose. The response then goes through three defensive layers: `extract_json()` (tolerates accidental code fences and extracts the outermost JSON object), `json.loads`, and `validate()` (checks the top-level key, the exact count of three insights, presence and non-emptiness of every required field, and that `confidence` is one of the allowed enum values). This mirrors production practice: LLM output is untrusted input until validated.

**Grounding.** The model is only given the statistics JSON and is asked to quote its evidence from those numbers, which makes fabricated evidence easy to spot in review.

**`--mock` mode** ships a canned response that passes through the *same* extraction and validation code path, so the grader can run and inspect the full pipeline without spending tokens or holding a key.

## Key findings (from a validated run)

The model consistently surfaces the same three insights the EDA supports: (1) smokers' mean charges ($30,039) are ~3.6× non-smokers' ($8,447) — high confidence; (2) the age–charges relationship is monotonic but non-linear (Spearman 0.508 vs Pearson 0.273), arguing for age-banded pricing; (3) charges are right-skewed (mean well above median), so medians should be reported and capital reserved for the tail.

## Repository contents

| file | purpose |
|---|---|
| `part3_llm.py` | stats computation, HTTP POST integration, schema validation |
| `cleaned_data.csv` | input data (from Part 1) |
| `.env.example` | documents the required environment variable (no real key) |
| `insights.json` | example validated output (generated) |
