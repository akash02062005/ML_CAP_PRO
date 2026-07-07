"""
Part 3 — LLM Integration with Structured Outputs
================================================
Sends dataset statistics (computed from Part 1's cleaned_data.csv) to a
publicly accessible LLM API via a raw HTTP POST with a JSON body, and
requires the model to answer in a STRICT JSON schema, which the script
validates before accepting.

The LLM is asked to act as a data analyst: given the summary statistics,
it must return exactly three structured insights, each with a `finding`,
supporting `evidence` (quoting the numbers it was given), a `confidence`
level, and a `recommendation` for the business.

API: Anthropic Messages API (plain HTTP POST + JSON — no SDK is used, to
demonstrate raw structured-output integration). Any compatible JSON API
could be substituted by editing `call_llm()`.

Setup
-----
1. `pip install pandas requests python-dotenv`
2. Create a `.env` file (NOT committed - see .gitignore) containing:
       ANTHROPIC_API_KEY=sk-ant-...
3. `python part3_llm.py`            # real API call
   `python part3_llm.py --mock`     # offline demo: validates a canned
                                    # response so graders can run the full
                                    # pipeline without an API key

Output: prints the validated insights and saves them to `insights.json`.
"""

import argparse
import json
import os
import sys

import pandas as pd
import requests

try:  # optional; environment variables may also be set directly
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5-20251001"

# ----------------------------------------------------------------- schema
REQUIRED_INSIGHT_KEYS = {"finding": str, "evidence": str,
                         "confidence": str, "recommendation": str}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}


def build_stats(path: str) -> dict:
    """Compute the summary statistics that will be sent to the LLM."""
    df = pd.read_csv(path)
    stats = {
        "n_rows": int(df.shape[0]),
        "columns": list(df.columns),
        "charges_mean": round(float(df["charges"].mean()), 2),
        "charges_median": round(float(df["charges"].median()), 2),
        "charges_skew": round(float(df["charges"].skew()), 3),
        "mean_charges_by_smoker": {
            k: round(v, 2) for k, v in
            df.groupby("smoker")["charges"].mean().items()},
        "mean_charges_by_region": {
            k: round(v, 2) for k, v in
            df.groupby("region")["charges"].mean().items()},
        "pearson_age_charges": round(float(df["age"].corr(df["charges"])), 3),
        "spearman_age_charges": round(
            float(df["age"].corr(df["charges"], method="spearman")), 3),
        "mean_bmi": round(float(df["bmi"].mean()), 2),
    }
    return stats


def build_prompt(stats: dict) -> str:
    return f"""You are a data analyst. Below are summary statistics from a
cleaned medical-insurance dataset (one row per insured person; `charges`
is the annual medical cost in USD).

STATISTICS (JSON):
{json.dumps(stats, indent=2)}

Respond with ONLY a JSON object - no markdown fences, no prose - matching
exactly this schema:
{{
  "insights": [
    {{
      "finding": "<one-sentence insight>",
      "evidence": "<the specific numbers from the statistics that support it>",
      "confidence": "low" | "medium" | "high",
      "recommendation": "<one actionable business recommendation>"
    }}
  ]
}}
The "insights" array must contain exactly 3 items."""


def call_llm(prompt: str) -> str:
    """Raw HTTP POST with a JSON body; returns the model's text reply."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: set ANTHROPIC_API_KEY in your environment or .env "
                 "file (see README), or run with --mock.")
    resp = requests.post(
        API_URL,
        headers={"x-api-key": api_key,
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": MODEL,
              "max_tokens": 1024,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


MOCK_RESPONSE = json.dumps({
    "insights": [
        {"finding": "Smokers cost dramatically more than non-smokers.",
         "evidence": "Mean charges: smokers $30,039.44 vs non-smokers "
                     "$8,446.88 (3.6x).",
         "confidence": "high",
         "recommendation": "Price smoker risk explicitly and fund smoking-"
                           "cessation programs to reduce claim costs."},
        {"finding": "Charges rise consistently but non-linearly with age.",
         "evidence": "Spearman age-charges 0.508 vs Pearson 0.273 indicates "
                     "a monotonic, non-proportional relationship.",
         "confidence": "high",
         "recommendation": "Use age-banded pricing tiers rather than a "
                           "single linear age loading."},
        {"finding": "The charges distribution is heavily right-skewed.",
         "evidence": "Mean $12,939.24 well above median $9,289.08; "
                     "skew about +1.5.",
         "confidence": "medium",
         "recommendation": "Report median (not mean) cost in dashboards and "
                           "reserve capital for the expensive tail."},
    ]
})


def extract_json(text: str) -> dict:
    """Parse the model reply, tolerating accidental markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.index("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model reply.")
    return json.loads(text[start:end + 1])


def validate(payload: dict) -> list:
    """Enforce the response schema; raise ValueError on any violation."""
    if not isinstance(payload, dict) or "insights" not in payload:
        raise ValueError("Top-level key 'insights' missing.")
    insights = payload["insights"]
    if not isinstance(insights, list) or len(insights) != 3:
        raise ValueError(f"'insights' must be a list of exactly 3 items, "
                         f"got {len(insights) if isinstance(insights, list) else type(insights)}.")
    for i, ins in enumerate(insights):
        for key, typ in REQUIRED_INSIGHT_KEYS.items():
            if key not in ins:
                raise ValueError(f"Insight {i}: missing key '{key}'.")
            if not isinstance(ins[key], typ) or not ins[key].strip():
                raise ValueError(f"Insight {i}: '{key}' must be a non-empty "
                                 f"string.")
        if ins["confidence"].lower() not in ALLOWED_CONFIDENCE:
            raise ValueError(f"Insight {i}: confidence must be one of "
                             f"{sorted(ALLOWED_CONFIDENCE)}.")
    return insights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true",
                    help="validate a canned response instead of calling "
                         "the API (no key needed)")
    ap.add_argument("--data", default=None, help="path to cleaned_data.csv")
    args = ap.parse_args()

    path = args.data or ("cleaned_data.csv" if os.path.exists(
        "cleaned_data.csv") else os.path.join("..", "part1",
                                              "cleaned_data.csv"))
    stats = build_stats(path)
    print("Statistics sent to the LLM:")
    print(json.dumps(stats, indent=2))

    if args.mock:
        print("\n--mock: using canned response (no API call).")
        raw = MOCK_RESPONSE
    else:
        raw = call_llm(build_prompt(stats))

    try:
        insights = validate(extract_json(raw))
    except (ValueError, json.JSONDecodeError) as e:
        sys.exit(f"SCHEMA VALIDATION FAILED: {e}\nRaw reply:\n{raw}")

    print("\n=== Validated structured insights ===")
    for i, ins in enumerate(insights, 1):
        print(f"\n[{i}] {ins['finding']}")
        print(f"    evidence:       {ins['evidence']}")
        print(f"    confidence:     {ins['confidence']}")
        print(f"    recommendation: {ins['recommendation']}")

    with open("insights.json", "w") as f:
        json.dump({"insights": insights}, f, indent=2)
    print("\nSaved insights.json — schema validation passed.")


if __name__ == "__main__":
    main()
