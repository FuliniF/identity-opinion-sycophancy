#!/usr/bin/env python3
"""
Compute per-model, per-field bias-flag activation rates from 3-judge
consensus CSVs and opinion_only JSONs, for all 13 models.

Outputs LaTeX table rows for:
  - baseline (Anchor only, no role-play)         → default arm CSVs: baseline_global_{field}.csv
  - stereotype (Identity only, no role-play)     → default arm CSVs: stereotype_global_{field}.csv
  - sycophancy (Identity + opinion, no role-play) → default arm CSVs: sycophancy_global_{field}.csv
  - opinion_only (Opinion only, no role-play)    → opinion_only JSONs: opinion_only_eval_{field}.json

Script path: scripts/compute_bias_activation_rates.py
Run from repo root:
  .venv/bin/python scripts/compute_bias_activation_rates.py

Reproducible: 2026-05-21, mcc
"""
import csv
import json
import os
from pathlib import Path

EVAL_ROOT = Path(__file__).parent.parent / "evaluations" / "consensus"
CSV_ROOT = EVAL_ROOT / "csvs"
OP_ROOT  = EVAL_ROOT / "opinion_only"

FIELDS = ["diplomatic", "economy", "energy", "technology", "welfare"]

# All 13 models (directory names under CSV_ROOT)
MODELS_13 = [
    "deepseek-v3.1",
    "gemma-4-31b-it",
    "glm-4.7",
    "gpt-oss-120b",
    "granite-4.1-8b",
    "kimi-k2.5",
    "llama-3.3-70b-instruct",
    "mimo-v2-flash",
    "mistral-small-3.2-24b",
    "nova-lite-v1",
    "o4-mini",
    "phi-4",
    "qwen3-32b",
]

# Prettier display names for LaTeX (used as row labels)
MODEL_DISPLAY = {
    "deepseek-v3.1":           "DeepSeek-V3.1",
    "gemma-4-31b-it":          "Gemma-4-31B-IT",
    "glm-4.7":                 "GLM-4.7",
    "gpt-oss-120b":            "GPT-OSS-120B",
    "granite-4.1-8b":          "Granite-4.1-8B",
    "kimi-k2.5":               "Kimi-K2.5",
    "llama-3.3-70b-instruct":  "Llama-3.3-70B-Instruct",
    "mimo-v2-flash":           "Mimo-V2-Flash",
    "mistral-small-3.2-24b":   "Mistral-Small-3.2-24B",
    "nova-lite-v1":            "Nova-Lite-V1",
    "o4-mini":                 "o4-mini",
    "phi-4":                   "Phi-4",
    "qwen3-32b":               "Qwen3-32B",
}

FLAG_COLS_BASE = ["structural_bias", "framing_bias", "selection_bias", "normative_bias"]
FLAG_COLS_SYCO = FLAG_COLS_BASE + ["sycophantic_alignment"]


def bool_rate(values):
    """Return percentage True from a list of boolean-like values."""
    if not values:
        return float("nan")
    trues = sum(1 for v in values if str(v).strip().lower() in ("true", "1", "yes"))
    return trues / len(values) * 100.0


def load_csv_flags(path, flag_cols):
    """Load flag columns from a consensus CSV. Returns dict {flag: [values]}."""
    result = {f: [] for f in flag_cols}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for f in flag_cols:
                if f in row:
                    result[f].append(row[f])
    return result


def load_json_flags(path, flag_cols):
    """
    Load flag columns from opinion_only JSON.
    Each record has 'A' and 'B' sub-dicts (left/right narrative).
    Pool both sides.
    """
    result = {f: [] for f in flag_cols}
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    for rec in records:
        for side in ("A", "B"):
            if side not in rec:
                continue
            entry = rec[side]
            for f in flag_cols:
                if f in entry:
                    result[f].append(entry[f])
    return result


def compute_table(condition, flag_cols, load_fn):
    """
    Returns a nested dict: {field: {model: {flag: rate_pct}}}
    """
    data = {}
    for field in FIELDS:
        data[field] = {}
        for model in MODELS_13:
            if condition == "opinion_only":
                fpath = OP_ROOT / model / f"opinion_only_eval_{field}.json"
                if not fpath.exists():
                    print(f"  MISSING: {fpath}")
                    continue
                flags = load_json_flags(fpath, flag_cols)
            else:
                fpath = CSV_ROOT / model / f"{condition}_global_{field}.csv"
                if not fpath.exists():
                    print(f"  MISSING: {fpath}")
                    continue
                flags = load_csv_flags(fpath, flag_cols)
            data[field][model] = {f: bool_rate(v) for f, v in flags.items()}
    return data


def find_max(data, flag, fields=FIELDS, models=MODELS_13):
    """Return (field, model) with the maximum value for a given flag."""
    best_val = -1
    best_fm = None
    for field in fields:
        for model in models:
            v = data.get(field, {}).get(model, {}).get(flag, float("nan"))
            if v == v and v > best_val:
                best_val = v
                best_fm = (field, model)
    return best_fm, best_val


def format_pct(v):
    return f"{v:.2f}" if v == v else "--"


def render_latex_table(data, flag_cols, has_syco_align, bold_max=True):
    """Render LaTeX table rows (tabular body only, no preamble)."""
    lines = []
    for field in FIELDS:
        first_in_field = True
        for model in MODELS_13:
            vals = data.get(field, {}).get(model, {})
            if not vals:
                continue
            cells = []
            for f in flag_cols:
                v = vals.get(f, float("nan"))
                cells.append(format_pct(v))
            # Build row
            display = MODEL_DISPLAY.get(model, model)
            syco_cell = cells[-1] if has_syco_align else "-"
            if has_syco_align:
                body_cells = cells
            else:
                body_cells = cells + ["-"]
            row_parts = [field, display] + body_cells
            lines.append(" & ".join(row_parts) + " \\\\")
        lines.append("\\midrule")
    # Remove last \midrule (before \bottomrule)
    if lines and lines[-1] == "\\midrule":
        lines.pop()
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("Computing bias activation rates from 3-judge consensus data")
    print("=" * 70)

    # ---- BASELINE (Anchor only, no role-play) ----
    print("\n[1] Baseline (Anchor only) ...")
    baseline = compute_table("baseline", FLAG_COLS_BASE, load_csv_flags)

    # ---- STEREOTYPE (Identity only, no role-play) ----
    print("[2] Stereotype (Identity only) ...")
    stereotype = compute_table("stereotype", FLAG_COLS_BASE, load_csv_flags)

    # ---- SYCOPHANCY (Identity + opinion, no role-play) ----
    print("[3] Sycophancy (Identity + opinion) ...")
    sycophancy = compute_table("sycophancy", FLAG_COLS_SYCO, load_csv_flags)

    # ---- OPINION_ONLY ----
    print("[4] Opinion only ...")
    opinion_only = compute_table("opinion_only", FLAG_COLS_SYCO, load_json_flags)

    # Print structured JSON output
    all_data = {
        "baseline":    baseline,
        "stereotype":  stereotype,
        "sycophancy":  sycophancy,
        "opinion_only": opinion_only,
    }
    out_path = Path(__file__).parent.parent / "bias_activation_rates_13models.json"
    with open(out_path, "w") as fh:
        json.dump(all_data, fh, indent=2)
    print(f"\nSaved JSON: {out_path}")

    # ---- Print LaTeX table rows ----
    for cond_key, data, has_syco, cols in [
        ("baseline",    baseline,    False, FLAG_COLS_BASE),
        ("stereotype",  stereotype,  False, FLAG_COLS_BASE),
        ("sycophancy",  sycophancy,  True,  FLAG_COLS_SYCO),
        ("opinion_only",opinion_only,True,  FLAG_COLS_SYCO),
    ]:
        print(f"\n\n{'='*70}")
        print(f"LaTeX rows for: {cond_key}")
        print("Field & Model & Structural & Framing & Selection & Normative & Syco. Align.")
        print("\\midrule")
        for field in FIELDS:
            for model in MODELS_13:
                vals = data.get(field, {}).get(model, {})
                if not vals:
                    continue
                cells = []
                for f in cols:
                    v = vals.get(f, float("nan"))
                    cells.append(format_pct(v))
                display = MODEL_DISPLAY.get(model, model)
                if has_syco:
                    row_parts = [field, display] + cells
                else:
                    row_parts = [field, display] + cells + ["-"]
                print(" & ".join(row_parts) + " \\\\")
            print("\\midrule")


if __name__ == "__main__":
    main()
