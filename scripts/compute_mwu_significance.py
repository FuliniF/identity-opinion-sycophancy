#!/usr/bin/env python3
"""
Compute per-model Mann-Whitney U significance tests on 3-judge consensus data.

TEST DESIGN (from tab:mwu_overview caption):
  "Mann-Whitney U test of the ASSOCIATION between each bias flag and stance shift"
  = within each (model, condition), compare overall_stance_score distribution
    for rows where flag=True vs. rows where flag=False.

This is NOT a condition-vs-baseline comparison; it tests whether each bias flag
predicts the stance shift within each condition.

Scope:
  - No-role-play (w/o) columns: 13 models (csvs/)
  - Role-play (w/) columns: 5 original models (csvs_as_characters/, opinion_only_roleplay/)

For the OVERVIEW table: pool all models together per (flag, condition, roleplay),
then test flag-ON vs. flag-OFF on overall_score.

For PER-MODEL tables: test flag-ON vs. flag-OFF within each model's condition data.

Multiple-testing correction: Holm-Bonferroni.
  - Per-model per-flag tables: one family = all (model, condition, roleplay) tests
    for that flag. Apply HB across the family.
  - Overview: one family = all (condition, roleplay) tests for that flag.

Sycophantic alignment special cases:
  - "Identity only" has no opinion stated, so sycophantic_alignment is never annotated
    -> '--' (not applicable)
  - Opinion only and Identity+opinion have sycophantic_alignment -> test normally
  - Baseline CSVs do not have sycophantic_alignment -> handled by flag absence

Significance: p < 0.05 after Holm-Bonferroni correction -> "Sig."
              otherwise -> "n.s."
              Undefined (no variance in flag -> all True or all False, MWU
              undefined) -> "(undef)"
              Not applicable (syco_align + identity-only, or missing data) -> "--"
              8 new models lack role-play data -> "---" in w/ columns

Outputs:
  - evaluations/consensus/mwu_significance_13models.json
    Full test statistics + p-values + corrected results.

Usage:
  cd /path/to/identity-opinion-sycophancy
  .venv/bin/python scripts/compute_mwu_significance.py

Reproducible: 2026-05-21, mcc
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
EVAL_ROOT = ROOT / "evaluations" / "consensus"
CSV_ROOT  = EVAL_ROOT / "csvs"               # no-role-play
CHAR_ROOT = EVAL_ROOT / "csvs_as_characters"  # role-play
OP_ROOT   = EVAL_ROOT / "opinion_only"       # no-role-play opinion_only JSONs
OPR_ROOT  = EVAL_ROOT / "opinion_only_roleplay"  # role-play opinion_only JSONs

FIELDS = ["diplomatic", "economy", "energy", "technology", "welfare"]

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

MODELS_5 = [
    "deepseek-v3.1",
    "llama-3.3-70b-instruct",
    "mistral-small-3.2-24b",
    "o4-mini",
    "qwen3-32b",
]

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

FLAG_COLS = ["structural_bias", "framing_bias", "selection_bias", "normative_bias",
             "sycophantic_alignment"]
FLAG_COLS_NO_SYCO = ["structural_bias", "framing_bias", "selection_bias", "normative_bias"]

SCORE_COL_CSV  = "overall_score"          # column name in CSV files
SCORE_COL_JSON = "overall_stance_score"   # key name in opinion_only JSON files


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def bool_val(v):
    """Convert string/bool/int to True, False, or None."""
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def load_csv_flag_scores(path, flag_cols):
    """
    Load (flag -> ([scores_on], [scores_off])) from CSV.
    Uses SCORE_COL_CSV = "overall_score".
    Returns dict {flag: ([scores_on], [scores_off])}.
    """
    result = {f: ([], []) for f in flag_cols}
    if not path.exists():
        return result
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            score_str = row.get(SCORE_COL_CSV, "")
            try:
                score = float(score_str)
            except (ValueError, TypeError):
                continue
            for f in flag_cols:
                if f not in row:
                    continue
                v = bool_val(row[f])
                if v is None:
                    continue
                if v:
                    result[f][0].append(score)
                else:
                    result[f][1].append(score)
    return result


def load_json_flag_scores_opinion(path, flag_cols):
    """
    Load (flag -> ([scores_on], [scores_off])) from opinion_only or
    opinion_only_roleplay JSON. Records have 'A' and 'B' sub-dicts.
    Uses SCORE_COL_JSON = "overall_stance_score".
    """
    result = {f: ([], []) for f in flag_cols}
    if not path.exists():
        return result
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    for rec in records:
        for side in ("A", "B"):
            entry = rec.get(side, {})
            score_val = entry.get(SCORE_COL_JSON)
            try:
                score = float(score_val)
            except (ValueError, TypeError):
                continue
            for f in flag_cols:
                if f not in entry:
                    continue
                v = bool_val(entry[f])
                if v is None:
                    continue
                if v:
                    result[f][0].append(score)
                else:
                    result[f][1].append(score)
    return result


def pool_all_fields_csv(root, model, condition, flag_cols):
    """Pool flag-score pairs across all 5 fields for a CSV-based condition."""
    combined = {f: ([], []) for f in flag_cols}
    for field in FIELDS:
        path = root / model / f"{condition}_global_{field}.csv"
        d = load_csv_flag_scores(path, flag_cols)
        for f in flag_cols:
            combined[f][0].extend(d[f][0])
            combined[f][1].extend(d[f][1])
    return combined


def pool_all_fields_json(op_root, model, prefix, flag_cols):
    """Pool flag-score pairs across all 5 fields for a JSON-based condition."""
    combined = {f: ([], []) for f in flag_cols}
    for field in FIELDS:
        path = op_root / model / f"{prefix}_{field}.json"
        d = load_json_flag_scores_opinion(path, flag_cols)
        for f in flag_cols:
            combined[f][0].extend(d[f][0])
            combined[f][1].extend(d[f][1])
    return combined


# ---------------------------------------------------------------------------
# MWU test
# ---------------------------------------------------------------------------

def run_mwu(scores_on, scores_off):
    """
    Run two-sided MWU test: flag=True scores vs. flag=False scores.
    Returns (stat, p, direction, n_on, n_off, note).
    direction: '+' if median(on) > median(off), '-' otherwise.
    note: '' normally, 'no_variance' if all same value, 'insufficient_data' otherwise.
    """
    n_on  = len(scores_on)
    n_off = len(scores_off)

    if n_on < 2 or n_off < 2:
        return None, None, None, n_on, n_off, "insufficient_data"

    arr_on  = np.array(scores_on,  dtype=float)
    arr_off = np.array(scores_off, dtype=float)

    # Check for zero variance in both groups
    if arr_on.std() == 0 and arr_off.std() == 0 and arr_on[0] == arr_off[0]:
        return None, None, None, n_on, n_off, "no_variance_both"

    try:
        stat, p = mannwhitneyu(arr_on, arr_off, alternative="two-sided")
        direction = "+" if np.median(arr_on) > np.median(arr_off) else "-"
        return float(stat), float(p), direction, n_on, n_off, ""
    except Exception as e:
        return None, None, None, n_on, n_off, str(e)


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------

def holm_bonferroni(keyed_pvalues, alpha=0.05):
    """
    Apply Holm-Bonferroni correction.
    Input: list of (key, p_value) where p_value may be None.
    Returns: dict {key: True (significant) / False (not sig) / None (undefined)}.
    """
    valid = [(k, p) for k, p in keyed_pvalues if p is not None and not math.isnan(p)]
    results = {k: None for k, _ in keyed_pvalues}

    if not valid:
        return results

    valid_sorted = sorted(valid, key=lambda x: x[1])
    n = len(valid_sorted)
    reject_up_to = -1
    for i, (k, p) in enumerate(valid_sorted):
        threshold = alpha / (n - i)
        if p <= threshold:
            reject_up_to = i
        else:
            break

    for i, (k, p) in enumerate(valid_sorted):
        results[k] = (i <= reject_up_to)

    return results


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_all_mwu():
    """
    Compute MWU for all (model, flag, condition, roleplay) combinations.
    """
    print("=" * 70)
    print("Computing MWU significance on 3-judge consensus data")
    print("Test: flag-ON vs. flag-OFF on overall_stance_score within each condition")
    print("=" * 70)

    # raw_results[flag] = list of dicts
    raw_results = {f: [] for f in FLAG_COLS}

    def record(flag, model, condition, roleplay, stat, p, direction, n_on, n_off, note):
        raw_results[flag].append({
            "model": model,
            "condition": condition,
            "roleplay": roleplay,
            "stat": stat,
            "p_raw": p,
            "direction": direction,
            "n_on": n_on,
            "n_off": n_off,
            "note": note,
        })

    # ---- CONDITIONS ----
    # (condition_name, csv_condition_key, roleplay, models, load_fn)
    # We define loader factories below

    print("\nComputing per-model tests...")

    # --- No-role-play: 13 models ---
    for condition, csv_key, json_prefix, use_json in [
        ("stereotype",   "stereotype",   None,                     False),
        ("sycophancy",   "sycophancy",   None,                     False),
        ("opinion_only", None,           "opinion_only_eval",      True),
    ]:
        flag_cols = FLAG_COLS if condition in ("opinion_only", "sycophancy") else FLAG_COLS_NO_SYCO
        print(f"  {condition} w/o (13 models)...")
        for model in MODELS_13:
            if use_json:
                data = pool_all_fields_json(OP_ROOT, model, json_prefix, flag_cols)
            else:
                data = pool_all_fields_csv(CSV_ROOT, model, csv_key, flag_cols)
            for flag in flag_cols:
                scores_on, scores_off = data[flag]
                stat, p, direction, n_on, n_off, note = run_mwu(scores_on, scores_off)
                record(flag, model, condition, False, stat, p, direction, n_on, n_off, note)
            # sycophantic_alignment not applicable for stereotype
            if condition == "stereotype":
                record("sycophantic_alignment", model, condition, False,
                       None, None, None, 0, 0, "not_applicable")

    # --- Role-play: 5 models ---
    for condition, csv_key, json_prefix, use_json in [
        ("stereotype",   "stereotype",              None,                            False),
        ("sycophancy",   "sycophancy",              None,                            False),
        ("opinion_only", None,                      "opinion_only_roleplay_eval",    True),
    ]:
        flag_cols = FLAG_COLS if condition in ("opinion_only", "sycophancy") else FLAG_COLS_NO_SYCO
        print(f"  {condition} w/ (5 models)...")
        for model in MODELS_5:
            if use_json:
                data = pool_all_fields_json(OPR_ROOT, model, json_prefix, flag_cols)
            else:
                data = pool_all_fields_csv(CHAR_ROOT, model, csv_key, flag_cols)
            for flag in flag_cols:
                scores_on, scores_off = data[flag]
                stat, p, direction, n_on, n_off, note = run_mwu(scores_on, scores_off)
                record(flag, model, condition, True, stat, p, direction, n_on, n_off, note)
            if condition == "stereotype":
                record("sycophantic_alignment", model, condition, True,
                       None, None, None, 0, 0, "not_applicable")

    # ---- POOLED (for overview table) ----
    print("\nComputing pooled (overview) tests...")
    pooled_raw = {}

    for condition, csv_key, json_prefix, use_json, models, root_dir in [
        ("stereotype",   "stereotype", None,                  False, MODELS_13, CSV_ROOT),
        ("stereotype",   "stereotype", None,                  False, MODELS_5,  CHAR_ROOT),
        ("sycophancy",   "sycophancy", None,                  False, MODELS_13, CSV_ROOT),
        ("sycophancy",   "sycophancy", None,                  False, MODELS_5,  CHAR_ROOT),
        ("opinion_only", None,         "opinion_only_eval",   True,  MODELS_13, OP_ROOT),
        ("opinion_only", None,         "opinion_only_roleplay_eval", True, MODELS_5, OPR_ROOT),
    ]:
        roleplay = (root_dir in (CHAR_ROOT, OPR_ROOT))
        flag_cols = FLAG_COLS if condition in ("opinion_only", "sycophancy") else FLAG_COLS_NO_SYCO
        for flag in FLAG_COLS:
            key = (flag, condition, roleplay)
            if key not in pooled_raw:
                pooled_raw[key] = {"scores_on": [], "scores_off": []}
            if flag == "sycophantic_alignment" and condition == "stereotype":
                # not applicable
                continue
            if flag not in flag_cols:
                continue
            for model in models:
                if use_json:
                    data = pool_all_fields_json(root_dir, model, json_prefix, [flag])
                else:
                    data = pool_all_fields_csv(root_dir, model, csv_key, [flag])
                pooled_raw[key]["scores_on"].extend(data[flag][0])
                pooled_raw[key]["scores_off"].extend(data[flag][1])

    # Run MWU on pooled data
    pooled_results = {}
    for (flag, condition, roleplay), d in pooled_raw.items():
        if flag == "sycophantic_alignment" and condition == "stereotype":
            pooled_results[(flag, condition, roleplay)] = {
                "stat": None, "p_raw": None, "direction": None,
                "n_on": 0, "n_off": 0, "note": "not_applicable", "significant": None
            }
            continue
        stat, p, direction, n_on, n_off, note = run_mwu(d["scores_on"], d["scores_off"])
        pooled_results[(flag, condition, roleplay)] = {
            "stat": stat, "p_raw": p, "direction": direction,
            "n_on": n_on, "n_off": n_off, "note": note, "significant": None
        }

    # ---- Apply Holm-Bonferroni correction ----
    print("\nApplying Holm-Bonferroni correction per flag...")

    # Per-model: one family per flag
    for flag in FLAG_COLS:
        tests = raw_results[flag]
        keyed = [(i, t["p_raw"]) for i, t in enumerate(tests)]
        corrected = holm_bonferroni(keyed)
        for i, t in enumerate(tests):
            t["significant"] = corrected.get(i)

    # Pooled: one family per flag
    for flag in FLAG_COLS:
        keyed = [((flag, cond, rp), entry["p_raw"])
                 for (f, cond, rp), entry in pooled_results.items()
                 if f == flag]
        corrected = holm_bonferroni(keyed)
        for (f, cond, rp), entry in pooled_results.items():
            if f == flag:
                entry["significant"] = corrected.get((flag, cond, rp))

    # ---- Organize output ----
    output = {
        "meta": {
            "description": "MWU significance: association between bias flag and stance shift",
            "test": "flag=True scores vs. flag=False overall_stance_score, two-sided MWU",
            "correction": "Holm-Bonferroni per flag (alpha=0.05)",
            "models_no_roleplay": MODELS_13,
            "models_roleplay": MODELS_5,
            "date": "2026-05-21",
        },
        "per_model": {},
        "pooled": {},
    }

    for flag in FLAG_COLS:
        output["per_model"][flag] = {}
        for t in raw_results[flag]:
            m, cond, rp = t["model"], t["condition"], t["roleplay"]
            if m not in output["per_model"][flag]:
                output["per_model"][flag][m] = {}
            if cond not in output["per_model"][flag][m]:
                output["per_model"][flag][m][cond] = {}
            output["per_model"][flag][m][cond][str(rp)] = {k: v for k, v in t.items()
                                                            if k not in ("model","condition","roleplay")}

    for (flag, cond, rp), entry in pooled_results.items():
        if flag not in output["pooled"]:
            output["pooled"][flag] = {}
        if cond not in output["pooled"][flag]:
            output["pooled"][flag][cond] = {}
        output["pooled"][flag][cond][str(rp)] = entry

    return output


# ---------------------------------------------------------------------------
# LaTeX formatting helpers
# ---------------------------------------------------------------------------

NA_CELL   = "--"          # not applicable (syco_align + stereotype)
MISS_CELL = "---"         # 8 new models: no role-play data
UNDEF_CELL = "(undef)"    # all flags identical -> MWU undefined


def sig_label(entry):
    """Return 'Sig.', 'n.s.', '--', or '(undef)' for a per-model entry."""
    if entry is None:
        return NA_CELL
    note = entry.get("note", "")
    if note in ("not_applicable",):
        return NA_CELL
    if note in ("no_variance_both", "insufficient_data"):
        return UNDEF_CELL
    sig = entry.get("significant")
    if sig is None:
        return NA_CELL
    return "Sig." if sig else "n.s."


def get_cell(flag, model, condition, roleplay, per_model):
    """Return cell string for per-model table."""
    if roleplay and model not in MODELS_5:
        return MISS_CELL
    entry = (per_model
             .get(flag, {})
             .get(model, {})
             .get(condition, {})
             .get(str(roleplay)))
    return sig_label(entry)


def pooled_bullet(flag, condition, roleplay, pooled):
    """Return $\\bullet$ or $\\circ$ or \\mbox{--} for the overview table."""
    entry = pooled.get(flag, {}).get(condition, {}).get(str(roleplay))
    if entry is None:
        return "\\mbox{--}"
    note = entry.get("note", "")
    if note in ("not_applicable",):
        return "\\mbox{--}"
    if note in ("no_variance_both", "insufficient_data"):
        return "\\mbox{--}"
    sig = entry.get("significant")
    if sig is None:
        return "\\mbox{--}"
    return "$\\bullet$" if sig else "$\\circ$"


def render_overview_table(pooled):
    """Render the overview table body rows."""
    flags = [
        ("structural_bias",       "Structural"),
        ("framing_bias",          "Framing"),
        ("selection_bias",        "Selection"),
        ("normative_bias",        "Normative"),
        ("sycophantic_alignment", "Sycophantic alignment"),
    ]
    lines = []
    for flag_key, flag_label in flags:
        cells = []
        # Column order: Identity-only (w/o, w/), Opinion-only (w/o, w/), ID+opin (w/o, w/)
        for cond in ["stereotype", "opinion_only", "sycophancy"]:
            for rp in [False, True]:
                cells.append(pooled_bullet(flag_key, cond, rp, pooled))
        lines.append(f"{flag_label:<22} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def render_per_flag_table(flag_key, per_model):
    """Render per-flag table body (13 rows, 6 columns: condition×roleplay)."""
    lines = []
    for model in MODELS_13:
        display = MODEL_DISPLAY[model]
        cells = []
        # Identity-only (stereotype), Opinion-only, ID+opin (sycophancy)
        for cond in ["stereotype", "opinion_only", "sycophancy"]:
            for rp in [False, True]:
                cells.append(get_cell(flag_key, model, cond, rp, per_model))
        lines.append(f"{display} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def print_summary(output):
    """Print human-readable per-model summary."""
    pm = output["per_model"]
    flags_labels = {
        "structural_bias": "Structural",
        "framing_bias": "Framing",
        "selection_bias": "Selection",
        "normative_bias": "Normative",
        "sycophantic_alignment": "SycoAlign",
    }
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (per-model, per-flag)")
    print("=" * 70)
    header = (f"{'Model':<35} {'ID-only w/o':<14} {'ID-only w/':<13} "
              f"{'Op-only w/o':<14} {'Op-only w/':<13} "
              f"{'ID+Op w/o':<12} {'ID+Op w/'}")
    for flag, flabel in flags_labels.items():
        print(f"\n--- {flabel} ---")
        print(header)
        for model in MODELS_13:
            display = MODEL_DISPLAY[model]
            cells = []
            for cond in ["stereotype", "opinion_only", "sycophancy"]:
                for rp in [False, True]:
                    cells.append(get_cell(flag, model, cond, rp, pm))
            print(f"{display:<35} " + " ".join(f"{c:<13}" for c in cells))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    output = compute_all_mwu()

    # Save JSON
    out_path = EVAL_ROOT / "mwu_significance_13models.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nSaved: {out_path}")

    pm = output["per_model"]
    pooled = output["pooled"]

    print("\n\n" + "=" * 70)
    print("OVERVIEW TABLE (tab:mwu_overview) BODY")
    print("=" * 70)
    print(render_overview_table(pooled))

    for flag_key, flag_label in [
        ("structural_bias",       "STRUCTURAL (tab:mwu_structural)"),
        ("framing_bias",          "FRAMING (tab:mwu_framing)"),
        ("selection_bias",        "SELECTION (tab:mwu_selection)"),
        ("normative_bias",        "NORMATIVE (tab:mwu_normative)"),
        ("sycophantic_alignment", "SYCO ALIGN (tab:mwu_syco_align)"),
    ]:
        print(f"\n\n{'='*70}")
        print(f"{flag_label}")
        print("=" * 70)
        print(render_per_flag_table(flag_key, pm))

    print_summary(output)
    return output


if __name__ == "__main__":
    main()
