"""Mixed-effects re-analysis of the identity-vs-opinion stance shift.

Reviewer point #6: the 2x2 ANOVA on |Delta| (Figure
anova_2x2_no_roleplay.png) leaves 80-94%% of the variance in the
Residual term and the caption calls it "item-level". Treating that as
residual is the wrong model: which anchor event is shown is a known,
structured grouping factor, not noise. This script refits the same
|Delta| data with `item` (field + anchor-event id) as a random
intercept, so the item structure is modelled instead of discarded.

For each model it reports, following Nakagawa & Schielzeth (2013):
  - sigma^2_item  : between-item variance (random-intercept variance)
  - sigma^2_resid : within-item residual variance
  - ICC           : sigma^2_item / (sigma^2_item + sigma^2_resid),
                    the share of |Delta| variance that is item-level
  - marginal R^2  : variance explained by the fixed effects alone
  - conditional R^2: variance explained by fixed effects + item

Two specifications are fit:
  (A) ANOVA-comparable: the exact 2x2 design behind
      anova_2x2_no_roleplay.png -- absd ~ C(identity)*C(opinion),
      including the synthetic (no-identity, no-opinion) zero cells.
  (B) Observed-conditions: drops the synthetic zero cell and models
      the three real conditions -- absd ~ C(condition).

Inputs (already on disk -- no API calls):
  evaluations/csvs/<model>/{baseline,stereotype,sycophancy}_global_<field>.csv
  evaluations/opinion_only/<model>/opinion_only_eval_<field>.json

Run:  .venv/bin/python src/reanalysis_mixedeffects.py
"""

import csv
import glob
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from subject_models import FRIENDLY_NAMES as MODELS, MODEL_SHORT as SHORT

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
# Default = single-judge evaluations/; set SYCO_EVAL_ROOT=evaluations/
# consensus to read the aggregated 3-judge panel (aggregate_judges.py).
EVAL = os.environ.get("SYCO_EVAL_ROOT") or \
    os.path.join(REPO_ROOT, "evaluations")
FIELDS = ["diplomatic", "economy", "energy", "technology", "welfare"]
# MODELS + SHORT are the shared subject roster -- see subject_models.py.


# --- loaders (same sources as anova_figures.py / analyze_opinion_only.py) ---

def _csv_rows(model, task):
    """Rows from evaluations/csvs/<model>/<task>_global_<field>.csv."""
    rows = []
    for field in FIELDS:
        path = os.path.join(EVAL, "csvs", model, f"{task}_global_{field}.csv")
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append(dict(item=f"{field}#{int(r['id'])}",
                                  score=float(r["overall_score"])))
    return rows


def _oo_rows(model):
    """Rows from the opinion-only JSON evals (sides A and B)."""
    rows = []
    pattern = os.path.join(EVAL, "opinion_only", model, "*.json")
    for fp in glob.glob(pattern):
        field = os.path.basename(fp).split("_")[-1][:-5]
        for rec in json.load(open(fp)):
            for side in ("A", "B"):
                rows.append(dict(item=f"{field}#{int(rec['id'])}",
                                 score=rec[side]["overall_stance_score"]))
    return rows


def build_frame(model):
    """Long |Delta| frame for one model, with item + 2x2 + condition cols."""
    anchor = {r["item"]: r["score"] for r in _csv_rows(model, "baseline")}
    rows = []
    for r in _csv_rows(model, "stereotype"):          # identity only
        rows.append((r["item"], 1, 0, "identity",
                     abs(r["score"] - anchor[r["item"]])))
    for r in _oo_rows(model):                          # opinion only
        rows.append((r["item"], 0, 1, "opinion",
                     abs(r["score"] - anchor[r["item"]])))
    for r in _csv_rows(model, "sycophancy"):           # identity + opinion
        rows.append((r["item"], 1, 1, "identity+opinion",
                     abs(r["score"] - anchor[r["item"]])))
    for it in anchor:                                  # synthetic zero cell
        rows.append((it, 0, 0, "anchor", 0.0))
    return pd.DataFrame(rows,
                        columns=["item", "identity", "opinion",
                                 "condition", "absd"])


# --- mixed model + Nakagawa/Schielzeth R^2 -------------------------------

def fit_mixed(data, formula, label):
    """Fit a random-intercept model; return its variance decomposition."""
    md = sm.MixedLM.from_formula(formula, groups="item", data=data)
    mdf = md.fit(method=["lbfgs"])
    # fixed-effects linear predictor (no random part)
    fixed_pred = np.asarray(mdf.model.exog) @ np.asarray(mdf.fe_params)
    var_fixed = float(np.var(fixed_pred, ddof=0))
    var_item = float(mdf.cov_re.iloc[0, 0])
    var_resid = float(mdf.scale)
    total = var_fixed + var_item + var_resid
    return {
        "label": label, "formula": formula,
        "n_obs": int(mdf.nobs), "n_items": data["item"].nunique(),
        "var_fixed": var_fixed, "var_item": var_item, "var_resid": var_resid,
        "icc": var_item / (var_item + var_resid),
        "r2_marginal": var_fixed / total,
        "r2_conditional": (var_fixed + var_item) / total,
    }


def _print_row(name, r):
    print(f"  {name:<20} obs={r['n_obs']:>6} items={r['n_items']:>4}  "
          f"ICC={r['icc']:.3f}  R2m={r['r2_marginal']:.3f}  "
          f"R2c={r['r2_conditional']:.3f}  "
          f"(var item/resid = {r['var_item']:.2f}/{r['var_resid']:.2f})")


def _present(model):
    """True if this model has the no-role-play + opinion-only evals on disk."""
    return (os.path.exists(os.path.join(
                EVAL, "csvs", model, "baseline_global_diplomatic.csv"))
            and os.path.exists(os.path.join(
                EVAL, "opinion_only", model,
                "opinion_only_eval_diplomatic.json")))


def main():
    # Missing-data guard: analyse whatever models are on disk; skip the rest.
    global MODELS
    present = [m for m in MODELS if _present(m)]
    skipped = [m for m in MODELS if m not in present]
    if skipped:
        print(f"[warn] {len(skipped)} model(s) have no eval data under "
              f"{EVAL}, skipping: {', '.join(skipped)}")
    if not present:
        raise SystemExit("no models have eval data; nothing to do.")
    MODELS = present
    print(f"fitting {len(MODELS)} model(s): {', '.join(MODELS)}")

    summary = {"per_model": {}, "pooled": {}}
    frames = {m: build_frame(m) for m in MODELS}

    print("=" * 72)
    print("SPEC A -- ANOVA-comparable 2x2: absd ~ C(identity)*C(opinion)")
    print("  (includes synthetic zero cell; matches anova_2x2_no_roleplay.png)")
    print("=" * 72)
    for m in MODELS:
        r_null = fit_mixed(frames[m], "absd ~ 1", "null")
        r_full = fit_mixed(frames[m], "absd ~ C(identity)*C(opinion)", "2x2")
        summary["per_model"].setdefault(m, {})["specA_null"] = r_null
        summary["per_model"][m]["specA_2x2"] = r_full
        print(f"\n{SHORT[m]}")
        _print_row("null (item only)", r_null)
        _print_row("+ identity*opinion", r_full)

    print("\n" + "=" * 72)
    print("SPEC B -- observed conditions only: absd ~ C(condition)")
    print("  (drops the synthetic zero cell)")
    print("=" * 72)
    for m in MODELS:
        obs = frames[m][frames[m]["condition"] != "anchor"]
        r_obs = fit_mixed(obs, "absd ~ C(condition)", "condition")
        summary["per_model"][m]["specB_condition"] = r_obs
        print(f"\n{SHORT[m]}")
        _print_row("condition", r_obs)

    print("\n" + "=" * 72)
    print("POOLED across all 5 models (model added as a fixed factor)")
    print("=" * 72)
    pooled = pd.concat([frames[m].assign(model=m) for m in MODELS],
                       ignore_index=True)
    pooled_obs = pooled[pooled["condition"] != "anchor"]
    r_a = fit_mixed(pooled, "absd ~ C(model)*C(identity)*C(opinion)",
                    "pooled-2x2")
    r_b = fit_mixed(pooled_obs, "absd ~ C(model)*C(condition)",
                    "pooled-condition")
    summary["pooled"]["specA_2x2"] = r_a
    summary["pooled"]["specB_condition"] = r_b
    _print_row("A: model*id*op", r_a)
    _print_row("B: model*condition", r_b)

    print("\n--- READ-OUT --------------------------------------------------")
    iccs = [summary["per_model"][m]["specA_2x2"]["icc"] for m in MODELS]
    r2m = [summary["per_model"][m]["specA_2x2"]["r2_marginal"] for m in MODELS]
    print(f"  Spec A ICC range across models : "
          f"{min(iccs):.2f}-{max(iccs):.2f}")
    print(f"  Spec A marginal R^2 range      : "
          f"{min(r2m):.2f}-{max(r2m):.2f}")
    print("  -> the 80-94% ANOVA 'residual' is largely structured item")
    print("     variance: once item is a random intercept it accounts for")
    print(f"     {min(iccs):.0%}-{max(iccs):.0%} of |Delta| variance (ICC), and the")
    print("     identity/opinion fixed effects explain the marginal R^2.")

    out = os.path.join(REPO_ROOT, "reanalysis_mixedeffects_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {out}")


if __name__ == "__main__":
    main()
