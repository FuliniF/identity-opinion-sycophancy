"""All three ANOVA figures, computed from one shared data pipeline.

Three variance decompositions, drawn off a single set of loaders so the
paper can rely on their being mutually consistent:

  (1) 2x2   |Delta| ~ identity x opinion                (no role-play)
  (2) 3-way stance score ~ anchor x identity x opinion   (role-play)
  (3) 2x2x2 |Delta| ~ identity x opinion x role-play

|Delta| is the absolute stance shift from the matching anchor-only
baseline -- the no-role-play baseline for no-role-play cells, the
role-play baseline for role-play cells. (1) and (3) therefore share the
|Delta| definition and the no-role-play baseline exactly. (2)
decomposes the *raw* stance score instead: a deliberately different DV,
because it answers "where the model sits", not "how far a user signal
moves it from its own baseline".

Outputs (latex/figures/):
  anova_2x2_no_roleplay.png  -- one panel,  Section 4.1
  anova_roleplay_pair.png    -- two panels, Section 4.4
Panels show the explained variance only; the residual (item-level, and
large) is reported in the caption, not drawn. An eta^2 table for every
analysis is printed for cross-checking against the paper.
"""

import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from matplotlib.patches import Patch
from statsmodels.formula.api import ols

from subject_models import FRIENDLY_NAMES as MODELS, MODEL_SHORT as SHORT

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
PAPER_FIG = os.path.join(os.path.dirname(ROOT), "latex", "figures")
# Default = single-judge evaluations/; set SYCO_EVAL_ROOT=evaluations/
# consensus to read the aggregated 3-judge panel (aggregate_judges.py).
EVAL = os.environ.get("SYCO_EVAL_ROOT") or os.path.join(ROOT, "evaluations")
FIELDS = ["diplomatic", "economy", "energy", "technology", "welfare"]
# MODELS + SHORT are the shared subject roster -- see subject_models.py.
# one factor -> one colour, shared across all three figures
COLOR = {"Model persona": "#8172B2", "Identity": "#4C72B0",
         "Opinion": "#C44E52", "Role-play": "#55A868",
         "Interactions": "#ABAE05"}


# --- loaders (one definition, used by all three analyses) ----------------

def _csv_rows(subdir, model, task):
    """Rows from evaluations/<subdir>/<model>/<task>_global_<field>.csv."""
    rows = []
    for field in FIELDS:
        path = os.path.join(EVAL, subdir, model, f"{task}_global_{field}.csv")
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append(dict(
                    field=field, id=int(float(r["id"])),  # mcc: id may be float-string ("0.0") in as_characters CSVs; original: int(r["id"])
                    score=float(r["overall_score"]),
                    anchor=r.get("was_responded_as", "") or "",
                    character=r.get("character", "") or "",
                    side=r.get("side", "") or ""))
    return rows


def _oo_rows(subdir, model):
    """Rows from the opinion-only JSON evals (A = left, B = right)."""
    rows = []
    for fp in glob.glob(os.path.join(EVAL, subdir, model, "*.json")):
        field = os.path.basename(fp).split("_")[-1][:-5]
        for rec in json.load(open(fp)):
            for s in ("A", "B"):
                rows.append(dict(
                    field=field, id=rec["id"], side=s,
                    score=rec[s]["overall_stance_score"],
                    anchor=rec.get("responding_as", "") or ""))
    return rows


def _eta(formula, data):
    """Type-II eta^2 (% of total SS) for every term, including Residual."""
    tab = sm.stats.anova_lm(ols(formula, data=data).fit(), typ=2)
    ss = tab["sum_sq"]
    return {k: 100 * ss[k] / ss.sum() for k in ss.index}


# --- the three analyses --------------------------------------------------

def anova_2x2(model):
    """(1) |Delta| ~ identity x opinion, no role-play."""
    anc = {(r["field"], r["id"]): r["score"]
           for r in _csv_rows("csvs", model, "baseline")}
    rows = []
    for r in _csv_rows("csvs", model, "stereotype"):
        k = (r["field"], r["id"])
        if k not in anc: continue
        rows.append((1, 0, abs(r["score"] - anc[k])))
    for r in _csv_rows("csvs", model, "sycophancy"):
        k = (r["field"], r["id"])
        if k not in anc: continue
        rows.append((1, 1, abs(r["score"] - anc[k])))
    for r in _oo_rows("opinion_only", model):
        k = (r["field"], r["id"])
        if k not in anc: continue
        rows.append((0, 1, abs(r["score"] - anc[k])))
    for _ in anc:                              # (no, no): |anchor - anchor| = 0
        rows.append((0, 0, 0.0))
    d = pd.DataFrame(rows, columns=["identity", "opinion", "absd"])
    return _eta("absd ~ C(identity)*C(opinion)", d)


def anova_3way(model):
    """(2) raw stance score ~ anchor x identity x opinion, role-play."""
    rows = [(r["anchor"], r["character"], r["side"], r["score"])
            for r in _csv_rows("csvs_as_characters", model, "sycophancy")]
    d = pd.DataFrame(rows, columns=["anchor", "identity", "opinion", "score"])
    return _eta("score ~ C(anchor)*C(identity)*C(opinion)", d)


def anova_2x2x2(model):
    """(3) |Delta| ~ identity x opinion x role-play."""
    anc_n = {(r["field"], r["id"]): r["score"]
             for r in _csv_rows("csvs", model, "baseline")}
    anc_r = {(r["field"], r["id"], r["anchor"]): r["score"]
             for r in _csv_rows("csvs_as_characters", model, "baseline")}
    rows = []
    for r in _csv_rows("csvs", model, "stereotype"):
        k = (r["field"], r["id"])
        if k not in anc_n: continue
        rows.append((1, 0, 0, abs(r["score"] - anc_n[k])))
    for r in _oo_rows("opinion_only", model):
        k = (r["field"], r["id"])
        if k not in anc_n: continue
        rows.append((0, 1, 0, abs(r["score"] - anc_n[k])))
    for r in _csv_rows("csvs", model, "sycophancy"):
        k = (r["field"], r["id"])
        if k not in anc_n: continue
        rows.append((1, 1, 0, abs(r["score"] - anc_n[k])))
    for _ in anc_n:
        rows.append((0, 0, 0, 0.0))
    for r in _csv_rows("csvs_as_characters", model, "stereotype"):
        k = (r["field"], r["id"], r["anchor"])
        if k not in anc_r: continue
        rows.append((1, 0, 1, abs(r["score"] - anc_r[k])))
    for r in _csv_rows("csvs_as_characters", model, "sycophancy"):
        k = (r["field"], r["id"], r["anchor"])
        if k not in anc_r: continue
        rows.append((1, 1, 1, abs(r["score"] - anc_r[k])))
    for r in _oo_rows("opinion_only_roleplay", model):
        k = (r["field"], r["id"], r["anchor"])
        if k not in anc_r: continue
        rows.append((0, 1, 1, abs(r["score"] - anc_r[k])))
    for _ in anc_r:
        rows.append((0, 0, 1, 0.0))
    d = pd.DataFrame(rows, columns=["identity", "opinion", "roleplay", "absd"])
    return _eta("absd ~ C(identity)*C(opinion)*C(roleplay)", d)


# --- turn an eta^2 dict into the segments a bar gets --------------------


def _seg_2x2(eta):
    return {"Identity": eta.get("C(identity)", 0.0),
            "Opinion": eta.get("C(opinion)", 0.0)}


def _seg_3way(eta):
    return {"Model persona": eta.get("C(anchor)", 0.0),
            "Identity": eta.get("C(identity)", 0.0),
            "Opinion": eta.get("C(opinion)", 0.0)}


def _seg_2x2x2(eta):
    return {"Identity": eta.get("C(identity)", 0.0),
            "Opinion": eta.get("C(opinion)", 0.0),
            "Role-play": eta.get("C(roleplay)", 0.0)}


def _frame(seg_by_model, models):
    """DataFrame indexed by short model name, first model at the top of barh."""
    df = pd.DataFrame({SHORT[m]: seg_by_model[m] for m in models}).T
    return df.reindex([SHORT[m] for m in reversed(models)])


def _draw(ax, df, title, label_fs, tick_fs, title_fs):
    df.plot(kind="barh", stacked=True, ax=ax, width=0.66, legend=False,
            color=[COLOR[c] for c in df.columns])
    ax.set_title(title, fontsize=title_fs, pad=7)
    ax.set_xlim(0, df.sum(axis=1).max() * 1.16)
    ax.set_xlabel("Variance explained ($\\eta^2$, %)", fontsize=label_fs)
    ax.set_ylabel("")
    ax.tick_params(labelsize=tick_fs)
    ax.spines[["top", "right"]].set_visible(False)
    thr = ax.get_xlim()[1] * 0.062            # skip slivers that would collide
    for p in ax.patches:
        w = p.get_width()
        if w < thr:
            continue
        x, y = p.get_xy()
        ax.text(x + w / 2, y + p.get_height() / 2, f"{w:.1f}", ha="center",
                va="center", fontsize=tick_fs - 1.5, color="white",
                fontweight="bold")


# --- figures -------------------------------------------------------------

# mcc: 2026-05-23 — flatter (wider, less tall) per user request. Previously
# mcc: (3.6, 0.30*N+0.7) was portrait single-column; now two-column landscape
# mcc: so the y-axis model list doesn't dominate vertical space.
# def figure_2x2(eta2x2, models):
#     """Section 4.1: single compact panel, single-column width."""
#     df = _frame({m: _seg_2x2(eta2x2[m]) for m in models}, models)
#     # height scales with the model count (the full ~14-model no-role-play set).
#     fig, ax = plt.subplots(figsize=(3.6, 0.30 * len(models) + 0.7))
#     _draw(ax, df, "$|\\Delta|$ shift: identity vs. opinion ($\\eta^2$)",
#           label_fs=8, tick_fs=7.5, title_fs=8.5)
#     ax.legend(df.columns, fontsize=6.8, loc="upper right", frameon=False,
#               handlelength=1.1, borderaxespad=0.2)
#     fig.tight_layout()
#     fig.savefig(os.path.join(PAPER_FIG, "anova_2x2_no_roleplay.png"), dpi=220,
#                 bbox_inches="tight")
#     plt.close(fig)
def figure_2x2(eta2x2, models):
    """Section 4.1: flatter cross-column panel."""
    df = _frame({m: _seg_2x2(eta2x2[m]) for m in models}, models)
    # mcc: two-column landscape. height scales with the model count but
    # mcc: with a smaller per-bar slice so the figure stays flat.
    fig, ax = plt.subplots(figsize=(7.0, 0.30 * len(models) + 0.7))
    _draw(ax, df, "$|\\Delta|$ shift: identity vs. opinion ($\\eta^2$)",
          label_fs=11, tick_fs=10.5, title_fs=12)
    ax.legend(df.columns, fontsize=10, loc="upper right", frameon=False,
              handlelength=1.1, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER_FIG, "anova_2x2_no_roleplay.png"), dpi=220,
                bbox_inches="tight")
    plt.close(fig)


def figure_roleplay_stance(eta3way, models):
    """Section 4.4 Fig A: stance score ~ model persona x identity x opinion."""
    df = _frame({m: _seg_3way(eta3way[m]) for m in models}, models)
    fig, ax = plt.subplots(figsize=(7.0, 0.30 * len(models) + 0.9))
    _draw(ax, df, "Which persona sets the baseline? (stance score, $\\eta^2$)",
          label_fs=11, tick_fs=10.5, title_fs=12)
    order = ["Model persona", "Identity", "Opinion"]
    ax.legend(handles=[Patch(facecolor=COLOR[k], label=k) for k in order],
              loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER_FIG, "anova_roleplay_stance.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def figure_roleplay_shift(eta2x2x2, models):
    """Section 4.4 Fig B: |Delta| ~ identity x opinion x role-play presence."""
    df = _frame({m: _seg_2x2x2(eta2x2x2[m]) for m in models}, models)
    fig, ax = plt.subplots(figsize=(7.0, 0.30 * len(models) + 0.9))
    _draw(ax, df, "Does signal presence change the shift? ($|\\Delta|$, $\\eta^2$)",
          label_fs=11, tick_fs=10.5, title_fs=12)
    order = ["Identity", "Opinion", "Role-play"]
    ax.legend(handles=[Patch(facecolor=COLOR[k], label=k) for k in order],
              loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER_FIG, "anova_roleplay_shift.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


# mcc: 2026-05-25 — split old two-panel figure_roleplay_pair into two
# mcc: single-column figures (figure_roleplay_stance / figure_roleplay_shift).
# def figure_roleplay_pair(eta3way, eta2x2x2, models):
#     """Section 4.4: two panels side by side, cross-column width."""
#     df_a = _frame({m: _seg_3way(eta3way[m]) for m in models}, models)
#     df_b = _frame({m: _seg_2x2x2(eta2x2x2[m]) for m in models}, models)
#     fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 0.34 * len(models) + 1.0))
#     _draw(axa, df_a, "(a) Role-play stance score", label_fs=12, tick_fs=11, title_fs=13)
#     _draw(axb, df_b, "(b) $|\\Delta|$ stance shift", label_fs=12, tick_fs=11, title_fs=13)
#     order = ["Model anchor", "Identity", "Opinion", "Role-play"]
#     fig.legend(handles=[Patch(facecolor=COLOR[k], label=k) for k in order],
#                loc="lower center", ncol=4, frameon=False, fontsize=11,
#                bbox_to_anchor=(0.5, -0.06))
#     fig.tight_layout(w_pad=2.5, rect=(0, 0.04, 1, 1))
#     fig.savefig(os.path.join(PAPER_FIG, "anova_roleplay_pair.png"), dpi=200,
#                 bbox_inches="tight")
#     plt.close(fig)


def _print_table(name, eta_by_model, terms, models):
    print(f"\n=== {name}: eta^2 (%) ===")
    print(f"{'Model':<20}" + "".join(f"{t:>14}" for t in terms))
    for m in models:
        e = eta_by_model[m]
        print(f"{SHORT[m]:<20}"
              + "".join(f"{e.get(t, 0.0):>14.2f}" for t in terms))


def _present_noroleplay(model):
    """True if this model has the no-role-play eval trees (ANOVA 1)."""
    return (os.path.exists(os.path.join(
                EVAL, "csvs", model, "baseline_global_diplomatic.csv"))
            and os.path.exists(os.path.join(
                EVAL, "opinion_only", model,
                "opinion_only_eval_diplomatic.json")))


def _present_roleplay(model):
    """True if this model has the role-play eval trees (ANOVAs 2 and 3)."""
    return (os.path.exists(os.path.join(
                EVAL, "csvs_as_characters", model,
                "baseline_global_diplomatic.csv"))
            and os.path.exists(os.path.join(
                EVAL, "opinion_only_roleplay", model,
                "opinion_only_roleplay_eval_diplomatic.json")))


def main():
    os.makedirs(PAPER_FIG, exist_ok=True)
    # Missing-data guard, split by arm. The no-role-play 2x2 (ANOVA 1) uses
    # the full roster; the two role-play ANOVAs (2, 3) use only the models
    # that have role-play data -- the 5 original models (team-lead, 2026-05-19).
    norp = [m for m in MODELS if _present_noroleplay(m)]
    rp = [m for m in MODELS
          if _present_roleplay(m) and _present_noroleplay(m)]
    skipped = [m for m in MODELS if m not in norp]
    if skipped:
        print(f"[warn] {len(skipped)} model(s) lack no-role-play eval data "
              f"under {EVAL}, skipping: {', '.join(skipped)}")
    if not norp:
        raise SystemExit("no models have eval data; nothing to do.")
    print(f"no-role-play ANOVA: {len(norp)} model(s) -- {', '.join(norp)}")
    print(f"role-play ANOVAs:   {len(rp)} model(s) -- {', '.join(rp)}")

    eta2x2 = {m: anova_2x2(m) for m in norp}
    eta3way = {m: anova_3way(m) for m in rp}
    eta2x2x2 = {m: anova_2x2x2(m) for m in rp}

    _print_table("(1) 2x2  |Delta| ~ identity x opinion", eta2x2,
                  ["C(identity)", "C(opinion)", "C(identity):C(opinion)",
                   "Residual"], norp)
    if rp:
        _print_table("(2) 3-way  score ~ anchor x identity x opinion", eta3way,
                      ["C(anchor)", "C(identity)", "C(opinion)",
                       "C(anchor):C(identity)", "C(anchor):C(opinion)",
                       "C(identity):C(opinion)",
                       "C(anchor):C(identity):C(opinion)", "Residual"], rp)
        _print_table("(3) 2x2x2  |Delta| ~ identity x opinion x role-play",
                      eta2x2x2,
                      ["C(identity)", "C(opinion)", "C(roleplay)",
                       "C(identity):C(opinion)", "C(identity):C(roleplay)",
                       "C(opinion):C(roleplay)",
                       "C(identity):C(opinion):C(roleplay)", "Residual"], rp)

    figure_2x2(eta2x2, norp)
    if rp:
        figure_roleplay_stance(eta3way, rp)
        figure_roleplay_shift(eta2x2x2, rp)
    else:
        print("[warn] no role-play data -- skipping roleplay figures")
    print(f"\nsaved -> {PAPER_FIG}/anova_2x2_no_roleplay.png")
    if rp:
        print(f"saved -> {PAPER_FIG}/anova_roleplay_stance.png")
        print(f"saved -> {PAPER_FIG}/anova_roleplay_shift.png")


if __name__ == "__main__":
    main()
