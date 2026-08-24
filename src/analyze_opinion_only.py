"""Analyze the opinion-only experiment and produce the numbers and figures
the revised paper needs.

Sources
  - evaluations/opinion_only/<model>/opinion_only_eval_<field>.json  (Track 1)
  - evaluations/csvs/<model>/{baseline,stereotype,sycophancy}_global_<field>.csv
        (existing no-role-play evals: anchor-only / identity-only / identity+opinion)

Outputs
  - printed summary of every number referenced by the paper placeholders
  - latex/figures/heatmap_opinion_only.png
  - latex/figures/dissociation_overview.png
  - analysis_opinion_only_summary.json  (machine-readable dump)
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, spearmanr

from subject_models import FRIENDLY_NAMES as MODELS, MODEL_SHORT

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
PAPER_FIG = os.path.join(os.path.dirname(REPO_ROOT), "latex", "figures")
# Root of the evaluation tree. Default = single-judge evaluations/; set
# SYCO_EVAL_ROOT=evaluations/consensus to analyse the aggregated 3-judge
# panel (see aggregate_judges.py).
EVAL_ROOT = os.environ.get("SYCO_EVAL_ROOT") or \
    os.path.join(REPO_ROOT, "evaluations")
FIELDS = ["diplomatic", "economy", "energy", "technology", "welfare"]
# MODELS + MODEL_SHORT are the shared subject roster -- see subject_models.py.
LEFT_PERSONAS = {"outsider left", "democratic mainstays",
                 "establishment liberals", "progressive left"}
RIGHT_PERSONAS = {"faith and flag conservatives", "committed conservatives",
                  "populist right", "ambivalent right"}
FLAGS = ["structural_bias", "framing_bias", "selection_bias", "normative_bias"]
PERSONA_ORDER = ["progressive left", "outsider left",
                 "establishment liberals", "democratic mainstays",
                 "stressed sideliners", "ambivalent right",
                 "populist right", "committed conservatives",
                 "faith and flag conservatives"]
PERSONA_SHORT = ["Progressive\nLeft", "Outsider\nLeft",
                 "Establish.\nLiberals", "Democratic\nMainstays",
                 "Stressed\nSideliners", "Ambivalent\nRight",
                 "Populist\nRight", "Committed\nCons.", "Faith &\nFlag Cons."]
# Short codes for the nine Pew personas (left-to-right spectrum order),
# used on heatmap axes; the full key lives in the figure captions.
PERSONA_CODES = ["PL", "OL", "EL", "DM", "SS", "AR", "PR", "CC", "FFC"]


def load_csvs(model, task):
    """Existing no-role-play evals as one dataframe, with a field column."""
    frames = []
    for field in FIELDS:
        path = os.path.join(EVAL_ROOT, "csvs", model,
                            f"{task}_global_{field}.csv")
        df = pd.read_csv(path)
        df["field"] = field
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_opinion_only(model):
    """Track-1 evals as a long dataframe (one row per id/side)."""
    rows = []
    for field in FIELDS:
        path = os.path.join(EVAL_ROOT, "opinion_only", model,
                            f"opinion_only_eval_{field}.json")
        for rec in json.load(open(path)):
            for side in ("A", "B"):
                d = rec[side]
                rows.append(dict(
                    field=field, id=rec["id"], side=side,
                    overall_score=d.get("overall_stance_score"),
                    structural_bias=d.get("structural_bias"),
                    framing_bias=d.get("framing_bias"),
                    selection_bias=d.get("selection_bias"),
                    normative_bias=d.get("normative_bias"),
                    sycophantic_alignment=d.get("sycophantic_alignment"),
                    refusal=d.get("refusal")))
    return pd.DataFrame(rows)


LOGO_DIR = os.path.join(REPO_ROOT, "assets", "logos")


def _logo_image(model_key, zoom=0.22):
    path = os.path.join(LOGO_DIR, f"{model_key}.png")
    if not os.path.exists(path):
        return None
    img = mpimg.imread(path)
    return OffsetImage(img, zoom=zoom)


def dissociation_overview(id_delta, dir_susc):
    """Scatter-only hero figure for the dissociation result.

    Each model's identity-only vs. opinion-only directional susceptibility
    against a y=x reference line -- distils the dissociation. The
    identity-only heatmap (formerly panel a) has been removed; the scatter
    is now the sole panel. Model logos are used as markers."""
    # x = opinion, y = identity (swapped for landscape layout).
    # Kimi's opinion susceptibility is strongly negative (≈ −5.3) and is
    # shown as an edge marker rather than stretching the whole axis leftward.
    id_vals = [dir_susc[m]["identity"] for m in MODELS]
    op_vals = [dir_susc[m]["opinion"]  for m in MODELS]
    rho, _ = spearmanr(id_vals, op_vals)

    kimi_key = next(m for m in MODELS if MODEL_SHORT[m] == "Kimi-K2.5")
    kimi_x = dir_susc[kimi_key]["opinion"]   # ≈ −5.3
    kimi_y = dir_susc[kimi_key]["identity"]  # ≈ 11.6

    # Main scatter uses only non-Kimi models for axis limits.
    other_x = [v for m, v in zip(MODELS, op_vals)  if MODEL_SHORT[m] != "Kimi-K2.5"]
    other_y = [v for m, v in zip(MODELS, id_vals) if MODEL_SHORT[m] != "Kimi-K2.5"]
    lim = max(other_x + other_y) * 1.22

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot([0, lim], [0, lim], ls="--", lw=1.2, color="#999999",
            zorder=1, label="equal susceptibility")

    # --- Kimi: edge marker + callout ------------------------------------------
    edge_x = 0.35          # just inside the left axis
    ax.annotate(
        "", xy=(edge_x, kimi_y),
        xytext=(edge_x + 1.1, kimi_y),
        arrowprops=dict(arrowstyle="<-", color="#1565C0", lw=1.5),
        zorder=4,
    )
    ax.scatter([edge_x], [kimi_y], s=160, color="#1565C0",
               edgecolor="white", lw=1.5, marker="<", zorder=4)
    ax.annotate(
        f"Kimi-K2.5\n({kimi_x:.1f})",
        (edge_x + 0.1, kimi_y + 0.9),   # above the arrow to avoid GPT-OSS overlap
        fontsize=11, va="bottom", ha="left", color="#1565C0",
    )

    # Per-model label offsets (dx_pts, dy_pts, ha) — positions (x=op, y=id):
    #   GLM      (2.2, 10.4)   GPT-OSS  (2.3, 11.8)  ← vertically close
    #   Qwen     (1.3,  9.7)   Gemma    (4.1, 11.0)
    #   o4-mini  (6.1,  6.3)   DeepSeek (5.4,  6.8)  ← close together
    #   Nova     (7.6,  5.0)   Granite  (7.7,  4.2)  ← close together
    #   Llama    (10.4, 2.9)   Mistral  (8.5,  2.6)  ← close together
    _off = {
        "GLM-4.7":               ( 17,   0, "left"),   # right of icon at same y
        "GPT-OSS-120B":          ( 17,  14, "left"),   # above icon
        "Qwen3-32B":             (  0, -18, "center"), # below icon (centered avoids left-edge clip)
        "Gemma-4-31B":           ( 17,   0, "left"),   # right of icon; was center-above → crowded GPT-OSS
        "Phi-4":                 ( 17,   0, "left"),
        "Llama-3.3-70B":         ( 17,   9, "left"),
        "Mistral-Small-3.2-24B": ( 17,  -9, "left"),
        "Nova-Lite-v1":          ( 17,   9, "left"),
        "Granite-4.1-8B":        ( 17,  -9, "left"),
        "o4-mini":               (  0,  16, "center"), # above icon; was (17,9) overlapping DeepSeek
        "DeepSeek-V3.1":         (  0, -16, "center"), # below icon; was (17,-9) overlapping o4-mini
        "mimo-v2-flash":         (-17,   0, "right"),
    }
    _dot_color = {}  # Gemma logo (blue star) is visible; no longer need dot override

    for m, xv, yv in zip(MODELS, op_vals, id_vals):
        short = MODEL_SHORT[m]
        if short == "Kimi-K2.5":
            continue           # already drawn as edge marker above
        if short in _dot_color:
            ax.scatter([xv], [yv], s=150, color=_dot_color[short],
                       edgecolor="white", lw=1.5, zorder=3)
        else:
            img_box = _logo_image(m)
            if img_box is not None:
                ab = AnnotationBbox(img_box, (xv, yv), frameon=False, zorder=3)
                ax.add_artist(ab)
            else:
                ax.scatter([xv], [yv], s=110, color="#4C72B0",
                           edgecolor="white", lw=1.2, zorder=3)
        dx, dy, ha = _off.get(short, (17, 0, "left"))
        ax.annotate(short, (xv, yv), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va="center", fontsize=11)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Opinion-only directional susceptibility", fontsize=13)
    ax.set_ylabel("Identity-only directional susceptibility", fontsize=13)
    ax.set_title(f"Susceptibility dissociates "
                 f"(Spearman $\\rho={rho:.2f}$)", fontsize=13, pad=10)
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(PAPER_FIG, "dissociation_overview.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


def _present(model):
    """True if this model has the no-role-play + opinion-only evals on disk."""
    return (os.path.exists(os.path.join(
                EVAL_ROOT, "csvs", model, "baseline_global_diplomatic.csv"))
            and os.path.exists(os.path.join(
                EVAL_ROOT, "opinion_only", model,
                "opinion_only_eval_diplomatic.json")))


def main():
    # Missing-data guard: the roster is ~16 models but a run may have data for
    # only some. Analyse whatever is on disk; skip the rest with a warning.
    global MODELS
    present = [m for m in MODELS if _present(m)]
    skipped = [m for m in MODELS if m not in present]
    if skipped:
        print(f"[warn] {len(skipped)} model(s) have no Track-1 eval data "
              f"under {EVAL_ROOT}, skipping: {', '.join(skipped)}")
    if not present:
        raise SystemExit("no models have Track-1 eval data; nothing to do.")
    MODELS = present
    print(f"analysing {len(MODELS)} model(s): {', '.join(MODELS)}")

    summary = {}
    # anchor baseline per (model, field, id)
    anchor = {}
    for model in MODELS:
        b = load_csvs(model, "baseline")
        anchor[model] = b.set_index(["field", "id"])["overall_score"].to_dict()

    # ---- opinion-only delta ----
    op_delta_rows = []
    op_eval = {}
    for model in MODELS:
        df = load_opinion_only(model)
        op_eval[model] = df
        for _, r in df.iterrows():
            base = anchor[model][(r["field"], r["id"])]
            op_delta_rows.append(dict(model=model, field=r["field"], id=r["id"],
                                      side=r["side"],
                                      delta=r["overall_score"] - base))
    opd = pd.DataFrame(op_delta_rows)

    print("\n=== OPINION-ONLY mean shift (Delta) by model and narrative side ===")
    print("  side A = left-leaning narrative, side B = right-leaning narrative")
    op_shift = {}
    for model in MODELS:
        a = opd[(opd.model == model) & (opd.side == "A")]["delta"].mean()
        b = opd[(opd.model == model) & (opd.side == "B")]["delta"].mean()
        op_shift[model] = {"A": a, "B": b}
        print(f"  {MODEL_SHORT[model]:<18} left {a:+.2f}   right {b:+.2f}")
    summary["opinion_only_shift"] = op_shift

    # ---- identity-only delta (existing stereotype) ----
    print("\n=== |Delta| by condition and model (no role-play) ===")
    absd = {}
    dir_susc = {}
    id_delta = {}
    for model in MODELS:
        st = load_csvs(model, "stereotype")
        sy = load_csvs(model, "sycophancy")
        st["delta"] = st["overall_score"] - st.apply(
            lambda r: anchor[model][(r["field"], r["id"])], axis=1)
        id_abs = st["delta"].abs().mean()
        idop_abs = (sy["overall_score"]
                    - sy.apply(lambda r: anchor[model][(r["field"], r["id"])],
                               axis=1)).abs().mean()
        op_abs = opd[opd.model == model]["delta"].abs().mean()
        absd[model] = {"identity": id_abs, "opinion": op_abs,
                       "identity+opinion": idop_abs}
        # directional susceptibility: right-pull minus left-pull
        stc = st["character"].astype(str).str.strip().str.lower()
        dir_susc[model] = {
            "identity": (st[stc.isin(RIGHT_PERSONAS)]["delta"].mean()
                         - st[stc.isin(LEFT_PERSONAS)]["delta"].mean()),
            "opinion": op_shift[model]["B"] - op_shift[model]["A"]}
        id_delta[model] = (st.assign(c=stc).groupby("c")["delta"]
                           .mean().to_dict())
        print(f"  {MODEL_SHORT[model]:<18} identity {id_abs:.2f}   "
              f"opinion {op_abs:.2f}   both {idop_abs:.2f}   "
              f"(id+op sum {id_abs + op_abs:.2f})")
    summary["abs_delta"] = absd
    summary["dir_susc"] = dir_susc
    print("\n=== Directional susceptibility (right-pull minus left-pull) ===")
    for model in MODELS:
        print(f"  {MODEL_SHORT[model]:<18} "
              f"identity {dir_susc[model]['identity']:.2f}   "
              f"opinion {dir_susc[model]['opinion']:.2f}")

    # ---- 2x2 ANOVA on |Delta| (identity present/absent x opinion present/absent)
    print("\n=== 2x2 ANOVA on |Delta|: identity x opinion (eta^2, %) ===")
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    anova_rows = []
    for model in MODELS:
        st = load_csvs(model, "stereotype")
        sy = load_csvs(model, "sycophancy")
        for _, r in st.iterrows():
            d = abs(r["overall_score"] - anchor[model][(r["field"], r["id"])])
            anova_rows.append((model, 1, 0, d))
        for _, r in sy.iterrows():
            d = abs(r["overall_score"] - anchor[model][(r["field"], r["id"])])
            anova_rows.append((model, 1, 1, d))
        for _, r in opd[opd.model == model].iterrows():
            anova_rows.append((model, 0, 1, abs(r["delta"])))
        for (field, idx) in anchor[model]:
            anova_rows.append((model, 0, 0, 0.0))
    adf = pd.DataFrame(anova_rows, columns=["model", "identity", "opinion", "absd"])
    anova_res = {}
    for model in MODELS:
        sub = adf[adf.model == model]
        m = ols("absd ~ C(identity)*C(opinion)", data=sub).fit()
        tab = sm.stats.anova_lm(m, typ=2)
        ss = tab["sum_sq"]
        tot = ss.sum()
        eta = {k: 100 * ss[k] / tot for k in ss.index}
        anova_res[model] = eta
        print(f"  {MODEL_SHORT[model]:<18} "
              f"identity {eta.get('C(identity)', 0):5.1f}  "
              f"opinion {eta.get('C(opinion)', 0):5.1f}  "
              f"interaction {eta.get('C(identity):C(opinion)', 0):5.1f}  "
              f"resid {eta.get('Residual', 0):5.1f}")
    summary["anova_2x2"] = anova_res

    # ---- consistency / conflict (existing identity+opinion) ----
    print("\n=== Consistency / conflict (identity+opinion), mean |Delta| ===")
    cc = {}
    for model in MODELS:
        sy = load_csvs(model, "sycophancy")
        sy["delta"] = sy["overall_score"] - sy.apply(
            lambda r: anchor[model][(r["field"], r["id"])], axis=1)

        def group(row):
            c = str(row["character"]).strip().lower()
            s = row["side"]
            if c == "stressed sideliners":
                return "neutral"
            left = c in LEFT_PERSONAS
            aligned = (left and s == "A") or ((not left) and s == "B")
            return "consistent" if aligned else "conflicted"

        sy["grp"] = sy.apply(group, axis=1)
        cc[model] = {g: sy[sy.grp == g]["delta"].abs().mean()
                     for g in ["consistent", "conflicted", "neutral"]}
        print(f"  {MODEL_SHORT[model]:<18} "
              f"consistent {cc[model]['consistent']:.2f}  "
              f"conflicted {cc[model]['conflicted']:.2f}  "
              f"neutral {cc[model]['neutral']:.2f}")
    summary["consistency_conflict"] = cc

    # ---- bias-flag rates by condition ----
    print("\n=== Bias-flag activation rate (%) by condition (pooled over models) ===")
    flag_rates = {}
    for cond, loader in [("anchor", "baseline"), ("identity", "stereotype"),
                         ("identity+opinion", "sycophancy")]:
        allm = pd.concat([load_csvs(m, loader) for m in MODELS], ignore_index=True)
        flag_rates[cond] = {f: 100 * allm[f].mean() for f in FLAGS}
    op_all = pd.concat([op_eval[m] for m in MODELS], ignore_index=True)
    flag_rates["opinion"] = {f: 100 * op_all[f].mean() for f in FLAGS}
    for cond in ["anchor", "identity", "opinion", "identity+opinion"]:
        print(f"  {cond:<18} " + "  ".join(
            f"{f.split('_')[0]} {flag_rates[cond][f]:5.1f}" for f in FLAGS))
    summary["flag_rates"] = flag_rates

    # ---- opinion-only refusal / sycophantic alignment ----
    print("\n=== Opinion-only refusal & sycophantic-alignment rate (%) ===")
    rr = {}
    for model in MODELS:
        df = op_eval[model]
        rr[model] = {"refusal": 100 * df["refusal"].mean(),
                     "syco_align": 100 * df["sycophantic_alignment"].mean()}
        print(f"  {MODEL_SHORT[model]:<18} refusal {rr[model]['refusal']:.2f}   "
              f"syco-align {rr[model]['syco_align']:.1f}")
    summary["opinion_only_refusal_syco"] = rr

    # ---- MWU: opinion-only, each flag vs |shift| ----
    print("\n=== Opinion-only Mann-Whitney U: flag vs |shift| (pooled) ===")
    mwu = {}
    rows = []
    for model in MODELS:
        df = op_eval[model].copy()
        df["absdelta"] = [abs(r["overall_score"]
                              - anchor[model][(r["field"], r["id"])])
                          for _, r in df.iterrows()]
        rows.append(df)
    pooled = pd.concat(rows, ignore_index=True)
    for f in FLAGS + ["sycophantic_alignment"]:
        on = pooled[pooled[f] == True]["absdelta"]
        off = pooled[pooled[f] == False]["absdelta"]
        if len(on) and len(off):
            u, p = mannwhitneyu(on, off, alternative="two-sided")
            mwu[f] = {"p": p, "sig": bool(p < 0.05)}
            print(f"  {f:<22} p={p:.2e}  {'Sig.' if p < 0.05 else 'n.s.'}")
    summary["opinion_only_mwu"] = mwu

    # ---- figures ----
    os.makedirs(PAPER_FIG, exist_ok=True)
    # heatmap: models x {left, right} mean delta
    mat = np.array([[op_shift[m]["A"], op_shift[m]["B"]] for m in MODELS])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    vmax = np.abs(mat).max()
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks([0, 1], ["Left narrative", "Right narrative"])
    ax.set_yticks(range(len(MODELS)), [MODEL_SHORT[m] for m in MODELS])
    for i in range(len(MODELS)):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(mat[i, j]) > vmax * 0.6 else "black")
    ax.set_title("Opinion-only stance shift ($\\Delta$)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(PAPER_FIG, "heatmap_opinion_only.png"), dpi=200)
    plt.close(fig)

    # combined hero figure: identity-only heatmap + dissociation scatter.
    dissociation_overview(id_delta, dir_susc)
    print(f"\nFigures written to {PAPER_FIG}")

    with open(os.path.join(REPO_ROOT, "analysis_opinion_only_summary.json"),
              "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("Summary written to analysis_opinion_only_summary.json")


if __name__ == "__main__":
    main()
