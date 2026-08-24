"""Regenerate the paper figures with the revised terminology.

Adapted from notebooks/final_plotting.ipynb. Only the labels change ---
the data and layout are the notebook's. Terminology updates:
  Arm 0 / Baseline      -> Anchor only
  Arm 1 / Stereotyping  -> Identity only
  Arm 2 / Sycophancy    -> Identity + opinion
  Setting 1 / 2         -> no role-play / role-play
  ANOVA factors         -> Model anchor / User identity / User opinion

Writes straight into ../latex/figures/. Run from anywhere.
"""

import csv
import os
from math import pi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
PAPER_FIG = os.path.join(os.path.dirname(REPO_ROOT), "latex", "figures")

from subject_models import (FRIENDLY_NAMES as MODELS,
                            MODEL_SHORT as MODEL_SHORT,
                            ROLEPLAY_MODELS)

FIELDS = ["diplomatic", "economy", "energy", "technology", "welfare"]
PERSONA_ORDER = ["progressive left", "outsider left", "establishment liberals",
                 "democratic mainstays", "stressed sideliners",
                 "ambivalent right", "populist right",
                 "committed conservatives", "faith and flag conservatives"]
# Short codes for the nine Pew personas (left-to-right spectrum order),
# used on heatmap axes; the full key lives in the figure captions.
PERSONA_CODES = ["PL", "OL", "EL", "DM", "SS", "AR", "PR", "CC", "FFC"]
ROLEPLAY_PERSONAS = ["outsider left", "stressed sideliners",
                     "faith and flag conservatives"]
MODEL_COLOR = {"deepseek-v3.1": "#1f77b4", "o4-mini": "#d62728",
               "llama-3.3-70b-instruct": "#2ca02c", "qwen3-32b": "#ffd30e",
               "mistral-small-3.2-24b": "#9467bd",
               "gemma-4-31b-it": "#ff7f0e", "phi-4": "#8c564b",
               "nova-lite-v1": "#e377c2", "granite-4.1-8b": "#17becf",
               "glm-4.7": "#bcbd22", "kimi-k2.5": "#aec7e8",
               "mimo-v2-flash": "#98df8a", "gpt-oss-120b": "#8c6d31"}


def _b(v):
    return str(v).strip().lower() == "true"


def load_frames(results_root):
    """Load baseline / stereotype / opinion_only / sycophancy eval CSVs."""
    base, ster, syco, oonly = [], [], [], []
    root = os.path.join(REPO_ROOT, results_root)
    is_roleplay = "as_characters" in results_root
    model_list = ROLEPLAY_MODELS if is_roleplay else MODELS
    for model in model_list:
        for field in FIELDS:
            bf = f"{root}/{model}/baseline_global_{field}.csv"
            if not os.path.exists(bf):
                continue
            with open(bf) as f:
                for r in csv.DictReader(f):
                    base.append(dict(
                        model=model, field=field, score=float(r["overall_score"]),
                        structural_bias=_b(r.get("structural_bias", "")),
                        framing_bias=_b(r.get("framing_bias", "")),
                        selection_bias=_b(r.get("selection_bias", "")),
                        normative_bias=_b(r.get("normative_bias", "")),
                        was_responded_as=r.get("was_responded_as", "")))
            sf = f"{root}/{model}/stereotype_global_{field}.csv"
            if not os.path.exists(sf):
                continue
            with open(sf) as f:
                for r in csv.DictReader(f):
                    ster.append(dict(
                        model=model, field=field, score=float(r["overall_score"]),
                        character=r["character"],
                        structural_bias=_b(r.get("structural_bias", "")),
                        framing_bias=_b(r.get("framing_bias", "")),
                        selection_bias=_b(r.get("selection_bias", "")),
                        normative_bias=_b(r.get("normative_bias", "")),
                        was_responded_as=r.get("was_responded_as", "")))
            yf = f"{root}/{model}/sycophancy_global_{field}.csv"
            if not os.path.exists(yf):
                continue
            with open(yf) as f:
                for r in csv.DictReader(f):
                    syco.append(dict(
                        model=model, field=field, score=float(r["overall_score"]),
                        character=r["character"], side=r["side"],
                        structural_bias=_b(r.get("structural_bias", "")),
                        framing_bias=_b(r.get("framing_bias", "")),
                        selection_bias=_b(r.get("selection_bias", "")),
                        normative_bias=_b(r.get("normative_bias", "")),
                        was_responded_as=r.get("was_responded_as", "")))
            if not is_roleplay:
                oof = f"{root}/{model}/opinion_only_global_{field}.csv"
                if os.path.exists(oof):
                    with open(oof) as f:
                        for r in csv.DictReader(f):
                            try:
                                score = float(r["overall_score"])
                            except (ValueError, TypeError):
                                continue
                            oonly.append(dict(
                                model=model, field=field, score=score,
                                side=r["side"],
                                structural_bias=_b(r.get("structural_bias", "")),
                                framing_bias=_b(r.get("framing_bias", "")),
                                selection_bias=_b(r.get("selection_bias", "")),
                                normative_bias=_b(r.get("normative_bias", ""))))
    b = pd.DataFrame(base)
    b["character"] = "baseline"
    return b, pd.DataFrame(ster), pd.DataFrame(syco), pd.DataFrame(oonly)


def fig_path(name):
    return os.path.join(PAPER_FIG, name)


# ----------------------------------------------------------------------
# Heatmaps -- no role-play (notebook cell 10)
# ----------------------------------------------------------------------
def heatmaps_no_roleplay(baseline_df, stereotype_df, sycophancy_df):
    baseline_mean = baseline_df.groupby(["model", "field"])["score"].mean().reset_index()
    baseline_mean.rename(columns={"score": "baseline_score"}, inplace=True)

    stereo = pd.merge(stereotype_df, baseline_mean, on=["model", "field"])
    stereo["delta_score"] = stereo["score"] - stereo["baseline_score"]
    stereo_pivot = (stereo.groupby(["model", "character"])["delta_score"].mean()
                    .reset_index()
                    .pivot(index="model", columns="character", values="delta_score")
                    .reindex(columns=PERSONA_ORDER))

    plt.figure(figsize=(12, 5))
    vmax = max(abs(stereo_pivot.min().min()), abs(stereo_pivot.max().max()))
    sns.heatmap(stereo_pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-vmax, vmax=vmax,
                cbar_kws={"label": "$\\Delta$ Score (shift from anchor-only baseline)"})
    plt.title("Identity Only (no role-play): stance shift by user identity, "
              "averaged over 5 domains", fontsize=14, pad=15)
    plt.xlabel("Disclosed user identity (left-to-right spectrum)", fontsize=12)
    plt.ylabel("Model", fontsize=12)
    plt.gca().set_xticklabels(PERSONA_CODES, rotation=0)
    plt.tight_layout()
    plt.savefig(fig_path("heatmap_stereotype_arm1.png"))
    plt.close()

    syco = pd.merge(sycophancy_df, baseline_mean, on=["model", "field"])
    syco["delta_score"] = syco["score"] - syco["baseline_score"]
    syco_summary = syco.groupby(["model", "character", "side"])["delta_score"].mean().reset_index()

    # mcc: 2026-05-24 — stacked vertically (was 1×2 figsize=(20,6) and
    # mcc: heavily squished when scaled to \textwidth in the 2-column
    # mcc: layout). Now 2×1, each panel takes the full figure width so
    # mcc: cell digits stay readable after page scaling.
    # fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    vmax = max(abs(syco_summary["delta_score"].min()), abs(syco_summary["delta_score"].max()))
    titles = ["Identity + Opinion: left-leaning narrative",
              "Identity + Opinion: right-leaning narrative"]
    for i, side in enumerate(["A", "B"]):
        pivot = (syco_summary[syco_summary["side"] == side]
                 .pivot(index="model", columns="character", values="delta_score")
                 .reindex(columns=PERSONA_ORDER))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                    vmin=-vmax, vmax=vmax, ax=axes[i],
                    cbar_kws={"label": "$\\Delta$ Score"} if i == 1 else {"drawedges": False})
        axes[i].set_title(titles[i], fontsize=14)
        # xlabel only on bottom panel (shared x-axis)
        axes[i].set_xlabel("Disclosed user identity" if i == 1 else "", fontsize=12)
        axes[i].set_xticklabels(PERSONA_CODES, rotation=0)
        axes[i].set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(fig_path("heatmap_sycophancy_arm2.png"))
    plt.close()

    # Single right-leaning panel, sized for one text column (Section 4.2);
    # the full left+right pair stays in the appendix.
    pivot_b = (syco_summary[syco_summary["side"] == "B"]
               .pivot(index="model", columns="character", values="delta_score")
               .reindex(columns=PERSONA_ORDER))
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    sns.heatmap(pivot_b, annot=True, fmt="+.0f", annot_kws={"fontsize": 6},
                cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, ax=ax,
                linewidths=0.4, linecolor="white", cbar_kws={"label": ""})
    ax.set_xlabel("Disclosed user identity", fontsize=7.5)
    ax.set_ylabel("")
    ax.set_xticklabels(PERSONA_CODES, rotation=0, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
    ax.tick_params(length=0)
    cb = ax.collections[0].colorbar
    cb.ax.tick_params(labelsize=6)
    cb.set_label("$\\Delta$ score", fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_path("heatmap_syco_arm2_right.png"), dpi=220,
                bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
# Heatmaps -- role-play (notebook cells 11, 12)
# ----------------------------------------------------------------------
def _roleplay_heatmap(merged, suptitle, out_name, side=None):
    # Appendix figure -- vertical space is free. A 2x3 grid (one empty
    # cell) gives each of the five model sub-plots room for legible cell
    # numbers and axis labels, instead of cramming them into one row.
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    vmax = 10.0
    for i, model in enumerate(ROLEPLAY_MODELS):
        ax = axes[i]
        md = merged[merged["model"] == model]
        if side is not None:
            md = md[md["side"] == side]
        pivot = (md.pivot_table(index="was_responded_as", columns="character",
                                values="delta_score", aggfunc="mean")
                 .reindex(index=ROLEPLAY_PERSONAS, columns=ROLEPLAY_PERSONAS))
        sns.heatmap(pivot, annot=True, fmt=".2f", annot_kws={"fontsize": 13},
                    cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax, ax=ax,
                    cbar=True, cbar_kws={"label": "$\\Delta$ Score (shift)"},
                    square=True, linewidths=1, linecolor="white")
        ax.set_title(f"{MODEL_SHORT[model]}", fontsize=16, pad=10)
        ax.set_xlabel("User identity", fontsize=14)
        if i % 3 == 0:
            ax.set_ylabel("Model anchor (role-play)", fontsize=14)
            ax.set_yticklabels(["Left\n(Outsider)", "Center\n(Sideliners)",
                                "Right\n(Faith & Flag)"], rotation=0,
                               va="center", fontsize=12)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels(["Left\n(Outsider)", "Center\n(Sideliners)",
                                "Right\n(Faith & Flag)"], rotation=0,
                               va="center", fontsize=12)
        ax.set_xticklabels(["Left", "Neutral", "Right"], rotation=0,
                           fontsize=12)
    for j in range(len(ROLEPLAY_MODELS), len(axes)):
        axes[j].axis("off")
    plt.suptitle(suptitle, fontsize=19, y=1.0)
    plt.tight_layout()
    plt.savefig(fig_path(out_name), bbox_inches="tight")
    plt.close()


def heatmaps_roleplay(baseline_as, stereotype_as, sycophancy_as):
    base_mean = (baseline_as.groupby(["model", "was_responded_as"])["score"].mean()
                 .reset_index().rename(columns={"score": "score_baseline_roleplay"}))

    ster = pd.merge(stereotype_as, base_mean, on=["model", "was_responded_as"], how="left")
    ster["delta_score"] = ster["score"] - ster["score_baseline_roleplay"]
    _roleplay_heatmap(
        ster,
        "Identity Only, role-play: model anchor vs. user identity",
        "heatmap_stereotype_roleplay.png")

    syco = pd.merge(sycophancy_as, base_mean, on=["model", "was_responded_as"], how="left")
    syco["delta_score"] = syco["score"] - syco["score_baseline_roleplay"]
    for side, lr in [("A", "left"), ("B", "right")]:
        _roleplay_heatmap(
            syco,
            f"Identity + Opinion, role-play: model anchor vs. user identity "
            f"(narrative {lr}-leaning)",
            f"heatmap_sycophancy_roleplay_{lr}.png", side=side)


# ----------------------------------------------------------------------
# Bias radar (notebook cells 40, 43)
# ----------------------------------------------------------------------
CATEGORIES = ["Structural", "Framing", "Selection", "Normative"]
BIAS4 = ["structural_bias", "framing_bias", "selection_bias", "normative_bias"]
COND_LABELS = ["Anchor only", "Identity only", "Opinion only", "Identity + opinion"]
COND_COLORS = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]


def _rates(df):
    return [df[c].mean() * 100 if len(df) else 0.0 for c in BIAS4]


def radar_overview(baseline_df, stereotype_df, sycophancy_df):
    angles = [n / 5.0 * 2 * pi for n in range(5)]
    angles += angles[:1]
    # Compact figsize + larger fonts so the labels survive layout scaling.
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.6), subplot_kw=dict(polar=True))
    for i, model in enumerate(MODELS):
        ax = axes[i]
        rates = [_rates(baseline_df[baseline_df["model"] == model]),
                 _rates(stereotype_df[stereotype_df["model"] == model]),
                 _rates(sycophancy_df[sycophancy_df["model"] == model])]
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.sca(ax)
        plt.xticks(angles[:-1], CATEGORIES, color="grey", size=13)
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80], ["20%", "40%", "60%", "80%"], color="grey", size=9)
        plt.ylim(0, 100)
        styles = ["-.", "--", "-"]
        widths = [1.5, 2, 2.5]
        for j, (label, rt) in enumerate(zip(COND_LABELS, rates)):
            ax.plot(angles, rt, linewidth=widths[j], linestyle=styles[j],
                    color=COND_COLORS[j], label=label)
            if j > 0:
                ax.fill(angles, rt, color=COND_COLORS[j], alpha=0.12 + 0.13 * (j - 1))
        ax.set_title(f"{model}", size=15, fontweight="bold", pad=18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.13),
               ncol=3, frameon=False, fontsize=15)
    plt.tight_layout()
    plt.savefig(fig_path("radar_bias_overview.png"), dpi=300, bbox_inches="tight")
    plt.close()


def bias_flags_heatmap(baseline_df, stereotype_df, sycophancy_df, opinion_df):
    """Domain-pooled bias-flag heatmap: 13 rows x 16 columns
    (4 flags x 4 conditions: Anchor / Identity / Opinion / Identity+opinion)."""
    conds = [baseline_df, stereotype_df, opinion_df, sycophancy_df]
    n_conds = len(conds)
    mat = []
    for model in MODELS:
        row = []
        for df in conds:
            sub = df[df["model"] == model]
            row += [sub[c].mean() * 100 if len(sub) else 0.0
                    for c in BIAS4]
        mat.append(row)
    mat = np.array(mat)

    fig, ax = plt.subplots(figsize=(14.5, 3.4))
    sns.heatmap(mat, ax=ax, cmap="rocket_r", vmin=0, vmax=100,
                annot=True, fmt=".0f", annot_kws={"fontsize": 8},
                linewidths=0.5, linecolor="white",
                xticklabels=["Struct.", "Fram.", "Sel.", "Norm."] * n_conds,
                yticklabels=[MODEL_SHORT[m] for m in MODELS],
                cbar_kws={"label": "Flag activation rate (%)",
                          "shrink": 0.85, "pad": 0.015})
    ax.tick_params(length=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for i, name in enumerate(COND_LABELS):
        ax.text(i * 4 + 2, -0.35, name, ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    for x in (4, 8, 12):
        ax.axvline(x, color="black", lw=2)
    plt.tight_layout()
    plt.savefig(fig_path("bias_flags_bar.png"), dpi=200, bbox_inches="tight")
    plt.close()


def heatmap_bias_by_field(baseline_df, stereotype_df, sycophancy_df, opinion_df):
    """Per-(domain, model) heatmap of the four response-bias-flag
    activation rates (%) under 4 conditions (Anchor / Identity / Opinion /
    Identity+opinion).

    mcc: 2026-05-23 — was a single 65×16 heatmap that overcrowded cell digits
    even after bumping figsize. Split into two 65×8 heatmaps, one per
    condition pair (a: Anchor + Identity; b: Opinion + Identity+opinion).
    Each is wide enough that cell digits read cleanly.

    mcc: 2026-05-24 — the 65-row split versions still felt cramped because
    each domain block was visually buried inside one tall figure. Switch to
    one figure per domain: 13 models × (4 conditions × 4 flags = 16 cols),
    landscape. Each figure is small enough to read at \\textwidth and each
    domain is now a distinct artifact.
    """
    conds_all = [baseline_df, stereotype_df, opinion_df, sycophancy_df]

    for field in FIELDS:
        # Build the (13 × 16) matrix for this single field.
        mat, ylabels = [], []
        for model in MODELS:
            row = []
            for df in conds_all:
                sub = df[(df["model"] == model) & (df["field"] == field)]
                row += [sub[c].mean() * 100 if len(sub) else 0.0
                        for c in BIAS4]
            mat.append(row)
            ylabels.append(MODEL_SHORT[model])
        mat = np.array(mat)

        # 13 rows × 16 cols, landscape, sized to render comfortably at \textwidth.
        fig, ax = plt.subplots(figsize=(11.0, 5.5))
        sns.heatmap(mat, ax=ax, cmap="rocket_r", vmin=0, vmax=100,
                    annot=True, fmt=".0f", annot_kws={"fontsize": 8.5},
                    linewidths=0.5, linecolor="white",
                    xticklabels=["Struct.", "Fram.", "Sel.", "Norm."] * len(conds_all),
                    yticklabels=ylabels,
                    cbar_kws={"label": "Flag activation rate (%)",
                              "shrink": 0.8, "pad": 0.015})
        ax.tick_params(length=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right",
                           fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
        # condition labels above each 4-flag block (tight against axes)
        for i, name in enumerate(COND_LABELS):
            ax.text(i * 4 + 2, -0.25, name, ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        # vertical separators between condition blocks
        for x in (4, 8, 12):
            ax.axvline(x, color="black", lw=2)
        # domain title — pushed well above the condition labels to avoid collision
        ax.set_title(field.capitalize(), fontsize=14, fontweight="bold", pad=32)
        plt.tight_layout()
        plt.savefig(fig_path(f"heatmap_bias_{field}.png"), dpi=200,
                    bbox_inches="tight")
        plt.close()


# ----------------------------------------------------------------------
# Violins (notebook cells 32, 48, 49)
# ----------------------------------------------------------------------
def violin_density(baseline_df, sycophancy_df):
    from matplotlib.patches import Patch
    bmean = (baseline_df.groupby(["model", "field"])["score"].mean()
             .reset_index().rename(columns={"score": "bs"}))
    d = pd.merge(sycophancy_df, bmean, on=["model", "field"])
    d["delta_score"] = d["score"] - d["bs"]

    plt.figure(figsize=(6.4, 4.2))
    positions = np.arange(1, len(MODELS) + 1)
    data_a = [d[(d.model == m) & (d.side == "A")]["delta_score"].dropna().values
              for m in MODELS]
    data_b = [d[(d.model == m) & (d.side == "B")]["delta_score"].dropna().values
              for m in MODELS]
    v1 = plt.violinplot(data_a, positions=positions, widths=0.8, showmeans=True, vert=False)
    v2 = plt.violinplot(data_b, positions=positions, widths=0.8, showmeans=True, vert=False)

    def _style(v, color):
        for body in v["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.4)
            body.set_edgecolor(color)
            body.set_linewidth(1.2)
        for key in ["cbars", "cmins", "cmaxes", "cmeans"]:
            if key in v and v[key] is not None:
                v[key].set_color(color)
                v[key].set_linewidth(1.5)

    _style(v1, "#1f77b4")
    _style(v2, "#d62728")
    plt.yticks(positions, MODELS, fontsize=11)
    plt.title("Identity + Opinion (no role-play): stance-shift density",
              fontsize=12, pad=10)
    plt.xlabel("$\\Delta$ Score (shift from anchor-only baseline)", fontsize=12)
    plt.ylabel("Model", fontsize=12)
    plt.grid(axis="x", alpha=0.3)
    plt.axvline(x=0.0, color="gray", linestyle="--", linewidth=1.5, zorder=0)
    plt.legend(handles=[
        Patch(facecolor="#1f77b4", edgecolor="#1f77b4", alpha=0.4,
              label="Left-leaning narrative"),
        Patch(facecolor="#d62728", edgecolor="#d62728", alpha=0.4,
              label="Right-leaning narrative")],
        frameon=True, loc="upper right", title="Narrative stance")
    plt.tight_layout()
    plt.savefig(fig_path("violin_sycophancy_density.png"), dpi=200,
                bbox_inches="tight")
    plt.close()


def violin_roleplay_stereo(stereotype_df, stereotype_as):
    persona = "faith and flag conservatives"
    mean_orig = stereotype_df.groupby("model")["score"].mean()
    sub = stereotype_as[stereotype_as["was_responded_as"] == persona]
    plt.figure(figsize=(6.6, 4.6))
    sns.violinplot(data=sub, x="score", y="character", hue="model",
                   palette=MODEL_COLOR, orient="h", order=ROLEPLAY_PERSONAS,
                   inner="quartile", bw_adjust=0.5, cut=0, alpha=0.7)
    for m in MODELS:
        if m in mean_orig:
            plt.axvline(x=mean_orig[m], color=MODEL_COLOR[m], linestyle="--",
                        alpha=0.7, linewidth=1.3)
    plt.title("Identity Only, role-play: model anchored as\n"
              "Faith and Flag Conservatives", fontsize=13, pad=12)
    plt.axvline(x=0, color="black", linewidth=1.5, alpha=0.6)
    plt.xlim(-10.5, 10.5)
    plt.xlabel("Stance score ($-10$ left to $+10$ right)", fontsize=12)
    plt.ylabel("Disclosed user identity", fontsize=12)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
               title="Model (dashed = no-role-play mean)", frameon=False,
               fontsize=9, title_fontsize=9)
    plt.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_path("violin_overview_stereotype_faith_and_flag.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def violin_roleplay_syco(sycophancy_df, sycophancy_as):
    persona = "faith and flag conservatives"
    mean_a = sycophancy_df[sycophancy_df.side == "A"].groupby("model")["score"].mean()
    mean_b = sycophancy_df[sycophancy_df.side == "B"].groupby("model")["score"].mean()
    sub = sycophancy_as[sycophancy_as["was_responded_as"] == persona]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for j, side in enumerate(["A", "B"]):
        sd = sub[sub.side == side]
        sns.violinplot(data=sd, x="score", y="character", hue="model",
                       palette=MODEL_COLOR, orient="h", order=ROLEPLAY_PERSONAS,
                       inner="quartile", bw_adjust=0.5, cut=0, ax=axes[j], alpha=0.7)
        mo = mean_a if side == "A" else mean_b
        for m in MODELS:
            if m in mo.index:
                axes[j].axvline(x=mo[m], color=MODEL_COLOR[m], linestyle="--",
                                alpha=0.6, linewidth=1.3)
        axes[j].set_title(f"User narrative: {'left' if side == 'A' else 'right'}-leaning",
                          fontsize=13)
        axes[j].axvline(x=0, color="black", linewidth=1.5, alpha=0.5)
        axes[j].set_xlim(-10.5, 10.5)
        axes[j].set_xlabel("Stance score", fontsize=12)
        axes[j].set_ylabel("Disclosed user identity" if j == 0 else "", fontsize=12)
        axes[j].grid(axis="x", linestyle=":", alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    plt.suptitle("Identity + Opinion, role-play: model anchored as "
                 "Faith and Flag Conservatives", fontsize=15, y=1.02)
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, -0.03),
               title="Model (dashed line = no-role-play mean)", title_fontsize=10)
    plt.tight_layout(rect=[0, 0.07, 1, 0.96])
    plt.savefig(fig_path("violin_overview_sycophancy_faith_and_flag.png"),
                dpi=200, bbox_inches="tight")
    plt.close()


def main():
    print("loading no-role-play evals...")
    b, s, y, oo = load_frames("evaluations/consensus/csvs")
    print("loading role-play evals...")
    b_as, s_as, y_as, _ = load_frames("evaluations/consensus/csvs_as_characters")

    print("heatmaps (no role-play)...")
    heatmaps_no_roleplay(b, s, y)
    print("heatmaps (role-play)...")
    heatmaps_roleplay(b_as, s_as, y_as)
    print("bias-flag heatmap (overview, no role-play)...")
    bias_flags_heatmap(b, s, y, oo)
    print("bias heatmap (by field, no role-play)...")
    heatmap_bias_by_field(b, s, y, oo)
    print("violin (density, no role-play)...")
    violin_density(b, y)
    print("violins (role-play overview)...")
    violin_roleplay_stereo(s, s_as)
    violin_roleplay_syco(y, y_as)
    print(f"done -> {PAPER_FIG}")


if __name__ == "__main__":
    main()
