"""Generate CSVs from consensus JSONs for use by anova_figures.py and regenerate_figures.py.

Writes into evaluations/consensus/csvs/ and evaluations/consensus/csvs_as_characters/
so that figure scripts can be run with:
    SYCO_EVAL_ROOT=evaluations/consensus python src/anova_figures.py
    SYCO_EVAL_ROOT=evaluations/consensus python src/regenerate_figures.py
"""
import csv
import glob
import json
import os
import re

from subject_models import FRIENDLY_NAMES as ALL_MODELS, ROLEPLAY_MODELS

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
CONSENSUS = os.path.join(ROOT, "evaluations", "consensus")

FIELDS = ["diplomatic", "economy", "energy", "technology", "welfare"]
GROUPS = [
    "progressive left", "outsider left", "establishment liberals",
    "democratic mainstays", "stressed sideliners", "ambivalent right",
    "populist right", "committed conservatives", "faith and flag conservatives",
]
ROLEPLAY_ANCHORS = ["outsider left", "stressed sideliners", "faith and flag conservatives"]


def bool_to_str(v):
    if v is True:
        return "True"
    if v is False:
        return "False"
    return ""


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ── no-role-play track ─────────────────────────────────────────────────────

BASE_FIELDS = ["id", "overall_score", "structural_bias", "framing_bias",
               "selection_bias", "normative_bias", "was_responded_as"]
STER_FIELDS = ["id", "character", "overall_score", "structural_bias",
               "framing_bias", "selection_bias", "normative_bias", "was_responded_as"]
SYCO_FIELDS = ["id", "character", "side", "overall_score", "structural_bias",
               "framing_bias", "selection_bias", "normative_bias",
               "sycophantic_alignment", "was_responded_as"]
OO_FIELDS = ["id", "side", "overall_score", "structural_bias", "framing_bias",
             "selection_bias", "normative_bias"]

n_csvs = 0
for model in ALL_MODELS:
    out_dir = os.path.join(CONSENSUS, "csvs", model)
    for field in FIELDS:
        src_base = os.path.join(CONSENSUS, "default", model, f"baseline_global_{field}.json")
        src_ster = os.path.join(CONSENSUS, "default", model, f"stereotype_global_{field}.json")
        src_syco = os.path.join(CONSENSUS, "default", model, f"sycophancy_global_{field}.json")

        # baseline
        if os.path.exists(src_base):
            with open(src_base) as f:
                rows = []
                for item in json.load(f):
                    rows.append({
                        "id": int(item["id"]),
                        "overall_score": item.get("overall_stance_score", ""),
                        "structural_bias": bool_to_str(item.get("structural_bias")),
                        "framing_bias": bool_to_str(item.get("framing_bias")),
                        "selection_bias": bool_to_str(item.get("selection_bias")),
                        "normative_bias": bool_to_str(item.get("normative_bias")),
                        "was_responded_as": item.get("was_responded_as", ""),
                    })
            write_csv(os.path.join(out_dir, f"baseline_global_{field}.csv"), rows, BASE_FIELDS)
            n_csvs += 1

        # stereotype
        if os.path.exists(src_ster):
            with open(src_ster) as f:
                rows = []
                for item in json.load(f):
                    for g in GROUPS:
                        if g not in item:
                            continue
                        gi = item[g]
                        rows.append({
                            "id": int(item["id"]),
                            "character": g,
                            "overall_score": gi.get("overall_stance_score", ""),
                            "structural_bias": bool_to_str(gi.get("structural_bias")),
                            "framing_bias": bool_to_str(gi.get("framing_bias")),
                            "selection_bias": bool_to_str(gi.get("selection_bias")),
                            "normative_bias": bool_to_str(gi.get("normative_bias")),
                            "was_responded_as": gi.get("was_responded_as", ""),
                        })
            write_csv(os.path.join(out_dir, f"stereotype_global_{field}.csv"), rows, STER_FIELDS)
            n_csvs += 1

        # sycophancy
        if os.path.exists(src_syco):
            with open(src_syco) as f:
                rows = []
                for item in json.load(f):
                    for g in GROUPS:
                        if g not in item:
                            continue
                        for side in ["A", "B"]:
                            if side not in item[g]:
                                continue
                            si = item[g][side]
                            rows.append({
                                "id": int(item["id"]),
                                "character": g,
                                "side": side,
                                "overall_score": si.get("overall_stance_score", ""),
                                "structural_bias": bool_to_str(si.get("structural_bias")),
                                "framing_bias": bool_to_str(si.get("framing_bias")),
                                "selection_bias": bool_to_str(si.get("selection_bias")),
                                "normative_bias": bool_to_str(si.get("normative_bias")),
                                "sycophantic_alignment": bool_to_str(si.get("sycophantic_alignment")),
                                "was_responded_as": si.get("was_responded_as", ""),
                            })
            write_csv(os.path.join(out_dir, f"sycophancy_global_{field}.csv"), rows, SYCO_FIELDS)
            n_csvs += 1

# ── role-play (as_characters) track ───────────────────────────────────────

for model in ROLEPLAY_MODELS:
    out_dir = os.path.join(CONSENSUS, "csvs_as_characters", model)
    for field in FIELDS:
        # baseline: one file per anchor persona
        base_rows = []
        for anchor in ROLEPLAY_ANCHORS:
            anchor_slug = anchor.replace(" ", "_")
            src = os.path.join(CONSENSUS, "as_characters", model,
                               f"baseline_global_{field}_responded_as_{anchor_slug}.json")
            if os.path.exists(src):
                with open(src) as f:
                    for item in json.load(f):
                        base_rows.append({
                            "id": int(item["id"]),
                            "overall_score": item.get("overall_stance_score", ""),
                            "structural_bias": bool_to_str(item.get("structural_bias")),
                            "framing_bias": bool_to_str(item.get("framing_bias")),
                            "selection_bias": bool_to_str(item.get("selection_bias")),
                            "normative_bias": bool_to_str(item.get("normative_bias")),
                            "was_responded_as": item.get("was_responded_as", anchor),
                        })
        if base_rows:
            write_csv(os.path.join(out_dir, f"baseline_global_{field}.csv"),
                      base_rows, BASE_FIELDS)
            n_csvs += 1

        # stereotype: per anchor persona file
        ster_rows = []
        for anchor in ROLEPLAY_ANCHORS:
            anchor_slug = anchor.replace(" ", "_")
            src = os.path.join(CONSENSUS, "as_characters", model,
                               f"stereotype_global_{field}_responded_as_{anchor_slug}.json")
            if os.path.exists(src):
                with open(src) as f:
                    for item in json.load(f):
                        for g in GROUPS:
                            if g not in item:
                                continue
                            gi = item[g]
                            ster_rows.append({
                                "id": int(item["id"]),
                                "character": g,
                                "overall_score": gi.get("overall_stance_score", ""),
                                "structural_bias": bool_to_str(gi.get("structural_bias")),
                                "framing_bias": bool_to_str(gi.get("framing_bias")),
                                "selection_bias": bool_to_str(gi.get("selection_bias")),
                                "normative_bias": bool_to_str(gi.get("normative_bias")),
                                "was_responded_as": gi.get("was_responded_as", anchor),
                            })
        if ster_rows:
            write_csv(os.path.join(out_dir, f"stereotype_global_{field}.csv"),
                      ster_rows, STER_FIELDS)
            n_csvs += 1

        # sycophancy: per anchor persona file
        syco_rows = []
        for anchor in ROLEPLAY_ANCHORS:
            anchor_slug = anchor.replace(" ", "_")
            src = os.path.join(CONSENSUS, "as_characters", model,
                               f"sycophancy_global_{field}_responded_as_{anchor_slug}.json")
            if os.path.exists(src):
                with open(src) as f:
                    for item in json.load(f):
                        for g in GROUPS:
                            if g not in item:
                                continue
                            for side in ["A", "B"]:
                                if side not in item[g]:
                                    continue
                                si = item[g][side]
                                syco_rows.append({
                                    "id": int(item["id"]),
                                    "character": g,
                                    "side": side,
                                    "overall_score": si.get("overall_stance_score", ""),
                                    "structural_bias": bool_to_str(si.get("structural_bias")),
                                    "framing_bias": bool_to_str(si.get("framing_bias")),
                                    "selection_bias": bool_to_str(si.get("selection_bias")),
                                    "normative_bias": bool_to_str(si.get("normative_bias")),
                                    "sycophantic_alignment": bool_to_str(si.get("sycophantic_alignment")),
                                    "was_responded_as": si.get("was_responded_as", anchor),
                                })
        if syco_rows:
            write_csv(os.path.join(out_dir, f"sycophancy_global_{field}.csv"),
                      syco_rows, SYCO_FIELDS)
            n_csvs += 1

# ── opinion-only track ────────────────────────────────────────────────────

OO_DIR = os.path.join(CONSENSUS, "opinion_only")

for model in ALL_MODELS:
    out_dir = os.path.join(CONSENSUS, "csvs", model)
    for field in FIELDS:
        src_oo = os.path.join(OO_DIR, model, f"opinion_only_eval_{field}.json")
        if not os.path.exists(src_oo):
            continue
        with open(src_oo) as f:
            rows = []
            for item in json.load(f):
                for side in ["A", "B"]:
                    if side not in item:
                        continue
                    si = item[side]
                    if si.get("overall_stance_score") is None:
                        continue
                    rows.append({
                        "id": int(float(item["id"])),
                        "side": side,
                        "overall_score": si["overall_stance_score"],
                        "structural_bias": bool_to_str(si.get("structural_bias")),
                        "framing_bias": bool_to_str(si.get("framing_bias")),
                        "selection_bias": bool_to_str(si.get("selection_bias")),
                        "normative_bias": bool_to_str(si.get("normative_bias")),
                    })
        if rows:
            write_csv(os.path.join(out_dir, f"opinion_only_global_{field}.csv"),
                      rows, OO_FIELDS)
            n_csvs += 1

print(f"wrote {n_csvs} CSV files -> evaluations/consensus/csvs[_as_characters]/")
