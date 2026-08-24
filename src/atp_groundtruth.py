"""Phase A / Task A3 -- per-dilemma human ground-truth distributions.

Writes ``data/atp/groundtruth.json``: for every dilemma in ``dilemmas.json``,
the real human response distribution over its two poles (r_L / r_R), in TWO
layers (team-lead spec, 2026-05-19):

  layer 1 -- ``party``        Democrat vs Republican, taken from the Törnberg
                              partisan extract (already carried per dilemma in
                              ``dilemmas.json`` ``meta.dist_dem`` / ``dist_rep``).
  layer 2 -- ``demographic``  response distribution by demographic subgroup
                              (region / age / sex / education / race /
                              religion / religious-attendance / income /
                              ideology / party), computed from OpinionQA
                              (Santurkar et al. 2023) respondent-level human
                              responses.

D2 (stereotyping-accuracy) chooses the granularity later; ``TAG_SUBGROUP_MAP``
proposes, per identity tag, the single demographic subgroup that tag most
foregrounds (design doc section 6) -- provisional, mcc/team-lead may revise.
The full per-attribute breakdown is emitted regardless, so D2 is not bound to
that mapping.

Data sources -- both open, NO Pew account, NO gated terms-of-use accepted:
  * layer 1: ``data/atp/raw/pew_atp_partisan_distributions.csv`` (Törnberg).
  * layer 2: OpinionQA ``human_resp`` respondent-level files, fetched from the
    public CodaLab bundle ``0x050b7e72abb04d1f9b493c1743e580cf`` into
    ``data/atp/raw/opinionqa/`` (cached; re-used if already present).

Note: the SubPOP HuggingFace dataset (the other layer-2 candidate) is
``gated:manual`` -- it needs an access request + terms agreement + owner
approval, so it is NOT used here. OpinionQA's CodaLab release is ungated.

Run:  python atp_groundtruth.py        (downloads ~140 MB on first run)
"""

import csv
import json
import os
import re
import sys
import urllib.request

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
DILEMMAS_PATH = os.path.join(REPO_ROOT, "data", "atp", "dilemmas.json")
OUT_PATH = os.path.join(REPO_ROOT, "data", "atp", "groundtruth.json")
OQA_CACHE = os.path.join(REPO_ROOT, "data", "atp", "raw", "opinionqa")

# OpinionQA human_resp CodaLab bundle (open; see opinions-qa-dataset worksheet).
OQA_BUNDLE = "0x050b7e72abb04d1f9b493c1743e580cf"
OQA_BLOB = ("https://worksheets.codalab.org/rest/bundles/"
            f"{OQA_BUNDLE}/contents/blob/human_resp")

# Demographic columns to break the response distribution down by. POLPARTY is
# OpinionQA's wave-specific party variable -- kept here as a layer-2 attribute;
# the headline party layer (layer 1) comes from Törnberg instead.
DEMOGRAPHIC_ATTRS = [
    "CREGION", "AGE", "SEX", "EDUCATION", "RACE", "RELIG", "RELIGATTEND",
    "INCOME", "POLIDEOLOGY", "POLPARTY",
]
csv.field_size_limit(10 ** 7)

# Per identity tag, the single demographic subgroup it most foregrounds
# (design doc section 6). PROVISIONAL -- D2/mcc may revise; the full per-
# attribute breakdown is emitted regardless so this mapping is not binding.
# OpinionQA has no urban/rural or "evangelical" variable, so those §6 cues
# fall back to the closest available subgroup (noted in ``via``).
TAG_SUBGROUP_MAP = {
    "progressive_left": {"attribute": "RELIG", "group": "Nothing in particular",
                         "via": "§6 'no religious affiliation' (also young/urban)"},
    "establishment_liberals": {"attribute": "EDUCATION",
                               "group": "College graduate/some postgrad",
                               "via": "§6 'college-educated professional'"},
    "democratic_mainstays": {"attribute": "RACE", "group": "Black",
                             "via": "§6 'older Black woman, attends church'"},
    "outsider_left": {"attribute": "AGE", "group": "18-29",
                      "via": "§6 'college student in their early 20s'"},
    "ambivalent_right": {"attribute": "AGE", "group": "30-49",
                         "via": "§6 'younger White person in their 30s'"},
    "populist_right": {"attribute": "EDUCATION", "group": "High school graduate",
                       "via": "§6 'working-class, high-school education'"},
    "committed_conservatives": {"attribute": "INCOME", "group": "$100,000 or more",
                                "via": "§6 'well-off, college-educated'"},
    "faith_and_flag_conservatives": {"attribute": "RELIGATTEND",
                                     "group": "More than once a week",
                                     "via": "§6 'evangelical' (no evangelical "
                                            "variable; using high attendance)"},
}

_SKIP_RESPONSE = {"", "refused", "no answer", "not asked"}


def _norm(text):
    """Normalise option / response text for matching across data sources.

    Törnberg and OpinionQA differ in curly vs straight quotes and "U.S." vs
    "US"; normalise punctuation and case so the pole texts join.
    """
    text = text or ""
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'),
                 ("”", '"'), ("—", "-"), ("–", "-")):
        text = text.replace(a, b)
    text = text.replace(".", "")
    return re.sub(r"\s+", " ", text.strip()).casefold()


def download_opinionqa(waves):
    """Fetch each wave's responses.csv from CodaLab into the cache (if absent).

    responses.csv stores response *labels* and demographic *labels* directly,
    so info.csv / metadata.csv are not needed.
    """
    os.makedirs(OQA_CACHE, exist_ok=True)
    for wave in waves:
        dest = os.path.join(OQA_CACHE, f"{wave}_responses.csv")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        url = f"{OQA_BLOB}/American_Trends_Panel_{wave}/responses.csv"
        print(f"  downloading OpinionQA {wave} ...", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            with open(dest, "wb") as f:
                f.write(data)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: {wave} download failed: {exc}")


def load_wave(wave):
    """Return (rows, weight_col) for a wave, or (None, None) if unavailable."""
    path = os.path.join(OQA_CACHE, f"{wave}_responses.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None, None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    cols = rows[0].keys()
    weight_col = None
    for cand in (f"WEIGHT_{wave}", f"WEIGHT_{wave}_ATP"):
        if cand in cols:
            weight_col = cand
            break
    if weight_col is None:
        weight_col = next((c for c in cols if c.startswith("WEIGHT")), None)
    return rows, weight_col


def _weight(row, weight_col):
    if not weight_col:
        return 1.0
    try:
        return float(row.get(weight_col) or 0.0)
    except ValueError:
        return 0.0


def pole_distribution(rows, qcol, weight_col, norm_L, norm_R, demo_col=None):
    """Weighted {r_L, r_R} distribution; grouped by ``demo_col`` if given.

    Returns either a single dict (demo_col=None) or {group: dict}. Each dict is
    ``{"r_L": p, "r_R": p, "n": unweighted_count}`` or ``None`` for an empty
    cell. ``unmatched`` counts non-refused answers matching neither pole.
    """
    acc = {}  # group -> [w_L, w_R, n, n_unmatched]

    def bucket(g):
        return acc.setdefault(g, [0.0, 0.0, 0, 0])

    for row in rows:
        ans = _norm(row.get(qcol))
        if ans in _SKIP_RESPONSE:
            continue
        group = "__all__" if demo_col is None else (row.get(demo_col) or "")
        if demo_col is not None and group.strip().casefold() in _SKIP_RESPONSE:
            continue
        cell = bucket(group)
        w = _weight(row, weight_col)
        if ans == norm_L:
            cell[0] += w
            cell[2] += 1
        elif ans == norm_R:
            cell[1] += w
            cell[2] += 1
        else:
            cell[3] += 1

    def finalise(cell):
        w_L, w_R, n, _ = cell
        total = w_L + w_R
        if total <= 0:
            return None
        return {"r_L": round(w_L / total, 4),
                "r_R": round(w_R / total, 4), "n": n}

    if demo_col is None:
        cell = acc.get("__all__")
        return (finalise(cell) if cell else None,
                cell[3] if cell else 0)
    return {g: finalise(c) for g, c in acc.items() if finalise(c)}


def build():
    with open(DILEMMAS_PATH) as f:
        dilemmas = json.load(f)
    waves = sorted({d["meta"]["wave"] for d in dilemmas},
                   key=lambda w: int(w[1:]))

    print(f"OpinionQA cache: {OQA_CACHE}")
    download_opinionqa(waves)

    wave_cache = {}  # wave -> (rows, weight_col)
    for wave in waves:
        wave_cache[wave] = load_wave(wave)

    out = []
    joined = 0
    unjoined_ids = []
    unmatched_ids = []

    for d in dilemmas:
        m = d["meta"]
        wave, var_name = m["wave"], m["var_name"]
        rec = {
            "id": d["id"],
            "var_name": var_name,
            "wave": wave,
            "r_L": d["r_L"],
            "r_R": d["r_R"],
            # layer 1 -- party (Törnberg; present for every dilemma).
            "party": {
                "democrat": {"r_L": m["dist_dem"]["r_L"],
                             "r_R": m["dist_dem"]["r_R"], "n": m.get("n_dem")},
                "republican": {"r_L": m["dist_rep"]["r_L"],
                               "r_R": m["dist_rep"]["r_R"], "n": m.get("n_rep")},
            },
            "demographic": {"joined": False},
        }

        rows, weight_col = wave_cache.get(wave, (None, None))
        qcol = f"{var_name}_{wave}"
        if rows is not None and qcol in rows[0]:
            norm_L, norm_R = _norm(d["r_L"]), _norm(d["r_R"])
            overall, n_unmatched = pole_distribution(
                rows, qcol, weight_col, norm_L, norm_R)
            if overall is None:
                # item column present but neither pole text matched OpinionQA's
                unmatched_ids.append(d["id"])
                unjoined_ids.append(d["id"])
            else:
                subgroups = {}
                for attr in DEMOGRAPHIC_ATTRS:
                    if attr in rows[0]:
                        dist = pole_distribution(
                            rows, qcol, weight_col, norm_L, norm_R,
                            demo_col=attr)
                        if dist:
                            subgroups[attr] = dist
                rec["demographic"] = {
                    "joined": True,
                    "opinionqa_qcol": qcol,
                    "weighted_by": weight_col,
                    "overall": overall,
                    "n_unmatched_responses": n_unmatched,
                    "subgroups": subgroups,
                }
                joined += 1
        else:
            unjoined_ids.append(d["id"])

        out.append(rec)

    doc = {
        "_meta": {
            "description": "Phase A / A3 -- human ground-truth response "
                           "distributions per ATP dilemma, two layers.",
            "n_dilemmas": len(out),
            "layer1_party": {
                "source": "data/atp/raw/pew_atp_partisan_distributions.csv "
                          "(Törnberg & Schimmel ATP extract)",
                "groups": ["democrat", "republican"],
            },
            "layer2_demographic": {
                "source": "OpinionQA / Santurkar et al. 2023 human_resp, "
                          f"CodaLab bundle {OQA_BUNDLE}",
                "weighted_by": "WEIGHT_W{wave} survey weight",
                "attributes": DEMOGRAPHIC_ATTRS,
            },
            "tag_subgroup_map": TAG_SUBGROUP_MAP,
            "join_coverage": {
                "layer1_party": len(out),
                "layer2_demographic_joined": joined,
                "layer2_demographic_unjoined": len(out) - joined,
                "unjoined_ids": unjoined_ids,
                "joined_but_pole_text_unmatched_ids": unmatched_ids,
            },
        },
        "dilemmas": out,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    return doc


def main():
    doc = build()
    cov = doc["_meta"]["join_coverage"]
    n = doc["_meta"]["n_dilemmas"]
    print("=" * 70)
    print(f"atp_groundtruth -- wrote {n} dilemmas -> {OUT_PATH}")
    print("=" * 70)
    print(f"\nlayer 1 (party):        {cov['layer1_party']}/{n} dilemmas")
    print(f"layer 2 (demographic):  {cov['layer2_demographic_joined']}/{n} "
          f"dilemmas joined to OpinionQA")
    print(f"  unjoined:             {cov['layer2_demographic_unjoined']} "
          f"(item not in OpinionQA release, or pole text unmatched)")
    if cov["joined_but_pole_text_unmatched_ids"]:
        print(f"  pole-text unmatched:  "
              f"{cov['joined_but_pole_text_unmatched_ids']}")

    # Verification: every dilemma has a party layer summing to ~1.
    bad = 0
    for rec in doc["dilemmas"]:
        for grp in ("democrat", "republican"):
            p = rec["party"][grp]
            if abs(p["r_L"] + p["r_R"] - 1) > 0.02:
                bad += 1
    print(f"\nverify: party distributions sum~1 -> "
          f"{'OK' if bad == 0 else f'{bad} BAD'}")

    # Spot-check: demographic spread on the first joined dilemma.
    for rec in doc["dilemmas"]:
        if rec["demographic"].get("joined"):
            dg = rec["demographic"]
            print(f"\nspot-check joined dilemma #{rec['id']} "
                  f"({rec['wave']}/{rec['var_name']}):")
            print(f"  overall r_R={dg['overall']['r_R']} "
                  f"(n={dg['overall']['n']})")
            for attr in ("RACE", "RELIGATTEND", "POLIDEOLOGY"):
                if attr in dg["subgroups"]:
                    parts = ", ".join(
                        f"{g}:{v['r_R']}" for g, v in
                        dg["subgroups"][attr].items())
                    print(f"  {attr} r_R -> {parts}")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
