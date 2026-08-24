"""Phase A / Task A2 -- extract ATP dilemmas into ``data/atp/dilemmas.json``.

Method specification: ATP redesign
sections 1 and 6. Each dilemma is a real Pew American Trends Panel (ATP)
**two-statement forced-choice** item turned into the contract consumed by
``atp_probe.py``::

    {"id", "domain", "q", "r_L", "r_R", "meta"}

There is **zero LLM synthesis**: ``r_L``/``r_R`` are the verbatim ATP response
statements and ``q`` is a deterministic assembly of the verbatim ATP question
stem plus those two statements (this is the whole point of the redesign --
resolving reviewer concern #2 that the old stimuli were "home-made").

Source data
-----------
The openly-released Törnberg & Schimmel ATP extract, vendored under
``data/atp/raw/`` (see ``data/atp/raw/SOURCE.md``); no Pew account is used.

  - ``pew_atp_items.csv``                 item text + response options
  - ``pew_atp_partisan_distributions.csv`` per-item Dem/Rep response split

Format filter (design doc section 6: "KEEP two-statement forced-choice;
DROP single-statement agree-disagree + non-ordinal")
----------------------------------------------------
An item becomes a dilemma iff ALL hold:

  * ``n_options == 2``                       -- exactly two poles
  * both options are statement-like          -- each >= MIN_OPTION_WORDS words
                                                (excludes scale labels such as
                                                "A great deal" / "Strongly
                                                agree" / "Major reason")
  * neither option is a Yes/No self-report   -- excludes factual items
  * neither option text is truncated ("...") -- a few source rows are clipped
  * a clean 2-way partisan distribution exists

L/R polarity (design doc section 1: "the side Republicans favour more = r_R")
-----------------------------------------------------------------------------
``r_R`` is the option with the larger Republican-minus-Democrat support
(``dist_rep - dist_dem``); ``r_L`` is the other. ``meta.partisan_gap`` records
the magnitude so downstream analysis can subset on partisan signal.

NOTE / known scope limits (reported to team-lead):
  * favor/oppose items are NOT included in v1 -- their poles ("Favor"/"Oppose")
    are not verbatim two-statement text. ~6 such items exist.
  * 3+ option forced-choice items (pole/pole/neutral) are NOT included in v1.
  * the 5-domain taxonomy (economy/welfare/energy/technology/diplomacy) was
    built for the old anchor-event data and does not cover the ATP political
    space (abortion, guns, gender, race, immigration, ...); ``domain`` is
    best-effort and ``null`` when nothing matches.

Run:  python atp_extract.py        (writes data/atp/dilemmas.json + a report)
"""

import csv
import json
import os
import re

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
RAW_DIR = os.path.join(REPO_ROOT, "data", "atp", "raw")
ITEMS_CSV = os.path.join(RAW_DIR, "pew_atp_items.csv")
PARTISAN_CSV = os.path.join(RAW_DIR, "pew_atp_partisan_distributions.csv")
OUT_PATH = os.path.join(REPO_ROOT, "data", "atp", "dilemmas.json")

OPTION_SEP = " | "
MIN_OPTION_WORDS = 5  # both poles must be statement-like, not scale labels

# Törnberg's pew_atp_items.csv caps option text at 120 characters, silently
# cutting longer statements mid-word (e.g. "...in top execu", "...compromises
# wit"). A pure length rule is unsafe -- a few genuine statements land exactly
# at 120 chars complete (e.g. EQUALPOLF2 "...in high political office"). These
# (wave, var_name) pairs were MANUALLY reviewed 2026-05-19 as having a
# cap-truncated pole and are dropped (a verbatim stimulus cannot be truncated).
_TRUNCATED_ITEMS = {
    ("W34", "EAT6"), ("W34", "EVOTHREE"), ("W36", "COLSPEECH"),
    ("W36", "EQUALBIZF2"), ("W42", "POLICY2"), ("W43", "RACESURV49"),
    ("W43", "RACESURV50"), ("W82", "GAP21Q36"), ("W92", "ALLIES"),
    ("W92", "FP_AUTH"), ("W92", "GOVAID"), ("W92", "PROG_RNEED2b"),
    ("W92", "VTRGHTPRIV1"),
}

# ---------------------------------------------------------------------------
# 5-domain taxonomy keyword map (design doc section 6). Best-effort only: the
# domain with the most whole-word keyword hits in the stem+options text wins;
# ties or zero hits -> domain = None. Keywords are matched with word-boundary
# regexes (so "war" does not hit "warming", "ally" does not hit "basically").
# Many ATP political items (abortion, guns, gender, race, free speech, ...)
# match nothing and are left unbinned -- the taxonomy predates the ATP data.
# ---------------------------------------------------------------------------
DOMAIN_KEYWORDS = {
    "economy": [
        "economy", "economic", "jobs", "wage", "wages", "tax", "taxes",
        "trade", "business", "businesses", "inequality", "poverty", "income",
        "deficit", "debt", "employment", "unemployment", "worker", "workers",
        "corporation", "corporations", "wealth", "capitalism", "manufacturing",
    ],
    "welfare": [
        "health care", "healthcare", "health insurance", "insurance",
        "social security", "medicare", "medicaid", "welfare", "tuition",
        "college", "student debt", "safety net", "housing", "child care",
        "childcare", "retirement", "retirees", "paid leave", "food stamps",
    ],
    "energy": [
        "energy", "climate", "global warming", "environment", "environmental",
        "oil", "natural gas", "coal", "fossil", "renewable", "solar",
        "emissions", "carbon", "pollution", "nuclear power", "power plant",
    ],
    "technology": [
        "technology", "technological", "robot", "robots", "computer",
        "computers", "internet", "online", "social media",
        "artificial intelligence", "automation", "scientific", "science",
        "research", "genetic", "algorithm", "smartphone", "evolved",
        "evolution", "vaccine",
    ],
    "diplomacy": [
        "foreign", "military", "war", "troops", "immigration", "immigrant",
        "immigrants", "border", "china", "russia", "terrorism", "nato",
        "allies", "alliance", "overseas", "refugee", "refugees", "defense",
        "world affairs",
    ],
}
DOMAIN_REGEXES = {
    dom: [re.compile(r"\b" + re.escape(kw) + r"\b", re.I) for kw in kws]
    for dom, kws in DOMAIN_KEYWORDS.items()
}

# Pure forced-choice boilerplate -- harmless if it stays in ``q`` (it just
# reads as a natural lead-in), so we do NOT strip it; listed here only for
# reference / possible future tightening.
FORCED_CHOICE_MARKERS = (
    "comes closer", "closer to your", "which statement",
    "which of the following statements", "best describes how you feel",
)


def _split_options(raw):
    return [o.strip() for o in raw.split(OPTION_SEP) if o.strip()]


def _word_count(text):
    return len(text.split())


def _is_yes_no(option):
    low = option.strip().lower()
    return low.startswith(("yes ", "yes,", "yes-", "no ", "no,", "no-")) or \
        low in ("yes", "no")


def _is_truncated(text):
    t = text.rstrip()
    return t.endswith("...") or t.endswith("…") or t.endswith("..")


def _assign_domain(text):
    """Best-effort 5-domain bin; None when nothing matches or on a tie."""
    scores = {
        domain: sum(1 for rx in rxs if rx.search(text))
        for domain, rxs in DOMAIN_REGEXES.items()
    }
    best = max(scores, key=lambda d: scores[d])
    if scores[best] == 0:
        return None
    # Reject ambiguous ties between distinct domains.
    top = sorted(scores.values(), reverse=True)
    if len(top) > 1 and top[0] == top[1]:
        return None
    return best


# Unresolved questionnaire piping ("[ITEM FROM DIFF1]", "[INSERT ...]") -- such
# stems are broken as standalone questions and the item is dropped.
_PIPING_RE = re.compile(r"\[[^\]]*\b(FROM|INSERT|PIPE|RANDOMIZE)\b", re.I)
# Conditional follow-ups ("When you say you favor ...") presuppose a prior
# answer and carry a filtered partisan distribution -- dropped.
_CONDITIONAL_RE = re.compile(r"^\s*when you say you\b", re.I)
# Bracketed construction notes ("[Summary variable to de-rotate ...]") -- these
# are internal annotations, stripped from the displayed stem.
_BRACKET_RE = re.compile(r"\s*\[[^\]]*\]\s*")


def _clean_stem(stem):
    """Strip internal bracketed annotations; collapse whitespace."""
    cleaned = _BRACKET_RE.sub(" ", stem)
    return re.sub(r"\s+", " ", cleaned).strip()


def _build_question(stem, opt_a, opt_b):
    """De-scaffold an ATP forced-choice item into an open question.

    ZERO added or generated wording (team-lead A2 constraint, 2026-05-19): q is
    the verbatim ATP stem followed by the two verbatim option statements, with
    only the (a)/(b) enumeration labels removed. The statements stay in their
    original ATP order so no L/R ordering is imposed on the prompt; r_L/r_R
    polarity is tracked separately in ``meta``. The "give your opinion"
    instruction is supplied by ``atp_probe.SYSTEM_PROMPT``, so q itself adds no
    question wording -- it is pure verbatim ATP text.
    """
    return f"{stem}\n{opt_a}\n{opt_b}"


def load_rows():
    with open(ITEMS_CSV, newline="") as f:
        items = list(csv.DictReader(f))
    with open(PARTISAN_CSV, newline="") as f:
        partisan = {(r["wave"], r["var_name"]): r for r in csv.DictReader(f)}
    return items, partisan


def extract():
    items, partisan = load_rows()
    dilemmas = []
    rejects = {
        "n_options_not_2": 0, "option_too_short": 0, "yes_no_factual": 0,
        "truncated_text": 0, "cap_truncated_option": 0,
        "piping_placeholder": 0, "conditional_followup": 0,
        "no_partisan_dist": 0, "dist_parse_error": 0,
    }
    next_id = 0

    for it in items:
        if it["n_options"] != "2":
            rejects["n_options_not_2"] += 1
            continue
        opts = _split_options(it["options"])
        if len(opts) != 2:
            rejects["n_options_not_2"] += 1
            continue
        if any(_is_truncated(o) for o in opts):
            rejects["truncated_text"] += 1
            continue
        if (it["wave"], it["var_name"]) in _TRUNCATED_ITEMS:
            rejects["cap_truncated_option"] += 1
            continue
        if min(_word_count(o) for o in opts) < MIN_OPTION_WORDS:
            rejects["option_too_short"] += 1
            continue
        if any(_is_yes_no(o) for o in opts):
            rejects["yes_no_factual"] += 1
            continue
        if _PIPING_RE.search(it["question"]):
            rejects["piping_placeholder"] += 1
            continue
        if _CONDITIONAL_RE.match(it["question"]):
            rejects["conditional_followup"] += 1
            continue

        key = (it["wave"], it["var_name"])
        pr = partisan.get(key)
        if pr is None or pr.get("error"):
            rejects["no_partisan_dist"] += 1
            continue
        try:
            labels = json.loads(pr["option_labels"])
            dist_rep = [float(x) for x in json.loads(pr["dist_rep"])]
            dist_dem = [float(x) for x in json.loads(pr["dist_dem"])]
        except (ValueError, KeyError, TypeError):
            rejects["dist_parse_error"] += 1
            continue
        if not (len(labels) == len(dist_rep) == len(dist_dem) == 2):
            rejects["dist_parse_error"] += 1
            continue

        # Polarity: r_R = the option Republicans favour more than Democrats do.
        lean = [dist_rep[i] - dist_dem[i] for i in range(2)]
        idx_R = 0 if lean[0] >= lean[1] else 1
        idx_L = 1 - idx_R
        r_R, r_L = opts[idx_R], opts[idx_L]
        partisan_gap = round(abs(lean[0]), 4)

        stem = _clean_stem(it["question"])
        domain = _assign_domain(it["question"] + " " + it["options"])

        dilemmas.append({
            "id": next_id,
            "domain": domain,
            "q": _build_question(stem, opts[0], opts[1]),
            "r_L": r_L,
            "r_R": r_R,
            "meta": {
                "wave": it["wave"],
                "var_name": it["var_name"],
                "format": "forced_choice_2",
                "stem": stem,
                "partisan_gap": partisan_gap,
                "dist_rep": {"r_L": round(dist_rep[idx_L], 4),
                             "r_R": round(dist_rep[idx_R], 4)},
                "dist_dem": {"r_L": round(dist_dem[idx_L], 4),
                             "r_R": round(dist_dem[idx_R], 4)},
                "n_rep": int(pr["n_rep"]) if pr.get("n_rep") else None,
                "n_dem": int(pr["n_dem"]) if pr.get("n_dem") else None,
            },
        })
        next_id += 1

    return dilemmas, rejects


def main():
    dilemmas, rejects = extract()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(dilemmas, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"atp_extract -- wrote {len(dilemmas)} dilemmas -> {OUT_PATH}")
    print("=" * 70)

    by_domain = {}
    by_wave = {}
    for d in dilemmas:
        by_domain[d["domain"]] = by_domain.get(d["domain"], 0) + 1
        w = d["meta"]["wave"]
        by_wave[w] = by_wave.get(w, 0) + 1

    print("\nper domain:")
    for dom in ["economy", "welfare", "energy", "technology", "diplomacy"]:
        print(f"  {dom:12s} {by_domain.get(dom, 0)}")
    print(f"  {'(none)':12s} {by_domain.get(None, 0)}")

    print("\nper wave:")
    for w in sorted(by_wave, key=lambda x: int(x[1:])):
        print(f"  {w:5s} {by_wave[w]}")

    print("\nrejected (not format-compatible):")
    for reason, n in rejects.items():
        print(f"  {reason:22s} {n}")

    gaps = sorted(d["meta"]["partisan_gap"] for d in dilemmas)
    if gaps:
        med = gaps[len(gaps) // 2]
        big = sum(1 for g in gaps if g >= 0.10)
        print(f"\npartisan gap: median {med:.3f}, "
              f"{big}/{len(gaps)} items with gap >= 0.10")


if __name__ == "__main__":
    main()
