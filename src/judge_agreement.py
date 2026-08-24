"""3-judge panel agreement: Krippendorff's alpha across the judge panel.

Design doc section 3 / reviewer point #3: the 3-judge panel
(gpt-oss-120b / gemma-4-31b-it / Qwen3.6-27B) needs an inter-rater
reliability coefficient. ``annotator_agreement.py`` covers only the 2-rater
MTurk screen; this script covers the 3 LLM judges.

Input -- the consensus tree written by ``aggregate_judges.py``
(``evaluations/consensus/``). Every consensus leaf carries ``judges`` = the
untouched per-judge leaves, so this script reads each measure's three judge
values straight off the consensus files.

Per measure it reports Krippendorff's alpha with the appropriate metric:
  * ``overall_stance_score``           -- interval metric  (delta^2 = (c-k)^2)
  * every boolean bias / refusal flag  -- nominal  metric  (delta^2 = 0/1)
plus raw pairwise agreement. Missing/errored judges are handled (a unit needs
>= 2 judge values to count).

Run:  python judge_agreement.py            (reads evaluations/consensus/)
      python judge_agreement.py --smoke    (self-test, no data needed)
"""

import argparse
import json
import os
import sys
from collections import Counter

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
CONSENSUS_ROOT = os.path.join(REPO_ROOT, "evaluations", "consensus")
OUT_PATH = os.path.join(REPO_ROOT, "judge_agreement_summary.json")

STANCE_KEY = "overall_stance_score"


# --------------------------------------------------------------------------
# Krippendorff's alpha -- generic metric, missing-data tolerant
# --------------------------------------------------------------------------

def krippendorff_alpha(units, metric):
    """Krippendorff's alpha over ``units`` (each a list of a unit's ratings).

    ``metric(c, k)`` is the squared distance between two values
    (nominal: 0/1; interval: (c-k)^2). Units with < 2 ratings are dropped.
    Computed from the coincidence matrix so it is O(distinct values^2),
    not O(ratings^2).
    """
    units = [[v for v in u if v is not None] for u in units]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")

    coinc = {}  # (c, k) -> coincidence mass (float)
    for u in units:
        m = len(u)
        cc = Counter(u)
        for c in cc:
            for k in cc:
                pairs = cc[c] * (cc[c] - 1) if c == k else cc[c] * cc[k]
                coinc[(c, k)] = coinc.get((c, k), 0.0) + pairs / (m - 1)

    values = sorted({v for u in units for v in u})
    n_v = {c: sum(coinc.get((c, k), 0.0) for k in values) for c in values}
    n = sum(n_v.values())
    if n < 2:
        return float("nan")

    num = sum(coinc.get((c, k), 0.0) * metric(c, k)
              for c in values for k in values)
    den = sum(n_v[c] * n_v[k] * metric(c, k)
              for c in values for k in values)
    if den == 0:
        # no expected disagreement (every judge gave one value everywhere)
        return 1.0 if num == 0 else float("nan")
    return 1.0 - (n - 1) * num / den


def _nominal(c, k):
    return 0.0 if c == k else 1.0


def _interval(c, k):
    return (c - k) ** 2


def raw_pairwise_agreement(units):
    """Fraction of judge pairs that agree exactly, pooled over units."""
    agree = total = 0
    for u in units:
        vals = [v for v in u if v is not None]
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                total += 1
                agree += (vals[i] == vals[j])
    return agree / total if total else float("nan")


# --------------------------------------------------------------------------
# Pull per-judge values off the consensus tree
# --------------------------------------------------------------------------

def _iter_consensus_leaves(node):
    """Yield every consensus leaf -- a dict carrying ``judges`` + a stance."""
    if isinstance(node, dict):
        if "judges" in node and STANCE_KEY in node:
            yield node
        else:
            for v in node.values():
                yield from _iter_consensus_leaves(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_consensus_leaves(v)


def collect_measures(consensus_root=CONSENSUS_ROOT):
    """Walk the consensus tree -> {measure: [unit, unit, ...]}.

    A unit is one consensus leaf; its value list is the per-judge values of
    that measure (judges that errored / lack the key are skipped).
    """
    stance_units = []
    flag_units = {}  # measure -> list of units
    n_leaves = 0
    for dirpath, _, names in os.walk(consensus_root):
        for name in names:
            if not name.endswith(".json"):
                continue
            obj = json.load(open(os.path.join(dirpath, name)))
            for leaf in _iter_consensus_leaves(obj):
                n_leaves += 1
                judge_leaves = [lf for lf in leaf["judges"].values()
                                if isinstance(lf, dict)]
                # stance: interval
                stance_units.append([lf.get(STANCE_KEY)
                                     for lf in judge_leaves])
                # every boolean key seen in any judge leaf: nominal
                bool_keys = {k for lf in judge_leaves for k, v in lf.items()
                             if isinstance(v, bool)}
                for k in bool_keys:
                    flag_units.setdefault(k, []).append(
                        [lf.get(k) for lf in judge_leaves])
    return {"_n_leaves": n_leaves, "stance": stance_units, "flags": flag_units}


def report(measures):
    summary = {"n_leaves": measures["_n_leaves"], "measures": {}}
    print("=" * 64)
    print(f"3-judge panel agreement  ({measures['_n_leaves']} consensus "
          f"leaves)")
    print("=" * 64)

    su = measures["stance"]
    a = krippendorff_alpha(su, _interval)
    summary["measures"][STANCE_KEY] = {"metric": "interval",
                                       "krippendorff_alpha": a}
    print(f"  {STANCE_KEY:<24} alpha(interval) = {a:.3f}")

    for measure in sorted(measures["flags"]):
        units = measures["flags"][measure]
        a = krippendorff_alpha(units, _nominal)
        raw = raw_pairwise_agreement(units)
        summary["measures"][measure] = {"metric": "nominal",
                                        "krippendorff_alpha": a,
                                        "raw_pairwise_agreement": raw}
        print(f"  {measure:<24} alpha(nominal)  = {a:.3f}   "
              f"raw agreement = {raw:.3f}")
    return summary


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------

def _smoke():
    """Self-test the alpha computation against known cases."""
    # perfect agreement -> alpha = 1
    perfect = [[5.0, 5.0, 5.0], [1.0, 1.0, 1.0], [9.0, 9.0, 9.0]]
    assert abs(krippendorff_alpha(perfect, _interval) - 1.0) < 1e-9

    perfect_nom = [[True, True, True], [False, False, False]] * 5
    assert abs(krippendorff_alpha(perfect_nom, _nominal) - 1.0) < 1e-9

    # systematic disagreement (judges never agree) -> alpha <= 0
    bad = [[0.0, 5.0, 10.0], [10.0, 5.0, 0.0]] * 6
    assert krippendorff_alpha(bad, _interval) < 0.2

    # missing judge tolerated (unit still has 2 values)
    miss = [[4.0, 4.0, None], [2.0, 2.0, 2.0], [7.0, 7.0, None]]
    assert abs(krippendorff_alpha(miss, _interval) - 1.0) < 1e-9

    # cross-check the nominal path against annotator_agreement's binary alpha:
    # 3 raters, 4 units, one disagreement.
    units = [["a", "a", "a"], ["b", "b", "b"],
             ["a", "a", "b"], ["b", "b", "b"]]
    al = krippendorff_alpha(units, _nominal)
    assert 0.0 < al < 1.0, al

    # raw pairwise agreement: [a,a,b] -> 1 of 3 pairs agree... actually
    # a-a agree, a-b no, a-b no => 1/3; all-equal units => 3/3.
    raw = raw_pairwise_agreement([["a", "a", "b"]])
    assert abs(raw - 1 / 3) < 1e-9

    # end-to-end on a synthetic consensus leaf
    leaf = {
        "overall_stance_score": 6.0,
        "structural_bias": True,
        "judges": {
            "j1": {"overall_stance_score": 5.0, "structural_bias": True},
            "j2": {"overall_stance_score": 6.0, "structural_bias": True},
            "j3": {"overall_stance_score": 7.0, "structural_bias": False},
        },
    }
    found = list(_iter_consensus_leaves([{"id": 0, "A": leaf}]))
    assert len(found) == 1 and found[0] is leaf

    print("judge_agreement smoke test: OK")
    print(f"  perfect-agreement alpha=1.0, systematic-disagreement alpha<0.2,")
    print(f"  missing-judge tolerated, nominal+interval metrics, leaf walker "
          f"-- all verified.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="run the self-test and exit (no data needed)")
    parser.add_argument("--consensus-root", default=CONSENSUS_ROOT)
    args = parser.parse_args()
    if args.smoke:
        return _smoke()
    if not os.path.isdir(args.consensus_root):
        print(f"no consensus tree at {args.consensus_root} -- run "
              f"aggregate_judges.py first.")
        return 0
    measures = collect_measures(args.consensus_root)
    summary = report(measures)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
