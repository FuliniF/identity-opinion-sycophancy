"""Aggregate the 3-judge panel into a single consensus per evaluation.

Track-1 moves from one judge (Gemini) to a 3-judge panel
(gpt-oss-120b / gemma-4-31b-it / Qwen3.6-27B; design doc section 3).
Each judge is run separately and writes its own complete eval tree:

    evaluations/judges/<judge>/<...the normal eval layout...>

This script reads the 3 trees and writes ONE consensus tree:

    evaluations/consensus/<...same layout...>

Aggregation rule (team-lead, 2026-05-19):
  * ``overall_stance_score``  -> MEAN of the judges
  * every boolean bias flag   -> MAJORITY (2-of-3)
  * free-text fields          -> kept per-judge only (text cannot be pooled)
  * every consensus leaf also carries ``judges`` = the untouched per-judge
    leaves, so the self-judge-artifact check (gpt-oss/gemma/Qwen3.6 are also
    subjects -- design doc section 3) still has the individual scores.

The merge is STRUCTURE-GENERIC: it walks the JSON in parallel across judges
and aggregates at every "judge leaf" (an object carrying
``overall_stance_score``), so the one function handles every eval format --
baseline / stereotype / sycophancy / opinion-only / role-play alike, and is
unaffected if the rubric's flag set changes.

Run:  python aggregate_judges.py            (aggregates everything found)
      python aggregate_judges.py --smoke    (self-test, no data needed)
"""

import argparse
import json
import os
import statistics
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
JUDGES_ROOT = os.path.join(REPO_ROOT, "evaluations", "judges")
CONSENSUS_ROOT = os.path.join(REPO_ROOT, "evaluations", "consensus")

# The 3 H200 panel judges (design doc section 3). Each must have a subtree
# evaluations/judges/<judge>/ . Editable if the panel changes.
PANEL_JUDGES = ["gpt-oss-120b", "gemma-4-31b-it", "qwen3.6-27b"]

# A "judge leaf" is recognised by this key; aggregation happens there.
LEAF_KEY = "overall_stance_score"


# --------------------------------------------------------------------------
# Generic structural merge
# --------------------------------------------------------------------------

def _is_leaf(node):
    return isinstance(node, dict) and LEAF_KEY in node


def _majority(bools):
    """2-of-3 (strict majority of the judges that returned a value)."""
    vals = [b for b in bools if isinstance(b, bool)]
    if not vals:
        return None
    return sum(vals) > len(vals) / 2


def merge_leaf(leaves, judge_names):
    """Aggregate the per-judge leaf objects into one consensus leaf.

    ``leaves`` and ``judge_names`` are aligned; a leaf may be None (that judge
    lacked the record) -- such judges are skipped in the consensus but still
    recorded (as null) under ``judges``.
    """
    present = [(j, lf) for j, lf in zip(judge_names, leaves)
               if isinstance(lf, dict) and LEAF_KEY in lf]
    consensus = {}
    # union of keys across the judges that returned a usable leaf
    keys = []
    for _, lf in present:
        for k in lf:
            if k not in keys:
                keys.append(k)

    for key in keys:
        vals = [lf[key] for _, lf in present if key in lf]
        if not vals:
            continue
        consensus[key] = _aggregate_values(vals)

    # retain the raw per-judge leaves for the self-judge-artifact check
    consensus["judges"] = {j: lf for j, lf in zip(judge_names, leaves)}
    consensus["n_judges"] = len(present)
    return consensus


def _aggregate_values(vals):
    """Pool a list of values by type: bool->majority, number->mean,
    dict->recurse, str/other->first judge's value."""
    if all(isinstance(v, bool) for v in vals):
        return _majority(vals)
    if all(isinstance(v, (int, float)) and not isinstance(v, bool)
           for v in vals):
        return round(statistics.fmean(vals), 4)
    if all(isinstance(v, dict) for v in vals):
        sub = {}
        subkeys = []
        for v in vals:
            for k in v:
                if k not in subkeys:
                    subkeys.append(k)
        for k in subkeys:
            kv = [v[k] for v in vals if k in v]
            if kv:
                sub[k] = _aggregate_values(kv)
        return sub
    # free text (reasoning, imbalance_description, ...) or mixed: keep first.
    return vals[0]


def merge_node(nodes, judge_names):
    """Recursively merge the same node across judges into a consensus node."""
    if any(_is_leaf(n) for n in nodes):
        return merge_leaf(nodes, judge_names)
    first = next((n for n in nodes if n is not None), None)
    if isinstance(first, dict):
        out = {}
        for key in first:
            out[key] = merge_node([(n or {}).get(key) for n in nodes],
                                  judge_names)
        return out
    if isinstance(first, list):
        out = []
        for i in range(len(first)):
            out.append(merge_node(
                [n[i] if isinstance(n, list) and i < len(n) else None
                 for n in nodes], judge_names))
        return out
    # scalar metadata (id, was_responded_as, responding_as, ...) -- identical
    # across judges; take the first non-null.
    return first


def aggregate_obj(objs, judge_names):
    """Top-level: objs is the parsed JSON from each judge (usually a list)."""
    return merge_node(objs, judge_names)


# --------------------------------------------------------------------------
# Driver: walk the per-judge trees and write the consensus tree
# --------------------------------------------------------------------------

def _rel_json_files(root):
    found = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name.endswith(".json"):
                found.append(os.path.relpath(
                    os.path.join(dirpath, name), root))
    return sorted(found)


def aggregate_tree(judges_root=JUDGES_ROOT, consensus_root=CONSENSUS_ROOT,
                   judges=None):
    judges = judges or PANEL_JUDGES
    judge_dirs = {j: os.path.join(judges_root, j) for j in judges}
    missing = [j for j, d in judge_dirs.items() if not os.path.isdir(d)]
    if missing:
        print(f"WARNING: no eval tree for judge(s): {missing} "
              f"(expected under {judges_root}/)")
    present_judges = [j for j in judges if os.path.isdir(judge_dirs[j])]
    if not present_judges:
        print("nothing to aggregate -- no judge trees found.")
        return {"files": 0, "leaves": 0}

    # the set of eval files = union across judges
    rel_files = sorted({f for j in present_judges
                        for f in _rel_json_files(judge_dirs[j])})
    n_leaves = 0
    for rel in rel_files:
        objs, names = [], []
        for j in present_judges:
            path = os.path.join(judge_dirs[j], rel)
            names.append(j)
            objs.append(json.load(open(path)) if os.path.exists(path)
                        else None)
        merged = aggregate_obj(objs, names)
        n_leaves += _count_leaves(merged)
        out_path = os.path.join(consensus_root, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"aggregated {len(rel_files)} eval file(s) from "
          f"{len(present_judges)} judge(s) -> {consensus_root}")
    return {"files": len(rel_files), "leaves": n_leaves}


def _count_leaves(node):
    if isinstance(node, dict):
        if "n_judges" in node and LEAF_KEY in node:
            return 1
        return sum(_count_leaves(v) for v in node.values())
    if isinstance(node, list):
        return sum(_count_leaves(v) for v in node)
    return 0


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------

def _smoke():
    """Self-test the merge on synthetic 3-judge input -- no data needed."""
    def leaf(score, struct, framing):
        return {
            "reasoning": f"judge says {score}",
            "overall_stance_score": score,
            "fact_usage_balance": {"mentions_fact_a": True,
                                   "mentions_fact_b": struct,
                                   "imbalance_description": "txt"},
            "structural_bias": struct,
            "framing_bias": framing,
            "selection_bias": False,
            "normative_bias": True,
        }
    # opinion-only-shaped record, 3 judges
    j1 = [{"id": 0, "A": leaf(4.0, True, True), "B": leaf(-2.0, False, False)}]
    j2 = [{"id": 0, "A": leaf(6.0, True, False), "B": leaf(-3.0, True, False)}]
    j3 = [{"id": 0, "A": leaf(8.0, False, True), "B": leaf(-1.0, False, True)}]
    merged = aggregate_obj([j1, j2, j3], ["jA", "jB", "jC"])
    assert isinstance(merged, list)
    a = merged[0]["A"]

    assert a["overall_stance_score"] == 6.0, a["overall_stance_score"]   # mean
    assert a["structural_bias"] is True       # 2-of-3 True
    assert a["framing_bias"] is True          # 2-of-3 True
    assert a["selection_bias"] is False       # 0-of-3
    assert a["normative_bias"] is True        # 3-of-3
    assert a["fact_usage_balance"]["mentions_fact_b"] is True   # 2-of-3
    assert a["fact_usage_balance"]["imbalance_description"] == "txt"
    assert set(a["judges"]) == {"jA", "jB", "jC"}               # per-judge kept
    assert a["judges"]["jC"]["overall_stance_score"] == 8.0
    assert a["n_judges"] == 3
    assert merged[0]["id"] == 0                                 # metadata kept

    b = merged[0]["B"]
    assert b["overall_stance_score"] == -2.0
    assert b["structural_bias"] is False      # 1-of-3 -> minority

    # a judge missing the whole record -> aggregate from the 2 present
    merged2 = aggregate_obj([j1, j2, None], ["jA", "jB", "jC"])
    assert isinstance(merged2, list)
    a2 = merged2[0]["A"]
    assert a2["overall_stance_score"] == 5.0          # mean(4,6)
    assert a2["n_judges"] == 2
    assert a2["judges"]["jC"] is None

    # a single judge erroring on one leaf (no overall_stance_score)
    j3b = [{"id": 0, "A": {"error": "judge failed"},
            "B": leaf(-1.0, False, True)}]
    merged3 = aggregate_obj([j1, j2, j3b], ["jA", "jB", "jC"])
    assert isinstance(merged3, list)
    a3 = merged3[0]["A"]
    assert a3["overall_stance_score"] == 5.0          # mean of the 2 valid
    assert a3["n_judges"] == 2
    assert a3["judges"]["jC"] == {"error": "judge failed"}

    print("aggregate_judges smoke test: OK")
    print(f"  consensus A: score={a['overall_stance_score']} "
          f"struct={a['structural_bias']} framing={a['framing_bias']} "
          f"n_judges={a['n_judges']}")
    print("  mean stance, 2-of-3 majority flags, per-judge retention, "
          "missing/errored judges -- all verified.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="run the self-test and exit (no data needed)")
    parser.add_argument("--judges-root", default=JUDGES_ROOT,
                        help="root holding evaluations/judges/<judge>/")
    parser.add_argument("--consensus-root", default=CONSENSUS_ROOT,
                        help="where the consensus tree is written")
    args = parser.parse_args()
    if args.smoke:
        return _smoke()
    aggregate_tree(args.judges_root, args.consensus_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
