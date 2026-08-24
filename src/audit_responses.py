"""Extended degeneracy re-audit of the anchor-event response dataset.

The #17 audit detected only empty / missing / ERROR-sentinel cells. This
re-audit (Task #24) classifies EVERY response cell across all 14 subject
models with ``degeneracy.classify`` -- catching the non-empty degeneracy modes
(provider-gateway refusals, cjk-heavy responses, reasoning-channel leakage,
truncations) that #17 was structurally blind to.

Output: per model x condition defect counts, per-model totals, grand total.

Run from the src/ directory:  python audit_responses.py [--model NAME]
"""

import argparse
import json
import os

from degeneracy import CLASSES, DEFECT_CLASSES, classify
from subject_models import FRIENDLY_NAMES

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
RESP_DIR = os.path.join(REPO_ROOT, "responses")
DOMAINS = ["diplomatic", "economy", "energy", "technology", "welfare"]

# condition -> (filename suffix, is_jsonl)
COND_SUFFIX = {
    "baseline":        ("_new_baseline_response.jsonl",               True),
    "baseline_rp":     ("_as_characters_new_baseline_response.jsonl", True),
    "stereotyping":    ("_stereotyping_response.json",                False),
    "stereotyping_rp": ("_as_characters_stereotyping_response.json",  False),
    "sycophancy":      ("_sycophancy_response.json",                  False),
    "sycophancy_rp":   ("_as_characters_sycophancy_response.json",    False),
    "opinion_only":    ("_opinion_only_response.json",                False),
    "opinion_only_rp": ("_as_characters_opinion_only_response.json",  False),
}
CONDITIONS = list(COND_SUFFIX)
_META_KEYS = {"id", "responding_as", "model", "was_responded_as"}


def classify_filename(fn):
    rp = "_as_characters_" in fn
    for cond, (suffix, _) in COND_SUFFIX.items():
        if fn.endswith(suffix) and ("_as_characters_" in suffix) == rp:
            return cond
    return None


def iter_cells(condition, data):
    """Yield every response string for one loaded response file."""
    base = condition.replace("_rp", "")
    if base == "baseline":
        for row in data:
            if isinstance(row, dict):
                yield row.get("response", "")
    elif base in ("stereotyping", "opinion_only"):
        for item in data:
            if not isinstance(item, dict):
                continue
            if base == "opinion_only":
                for side in ("response_A", "response_B"):
                    if side in item:
                        yield item[side]
            else:  # stereotyping: char key -> response string
                for k, v in item.items():
                    if k not in _META_KEYS and isinstance(v, str):
                        yield v
    elif base == "sycophancy":
        for item in data:
            if not isinstance(item, dict):
                continue
            for k, v in item.items():
                if k in _META_KEYS or not isinstance(v, dict):
                    continue
                for side in ("response_A", "response_B"):
                    if side in v:
                        yield v[side]


def load_file(path, is_jsonl):
    if is_jsonl:
        with open(path) as f:
            return [json.loads(ln) for ln in f if ln.strip()]
    with open(path) as f:
        return json.load(f)


def audit_model(model):
    """Return {condition: {class: count}} for one model."""
    by_condition = {}
    for domain in DOMAINS:
        ddir = os.path.join(RESP_DIR, model, domain)
        if not os.path.isdir(ddir):
            continue
        for fn in sorted(os.listdir(ddir)):
            condition = classify_filename(fn)
            if condition is None:
                continue
            is_jsonl = COND_SUFFIX[condition][1]
            try:
                data = load_file(os.path.join(ddir, fn), is_jsonl)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {model}/{domain}/{fn}: unreadable ({e})")
                continue
            counts = by_condition.setdefault(
                condition, {c: 0 for c in CLASSES})
            for resp in iter_cells(condition, data):
                counts[classify(resp)] += 1
    return by_condition


def main(args):
    models = ([args.model] if args.model != "all" else FRIENDLY_NAMES)
    grand = {c: 0 for c in CLASSES}
    clean_models, defect_models = [], []

    for model in models:
        by_condition = audit_model(model)
        if not by_condition:
            print(f"\n{model}: no response files found")
            continue
        model_tot = {c: 0 for c in CLASSES}
        for counts in by_condition.values():
            for c in CLASSES:
                model_tot[c] += counts[c]
        cells = sum(model_tot.values())
        defects = sum(model_tot[c] for c in DEFECT_CLASSES)
        for c in CLASSES:
            grand[c] += model_tot[c]

        tag = "CLEAN" if defects == 0 else f"{defects} defects"
        (clean_models if defects == 0 else defect_models).append(model)
        print(f"\n{'='*78}\n{model}  --  {cells} cells, {tag}")
        if defects:
            summary = "  ".join(f"{c}={model_tot[c]}"
                                for c in DEFECT_CLASSES if model_tot[c])
            print(f"  TOTAL: {summary}")
            for cond in CONDITIONS:
                counts = by_condition.get(cond)
                if not counts:
                    continue
                d = "  ".join(f"{c}={counts[c]}"
                              for c in DEFECT_CLASSES if counts[c])
                if d:
                    print(f"    {cond:18s} {d}")

    print(f"\n{'='*78}\nGRAND TOTAL across {len(models)} models")
    print(f"  cells scanned : {sum(grand.values())}")
    print(f"  usable (ok)   : {grand['ok']}")
    for c in DEFECT_CLASSES:
        if grand[c]:
            print(f"  {c:15s}: {grand[c]}")
    print(f"  TOTAL DEFECTS : {sum(grand[c] for c in DEFECT_CLASSES)}")
    print(f"\n  clean models  ({len(clean_models)}): "
          f"{', '.join(clean_models) or '-'}")
    print(f"  models w/ defects ({len(defect_models)}): "
          f"{', '.join(defect_models) or '-'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        help="'all' or one friendly model name")
    main(parser.parse_args())
