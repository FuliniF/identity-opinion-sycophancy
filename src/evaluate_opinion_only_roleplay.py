"""Judge the role-play opinion-only responses with Gemini-2.5-flash.

Same protocol as evaluate_opinion_only.py, but for the responses where
the model role-plays a persona. Each response file holds three records
per dilemma (one per role-play anchor); the judge sees only the response
and the narrative, and the persona is recorded separately as
``responding_as``.

Inputs  : responses/<model>/<field>/<model>_as_characters_opinion_only_response.json
Output  : evaluations/opinion_only_roleplay/<model>/opinion_only_roleplay_eval_<field>.json
          [ {"id": 0, "responding_as": "...", "A": {...}, "B": {...}}, ... ]
"""

import argparse
import concurrent.futures as cf
import json
import os

from evaluate_opinion_only import (
    make_client, make_pool, judge, USER_PROMPT_GLOBAL, USER_PROMPT_REFUSAL,
    SCHEMA_GLOBAL, SCHEMA_REFUSAL, FIELDS, MODELS, SCEN_DIR, ORIG_DIR,
    RESP_DIR, REPO_ROOT)
from judge_pool import claim_output

OUT_DIR = os.path.join(REPO_ROOT, "evaluations", "opinion_only_roleplay")


def eval_model_field(client, model, field, args):
    out_model_dir = os.path.join(OUT_DIR, model)
    os.makedirs(out_model_dir, exist_ok=True)
    out_path = os.path.join(out_model_dir,
                            f"opinion_only_roleplay_eval_{field}.json")

    if args.overwrite and os.path.exists(out_path):
        try:
            os.remove(out_path)
        except OSError:
            pass

    with claim_output(out_path, label=f"{model}/{field}") as owned:
        if not owned:
            return

        with open(os.path.join(ORIG_DIR, field, "train.json")) as f:
            orig = json.load(f)
        with open(os.path.join(SCEN_DIR, field, "anchor_events.json")) as f:
            anchors = json.load(f)
        with open(os.path.join(SCEN_DIR, field, "narratives_opinion_only.json")) as f:
            narratives = {n["id"]: n for n in json.load(f)}
        resp_path = os.path.join(
            RESP_DIR, model, field,
            f"{model}_as_characters_opinion_only_response.json")
        with open(resp_path) as f:
            responses = json.load(f)  # list of {id, responding_as, response_A/B}

        def run(ri):
            rec = responses[ri]
            idx = rec["id"]
            a, o = anchors[idx], orig[idx]
            out = {"id": idx, "responding_as": rec["responding_as"]}
            for side in ("A", "B"):
                ctx = dict(focus_topic=o["focus_topic"], question=o["prompt"],
                           accept=o["accept"], reject=o["reject"],
                           objective_anchor=a["objective_anchor"],
                           mapping_A=a["mapping_A"], mapping_B=a["mapping_B"],
                           user_narrative=narratives[idx][f"narrative_{side}"],
                           ai_response=rec[f"response_{side}"])
                g = judge(client, args.eval_model,
                          USER_PROMPT_GLOBAL.format(**ctx), SCHEMA_GLOBAL,
                          max_tokens=args.max_tokens,
                          reasoning_effort=args.reasoning_effort or None)
                r = judge(client, args.eval_model,
                          USER_PROMPT_REFUSAL.format(**ctx), SCHEMA_REFUSAL,
                          max_tokens=args.max_tokens,
                          reasoning_effort=args.reasoning_effort or None)
                g["refusal"] = r.get("refusal")
                out[side] = g
            return ri, out

        results = {}
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            for ri, out in ex.map(run, range(len(responses))):
                results[ri] = out

        records = [results[i] for i in sorted(results)]
        with open(out_path, "w") as f:
            json.dump(records, f, indent=4)
        print(f"[done] {model}/{field}: {len(records)} records -> {out_path}",
              flush=True)


def main(args):
    # --out_dir routes results to a judge-specific tree (one per panel judge)
    # so they don't collide with the original Gemini evals.
    global OUT_DIR
    if args.out_dir:
        OUT_DIR = args.out_dir
    # Multi-endpoint pool; single-endpoint and OpenRouter defaults still
    # work unchanged.
    client = make_pool()
    models = MODELS if args.model == "all" else [args.model]
    for model in models:
        print(f"=== judging role-play opinion-only: {model} ===", flush=True)
        for field in FIELDS:
            eval_model_field(client, model, field, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all")
    parser.add_argument("--eval_model", default="google/gemini-2.5-flash")
    parser.add_argument("--workers", type=int, default=64,
                        help="concurrent judge calls (vLLM judge saturates ~64)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out_dir", default="",
                        help="output root for eval results; default "
                             "evaluations/opinion_only_roleplay. Set per judge "
                             "to keep panel judges' outputs separate.")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="judge completion budget; raise for reasoning "
                             "judges (e.g. 3072 for gpt-oss).")
    parser.add_argument("--reasoning_effort", default="",
                        help="reasoning effort for reasoning judges "
                             "(e.g. low/medium); empty = omit.")
    main(parser.parse_args())
