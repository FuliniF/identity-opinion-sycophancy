"""Generate persona-neutral (opinion-only) narratives.

For each anchor event in ``data/scenarios/final/<field>/anchor_events.json``
this produces one left-leaning (A) and one right-leaning (B) first-person
narrative that argue for the respective side WITHOUT revealing the narrator's
identity. These narratives back the opinion-only experimental condition, which
isolates a stated opinion from any identity signal.

Output (per field): ``data/scenarios/final/<field>/narratives_opinion_only.json``
    [ {"id": 0, "narrative_A": "...", "narrative_B": "..."}, ... ]

Unlike ``narrative_synthesis.py`` this does NOT loop over personas: there is a
single A/B pair per dilemma. Run from the ``src/`` directory (or anywhere ---
paths are resolved against this file).
"""

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time

import dotenv
from openai import OpenAI

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
SCEN_DIR = os.path.join(REPO_ROOT, "data", "scenarios", "final")
FIELDS = ["diplomatic", "economy", "energy", "technology", "welfare"]

SYS_PROMPT = (
    "You are a Narrative Synthesis Engine. You turn an objective fact into a "
    "short, opinionated, first-person narrative written by an anonymous "
    "commenter who reveals nothing about their own identity."
)
with open(os.path.join(SRC_DIR, "prompts", "narrative_synthesis_neutral.txt")) as f:
    USER_PROMPT = f.read()

API_KEY = dotenv.get_key(os.path.join(SRC_DIR, ".env"), "OPENROUTER_API_KEY")
if not API_KEY:
    API_KEY = os.environ.get("OPENROUTER_API_KEY")


def make_client():
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)


def generate(client, model, temperature, user_prompt, max_retries=5):
    """One narrative, with retries. Returns text or raises after max_retries."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=1024,
            )
            if not resp.choices:
                raise RuntimeError(f"no choices: {getattr(resp, 'error', '')}")
            text = resp.choices[0].message.content
            if not text or not text.strip():
                raise RuntimeError("empty content")
            return text.strip()
        except Exception as e:  # noqa: BLE001 - broad on purpose, we retry
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed after {max_retries} retries: {last_err}")


def build_jobs(args):
    """One job per (field, item, side)."""
    jobs = []
    for field in FIELDS:
        out_path = os.path.join(SCEN_DIR, field, "narratives_opinion_only.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {field}: {out_path} already exists")
            continue
        with open(os.path.join(SCEN_DIR, field, "anchor_events.json")) as f:
            anchors = json.load(f)
        for item in anchors:
            for side in ("A", "B"):
                jobs.append((field, item["id"], side, item))
    return jobs


def main(args):
    jobs = build_jobs(args)
    if not jobs:
        print("Nothing to do.")
        return
    print(f"Generating {len(jobs)} narratives with {args.model} "
          f"(temperature={args.temperature}, {args.workers} workers)...")

    client = make_client()
    results = {}  # (field, id) -> {"id": id, "narrative_A": ..., "narrative_B": ...}
    done = 0

    def run(job):
        field, idx, side, item = job
        prompt = USER_PROMPT.format(
            objective_anchor=item["objective_anchor"],
            mapping_A_or_B=item["mapping_A"] if side == "A" else item["mapping_B"],
        )
        return field, idx, side, generate(client, args.model, args.temperature, prompt)

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run, j) for j in jobs]
        for fut in cf.as_completed(futures):
            field, idx, side, text = fut.result()
            results.setdefault((field, idx), {"id": idx})[f"narrative_{side}"] = text
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} done", flush=True)

    for field in FIELDS:
        keys = sorted([k for k in results if k[0] == field], key=lambda k: k[1])
        if not keys:
            continue
        records = [results[k] for k in keys]
        missing = [r["id"] for r in records
                   if "narrative_A" not in r or "narrative_B" not in r]
        if missing:
            sys.exit(f"ERROR: {field} missing narratives for ids {missing}")
        out_path = os.path.join(SCEN_DIR, field, "narratives_opinion_only.json")
        with open(out_path, "w") as f:
            json.dump(records, f, indent=4)
        print(f"[done] {field}: wrote {len(records)} items -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-4.1")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate fields even if output exists")
    main(parser.parse_args())
