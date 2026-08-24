"""Run the configured subject models over every ATP probe.

For each dilemma in ``data/atp/dilemmas.json`` this generates one open-ended
response per (condition x identity tag x opinion direction) -- 28 probes per
dilemma (see ``atp_probe.enumerate_probes``). There is no anchor event and no
narrative synthesis.

Routing
-------
- Closed subjects (``o4-mini``, ``nova-lite-v1``) always go to OpenRouter.
- Open subjects go to the H200 vLLM endpoint when ``GEN_BASE_URL`` is set
  (``GEN_API_KEY`` optional), otherwise fall back to OpenRouter.

The 16-model subject set, temperatures and reasoning efforts are reused from
``opinion_only.MODELS`` so the ATP run stays in sync with the pilot.

Output
------
``responses/atp/<friendly>/atp_responses.jsonl`` -- one JSON record per probe:
  {"dilemma_id", "domain", "condition", "identity_tag", "opinion_dir",
   "response"}
The run is resumable: records already on disk are loaded and their probes
skipped. ``--overwrite`` discards the file and regenerates from scratch.

Smoke test (needs ``data/atp/dilemmas.json`` + reachable endpoints):
    python atp_generate.py --model deepseek-v3.1 --limit 3

Run from the src/ directory.
"""

import argparse
import concurrent.futures as cf
import json
import os
import threading
import time

import dotenv
from openai import OpenAI

from atp_probe import enumerate_probes, load_config
from opinion_only import MODELS  # reuse the 16-model subject set
from usage_logger import log_usage

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
DILEMMAS_PATH = os.path.join(REPO_ROOT, "data", "atp", "dilemmas.json")
RESP_ROOT = os.path.join(REPO_ROOT, "responses", "atp")

# Closed subjects: no self-hosted weights, so they always go to OpenRouter --
# even when GEN_BASE_URL points the open subjects at the H200 vLLM cluster.
CLOSED_MODELS = {"o4-mini", "nova-lite-v1"}

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

API_KEY = dotenv.get_key(os.path.join(SRC_DIR, ".env"), "OPENROUTER_API_KEY")
if not API_KEY:
    API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Probes per dilemma (baseline + neutral + 8 identity + 2 opinion + 16
# identity*opinion); used only for the run-size banner.
PROBES_PER_DILEMMA = 28


def make_client(friendly):
    """OpenAI-compatible client for one subject, routed per the rules above."""
    if friendly not in CLOSED_MODELS:
        gen_base = os.environ.get("GEN_BASE_URL")
        if gen_base:
            return OpenAI(
                base_url=gen_base,
                api_key=os.environ.get("GEN_API_KEY") or API_KEY or "EMPTY",
            )
    return OpenAI(base_url=OPENROUTER_BASE, api_key=API_KEY)


def generate(client, model, temperature, effort, system_prompt, user_prompt,
             max_tokens=4096, max_retries=5):
    """One response, with retry/backoff. Returns text or an ERROR sentinel.

    On persistent failure returns the "ERROR WHEN GENERATING RESPONSE: ..."
    sentinel (the same string the pilot scripts use) rather than raising, so a
    thread-pool run is not aborted by one bad call.
    """
    extra = {"extra_body": {"reasoning": {"effort": effort}}} if effort else {}
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                **extra,
            )
            if not resp.choices:
                raise RuntimeError(f"no choices: {getattr(resp, 'error', '')}")
            choice = resp.choices[0]
            text = choice.message.content
            if not text or not text.strip():
                raise RuntimeError("empty content")
            # A healthy completion ends with finish_reason "stop" (or "length"
            # if it hit max_tokens); anything else signals provider truncation.
            if choice.finish_reason not in ("stop", "length"):
                raise RuntimeError(f"bad finish_reason {choice.finish_reason!r} "
                                   f"({len(text)} chars)")
            log_usage(model, "atp_generate", getattr(resp, "usage", None))
            return text.strip()
        except Exception as e:  # noqa: BLE001 - broad on purpose, we retry
            last_err = e
            time.sleep(2 * (attempt + 1))
    return f"ERROR WHEN GENERATING RESPONSE: {last_err}"


def _probe_key(rec):
    """Identity of one probe -- stable across the probe dict and a JSONL row."""
    return (rec["dilemma_id"], rec["condition"],
            rec.get("identity_tag"), rec.get("opinion_dir"))


def load_done(out_path):
    """Set of probe keys already present in a resumable JSONL output file."""
    done = set()
    if not os.path.exists(out_path):
        return done
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(_probe_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a partial last line from a crashed run
    return done


def run_model(friendly, api_id, temperature, effort, dilemmas, config, args):
    out_dir = os.path.join(RESP_ROOT, friendly)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "atp_responses.jsonl")

    if args.overwrite and os.path.exists(out_path):
        os.remove(out_path)
    done = load_done(out_path)

    probes = []
    for dilemma in dilemmas:
        for probe in enumerate_probes(dilemma, config):
            if _probe_key(probe) not in done:
                probes.append(probe)

    if not probes:
        print(f"[skip] {friendly}: all probes already generated "
              f"({len(done)} in {out_path})", flush=True)
        return

    client = make_client(friendly)
    lock = threading.Lock()

    def run(probe):
        text = generate(client, api_id, temperature, effort,
                         probe["system_prompt"], probe["user_prompt"],
                         max_tokens=args.max_tokens)
        rec = {
            "dilemma_id": probe["dilemma_id"],
            "domain": probe["domain"],
            "condition": probe["condition"],
            "identity_tag": probe["identity_tag"],
            "opinion_dir": probe["opinion_dir"],
            "response": text,
        }
        # Append-as-completed under a lock so a crashed run stays resumable.
        with lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    count = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in ex.map(run, probes):
            count += 1
    print(f"[done] {friendly}: {count} probes generated "
          f"({len(done)} skipped) -> {out_path}", flush=True)


def main(args):
    if not os.path.exists(args.dilemmas):
        raise SystemExit(
            f"ATP dilemmas not found: {args.dilemmas}\n"
            f"Phase A (atp_extract.py -> data/atp/dilemmas.json) must run "
            f"first.")
    with open(args.dilemmas) as f:
        dilemmas = json.load(f)
    if args.limit > 0:
        dilemmas = dilemmas[:args.limit]
    config = load_config(args.config or None)

    selected = [m for m in MODELS if args.model in ("all", m[0])]
    if not selected:
        raise SystemExit(f"unknown model '{args.model}'; choose from: "
                         f"all, {', '.join(m[0] for m in MODELS)}")

    print(f"{len(dilemmas)} dilemmas x {PROBES_PER_DILEMMA} probes "
          f"x {len(selected)} model(s)", flush=True)
    for friendly, api_id, temperature, effort in selected:
        print(f"=== {friendly} ({api_id}) temp={temperature} "
              f"effort={effort} ===", flush=True)
        run_model(friendly, api_id, temperature, effort,
                  dilemmas, config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        help="'all' or one friendly model name")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true",
                        help="discard existing responses and regenerate")
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke test: cap to the first N dilemmas "
                             "(all 28 conditions each)")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--dilemmas", default=DILEMMAS_PATH,
                        help="path to data/atp/dilemmas.json")
    parser.add_argument("--config", default="",
                        help="path to probe_config.json; default "
                             "data/atp/probe_config.json, built-in defaults "
                             "if absent")
    main(parser.parse_args())
