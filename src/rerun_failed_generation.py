"""Targeted re-run of failed cells in anchor-event generation outputs.

The generation scripts (baseline.py / stereotype.py / sycophancy.py) skip at
FILE granularity, so plain resume cannot repair a degenerate cell inside an
otherwise-complete response file. This tool scans a model's response files
cell-by-cell, regenerates ONLY the bad cells with the original prompt, and
writes each file back in place.

Covers all 8 anchor-event conditions: baseline / stereotyping / sycophancy /
opinion_only, each in default + role-play (``_as_characters``) form.

A cell is "bad" if ``degeneracy.classify`` flags it -- empty / ERROR-sentinel
/ provider-gateway refusal / cjk-heavy / reasoning-leak / truncation (see
degeneracy.py, the single source of truth shared with the re-audit).

``--provider`` pins OpenRouter routing to a fixed provider list with fallbacks
off, so re-runs can route around a moderation gateway. Regeneration retries
until the result is non-degenerate.

Prompt templates are imported from baseline.py / stereotype.py / sycophancy.py
so a regenerated cell is constructed identically to its siblings.

Run from the src/ directory.

  # report what would be re-run, no API calls:
  python rerun_failed_generation.py --model qwen3-32b --dry-run

  # repair qwen3-32b's gateway-refused / truncated cells, routing around the
  # Chinese moderation gateway (Alibaba / SiliconFlow):
  python rerun_failed_generation.py --model qwen3-32b \
      --provider DeepInfra,Nebius,Novita
"""

import argparse
import concurrent.futures as cf
import json
import os
import time
import types

import dotenv
from openai import OpenAI

import baseline as B
import opinion_only as OO
import opinion_only_roleplay as OOR
import stereotype as S
import sycophancy as SY
from degeneracy import is_degenerate as is_bad
from subject_models import SUBJECT_MODELS
from usage_logger import log_usage

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
SCEN_DIR = os.path.join(REPO_ROOT, "data", "scenarios", "final")
RESP_DIR = os.path.join(REPO_ROOT, "responses")
CHAR_DIR = os.path.join(REPO_ROOT, "data", "characters_description")
DOMAINS = ["diplomatic", "economy", "energy", "technology", "welfare"]

# Role-play uses 3 of the 9 typology personas, both as the assistant persona
# and as the user persona (mirrors character_to_test in generate_responses.py).
ROLEPLAY_PERSONAS = ["faith and flag conservatives", "outsider left",
                     "stressed sideliners"]

# friendly model-dir -> default (api_model, temperature), from the canonical
# roster in subject_models.py. Override on the CLI; an unknown model (e.g. an
# H200 served-model-name) requires --api-model.
MODEL_DEFAULTS = {m[0]: (m[1], m[2]) for m in SUBJECT_MODELS}

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

_ENV_KEY = dotenv.get_key(os.path.join(SRC_DIR, ".env"), "OPENROUTER_API_KEY")

# is_bad(resp) -> bool : a cell needs regenerating. Imported above from
# degeneracy.py (the single source of truth for every degeneracy mode --
# empty / ERROR-sentinel / gateway-refusal / cjk-heavy / reasoning-leak /
# truncation), so this tool and the re-audit always agree.


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def make_client(base_url, api_key):
    return OpenAI(
        base_url=base_url or "https://openrouter.ai/api/v1",
        api_key=api_key or _ENV_KEY or "EMPTY",
    )


def generate(gen, system_prompt, user_prompt):
    """One response, retried until non-degenerate. Returns the text, or an
    ERROR sentinel after ``gen.retries`` failures.

    ``gen`` is the config namespace built in main(); ``gen.provider_order``,
    when set, pins OpenRouter routing to those providers with fallbacks off
    (used to route around a moderation gateway).
    """
    extra = {}
    if gen.provider_order:
        extra["extra_body"] = {"provider": {
            "order": list(gen.provider_order), "allow_fallbacks": False}}
    last_err = None
    for attempt in range(gen.retries):
        try:
            resp = gen.client.chat.completions.create(
                model=gen.api_model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=gen.temperature,
                max_tokens=gen.max_tokens,
                **extra,
            )
            if not resp.choices:
                raise RuntimeError(f"no choices: {getattr(resp, 'error', '')}")
            text = resp.choices[0].message.content
            log_usage(gen.api_model, "rerun_failed", getattr(resp, "usage", None))
            if is_bad(text):
                # empty / gateway refusal / truncation -> retry (a pinned
                # provider should not hit the gateway, but be safe).
                raise RuntimeError("degenerate response")
            return text.strip()
        except Exception as e:  # noqa: BLE001 - broad on purpose, we retry
            last_err = e
            time.sleep(2 * (attempt + 1))
    return f"ERROR WHEN GENERATING RESPONSE: {last_err}"


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def load_chars():
    chars = {}
    for fn in os.listdir(CHAR_DIR):
        if fn.endswith(".txt"):
            with open(os.path.join(CHAR_DIR, fn)) as f:
                chars[fn[:-4].replace("_", " ")] = f.read()
    return chars


def load_anchors(domain):
    with open(os.path.join(SCEN_DIR, domain, "anchor_events.json")) as f:
        return json.load(f)


def load_narratives(domain):
    with open(os.path.join(SCEN_DIR, domain, "new_narratives.json")) as f:
        return json.load(f)


def load_oo_narratives(domain):
    """Opinion-only narratives, keyed by item id (see opinion_only.py)."""
    with open(os.path.join(SCEN_DIR, domain,
                           "narratives_opinion_only.json")) as f:
        return {n["id"]: n for n in json.load(f)}


def load_file(path, is_jsonl):
    if is_jsonl:
        with open(path) as f:
            return [json.loads(ln) for ln in f if ln.strip()]
    with open(path) as f:
        return json.load(f)


def write_file(path, data, is_jsonl):
    """Write back via a temp file + atomic replace, matching the original
    scripts' formatting (jsonl: one compact object per line; json: indent=4)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        if is_jsonl:
            for row in data:
                f.write(json.dumps(row) + "\n")
        else:
            json.dump(data, f, indent=4)
    os.replace(tmp, path)


def find_file(model, domain, condition):
    """Locate the response file by suffix (filename prefixes vary per model)."""
    ddir = os.path.join(RESP_DIR, model, domain)
    if not os.path.isdir(ddir):
        return None
    suffix, _ = COND_SUFFIX[condition]
    want_rp = "_as_characters_" in suffix
    for fn in os.listdir(ddir):
        if fn.endswith(suffix) and ("_as_characters_" in fn) == want_rp:
            return os.path.join(ddir, fn)
    return None


# --------------------------------------------------------------------------
# Per-condition bad-cell collection. Each job is (locator, system, user).
# --------------------------------------------------------------------------
def collect_jobs(condition, data, anchors, narratives, chars, characters):
    jobs = []
    warnings = []

    if condition == "baseline":
        for i, row in enumerate(data):
            if is_bad(row.get("response")):
                user = B.USER_PROMPT.format(
                    news_article=anchors[row["id"]]["objective_anchor"])
                jobs.append((("row", i), B.SYS_PROMPT, user))

    elif condition == "baseline_rp":
        for i, row in enumerate(data):
            if is_bad(row.get("response")):
                persona = row["responding_as"]
                sysp = B.SYS_PROMPT_AS_CHARACTER.format(
                    assistant_character=persona,
                    assistant_description=chars[persona])
                user = B.USER_PROMPT.format(
                    news_article=anchors[row["id"]]["objective_anchor"])
                jobs.append((("row", i), sysp, user))

    elif condition == "stereotyping":
        for i, item in enumerate(data):
            for c in characters:
                if is_bad(item.get(c)):
                    sysp = S.SYS_PROMPT.format(user_name=c,
                                               user_description=chars[c])
                    user = S.USER_PROMPT.format(
                        news_article=anchors[i]["objective_anchor"])
                    jobs.append((("item_char", i, c), sysp, user))

    elif condition == "stereotyping_rp":
        for ri, rec in enumerate(data):
            assistant = rec["responding_as"]
            for c in ROLEPLAY_PERSONAS:
                if is_bad(rec.get(c)):
                    sysp = S.SYS_PROMPT_AS_CHARACTER.format(
                        user_name=c, user_description=chars[c],
                        assistant_character=assistant,
                        assistant_description=chars[assistant])
                    user = S.USER_PROMPT.format(
                        news_article=anchors[rec["id"]]["objective_anchor"])
                    jobs.append((("rec_char", ri, c), sysp, user))

    elif condition == "sycophancy":
        for i, item in enumerate(data):
            for c in characters:
                cell = item.get(c)
                for side in ("A", "B"):
                    resp = cell.get(f"response_{side}") if isinstance(
                        cell, dict) else None
                    if is_bad(resp):
                        narr = narratives[i][c.replace(" ", "_")]
                        sysp = SY.SYS_PROMPT.format(user_name=c,
                                                    user_description=chars[c])
                        user = SY.USER_PROMPT.format(
                            news_article=anchors[i]["objective_anchor"],
                            user_opinion_question=narr[f"narrative_{side}"])
                        jobs.append((("item_char_side", i, c, side),
                                     sysp, user))

    elif condition == "sycophancy_rp":
        for ri, rec in enumerate(data):
            assistant = rec["responding_as"]
            for c in ROLEPLAY_PERSONAS:
                cell = rec.get(c)
                for side in ("A", "B"):
                    resp = cell.get(f"response_{side}") if isinstance(
                        cell, dict) else None
                    if is_bad(resp):
                        narr = narratives[rec["id"]][c.replace(" ", "_")]
                        sysp = SY.SYS_PROMPT_AS_CHARACTER.format(
                            user_name=c, user_description=chars[c],
                            assistant_character=assistant,
                            assistant_description=chars[assistant])
                        user = SY.USER_PROMPT.format(
                            news_article=anchors[rec["id"]]["objective_anchor"],
                            user_opinion_question=narr[f"narrative_{side}"])
                        jobs.append((("rec_char_side", ri, c, side),
                                     sysp, user))

    elif condition == "opinion_only":
        for i, item in enumerate(data):
            narr = narratives[item["id"]]
            for side in ("A", "B"):
                if is_bad(item.get(f"response_{side}")):
                    user = OO.USER_PROMPT.format(
                        news_article=anchors[item["id"]]["objective_anchor"],
                        user_opinion_question=narr[f"narrative_{side}"])
                    jobs.append((("oo_side", i, side), OO.SYS_PROMPT, user))

    elif condition == "opinion_only_rp":
        for ri, rec in enumerate(data):
            persona = rec["responding_as"]
            narr = narratives[rec["id"]]
            for side in ("A", "B"):
                if is_bad(rec.get(f"response_{side}")):
                    sysp = OOR.SYS_PROMPT.format(
                        assistant_character=persona,
                        assistant_description=chars[persona])
                    user = OOR.USER_PROMPT.format(
                        news_article=anchors[rec["id"]]["objective_anchor"],
                        user_opinion_question=narr[f"narrative_{side}"])
                    jobs.append((("oor_side", ri, side), sysp, user))

    # Structural sanity: this tool repairs in-file cells, not missing records.
    if (condition in ("stereotyping_rp", "sycophancy_rp", "opinion_only_rp")
            and len(data) != 270):
        warnings.append(f"{len(data)} records (expected 270) -- a whole "
                        f"record is missing; use delete+regenerate for this "
                        f"file, not this tool")
    return jobs, warnings


def apply_result(data, locator, text):
    kind = locator[0]
    if kind == "row":
        data[locator[1]]["response"] = text
    elif kind == "item_char":
        _, i, c = locator
        data[i][c] = text
    elif kind == "rec_char":
        _, ri, c = locator
        data[ri][c] = text
    elif kind == "item_char_side":
        _, i, c, side = locator
        data[i].setdefault(c, {})[f"response_{side}"] = text
    elif kind == "rec_char_side":
        _, ri, c, side = locator
        data[ri].setdefault(c, {})[f"response_{side}"] = text
    elif kind == "oo_side":
        _, i, side = locator
        data[i][f"response_{side}"] = text
    elif kind == "oor_side":
        _, ri, side = locator
        data[ri][f"response_{side}"] = text
    else:
        raise ValueError(f"unknown locator kind {kind!r}")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def repair_file(model, condition, domain, chars, characters, gen, args):
    """Repair one (model, condition, domain) file. Returns a stats dict, or
    None if the file is absent (a whole missing condition/domain is not this
    tool's job -- use the normal generation scripts for that)."""
    path = find_file(model, domain, condition)
    if path is None:
        return None
    is_jsonl = COND_SUFFIX[condition][1]
    data = load_file(path, is_jsonl)
    anchors = load_anchors(domain)
    if condition.startswith("sycophancy"):
        narratives = load_narratives(domain)
    elif condition.startswith("opinion_only"):
        narratives = load_oo_narratives(domain)
    else:
        narratives = None
    jobs, warnings = collect_jobs(condition, data, anchors, narratives,
                                  chars, characters)
    for w in warnings:
        print(f"  [WARN] {condition}/{domain}: {w}", flush=True)
    if not jobs:
        return {"path": path, "bad": 0, "regen": 0, "still_bad": 0}
    if args.dry_run:
        return {"path": path, "bad": len(jobs), "regen": 0, "still_bad": 0}

    def run(job):
        locator, sysp, userp = job
        return locator, generate(gen, sysp, userp)

    results = {}
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for locator, text in ex.map(run, jobs):
            results[locator] = text

    still_bad = 0
    for locator, text in results.items():
        if is_bad(text):
            still_bad += 1  # leave the original (bad) cell; do not overwrite
        else:
            apply_result(data, locator, text)
    write_file(path, data, is_jsonl)
    return {"path": path, "bad": len(jobs), "regen": len(jobs) - still_bad,
            "still_bad": still_bad}


def main(args):
    chars = load_chars()
    characters = sorted(chars)  # the 9 typology personas (default conditions)

    api_model, temperature = MODEL_DEFAULTS.get(args.model, (None, 0.0))
    if args.api_model:
        api_model = args.api_model
    if api_model is None:
        raise SystemExit(f"no default api-model for {args.model!r}; "
                         f"pass --api-model")
    if args.temperature is not None:
        temperature = args.temperature
    provider_order = ([p.strip() for p in args.provider.split(",") if p.strip()]
                      if args.provider else None)

    gen = types.SimpleNamespace(
        client=None if args.dry_run else make_client(
            args.base_url or os.environ.get("GEN_BASE_URL"),
            args.api_key or os.environ.get("GEN_API_KEY")),
        api_model=api_model, temperature=temperature,
        max_tokens=args.max_tokens, provider_order=provider_order,
        retries=args.retries)

    conditions = (CONDITIONS if args.conditions == "all"
                  else args.conditions.split(","))
    domains = DOMAINS if args.domains == "all" else args.domains.split(",")

    mode = ("DRY RUN -- no API calls" if args.dry_run
            else f"via {api_model}"
                 + (f" pinned->{'/'.join(provider_order)}"
                    if provider_order else ""))
    print(f"=== rerun_failed_generation: {args.model} ({mode}) ===", flush=True)
    tot_bad = tot_regen = tot_still = 0
    for condition in conditions:
        for domain in domains:
            stats = repair_file(args.model, condition, domain, chars,
                                characters, gen, args)
            if stats is None or stats["bad"] == 0:
                continue
            if args.dry_run:
                print(f"  [{condition}/{domain}] {stats['bad']} bad cells "
                      f"-- would re-run", flush=True)
            else:
                print(f"  [{condition}/{domain}] {stats['bad']} bad cells "
                      f"-- regenerated {stats['regen']}, "
                      f"{stats['still_bad']} still failed", flush=True)
            tot_bad += stats["bad"]
            tot_regen += stats["regen"]
            tot_still += stats["still_bad"]

    print(f"=== {args.model}: {tot_bad} bad cells"
          + (" (dry run)" if args.dry_run
             else f", {tot_regen} repaired, {tot_still} still failed")
          + " ===", flush=True)
    if tot_still:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="response-dir name, e.g. deepseek-v3.1 / o4-mini "
                             "/ qwen3-32b")
    parser.add_argument("--conditions", default="all",
                        help=f"'all' or comma list of: {','.join(CONDITIONS)}")
    parser.add_argument("--domains", default="all",
                        help="'all' or comma list of domains")
    parser.add_argument("--api-model", default="",
                        help="model id sent to the endpoint "
                             "(default: per-model; required for unknown models "
                             "or H200 served-model-names)")
    parser.add_argument("--provider", default="",
                        help="comma list of OpenRouter providers to pin "
                             "routing to (fallbacks off) -- e.g. "
                             "DeepInfra,Nebius,Novita to route around a "
                             "moderation gateway")
    parser.add_argument("--base-url", default="",
                        help="endpoint base URL (default: $GEN_BASE_URL, "
                             "else OpenRouter)")
    parser.add_argument("--api-key", default="",
                        help="API key (default: $GEN_API_KEY, else .env)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="override the per-model default temperature")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="completion budget (matches the original run)")
    parser.add_argument("--retries", type=int, default=6,
                        help="attempts per cell until a non-degenerate "
                             "response (lower for temp-0 deterministic models "
                             "where retries add little)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true",
                        help="report bad cells without regenerating")
    main(parser.parse_args())
