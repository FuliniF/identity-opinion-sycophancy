"""Multi-endpoint judge pool + atomic per-file claim.

This module gives the three judge entry points (evaluate.py,
evaluate_opinion_only.py, evaluate_opinion_only_roleplay.py) two things:

1. Round-robin over N OpenAI-compatible judge endpoints.

   Endpoints are passed via the env var ``JUDGE_BASE_URLS`` (comma
   separated). For backward compat, if only ``JUDGE_BASE_URL`` is set
   the pool falls back to that single endpoint, and if neither is set
   the pool falls back to OpenRouter's public endpoint (the original
   default). The same ``JUDGE_API_KEY`` is used for every endpoint
   unless ``JUDGE_API_KEYS`` (comma separated, same length) is set.

   The pool is thread-safe (an ``itertools.count`` + modulo gives a
   lock-free round-robin index). Each endpoint has its own OpenAI
   client instance, so the underlying HTTP connection pools are
   isolated.

2. Atomic per-file claim. Two parallel evaluate.py processes covering
   overlapping shards would otherwise both see the result file
   missing, both judge it, and both write it. The claim is a sibling
   ``<result>.claim/`` directory created by ``mkdir`` (atomic on
   POSIX): only one worker can win. A small ``meta.json`` inside it
   records the owning PID + start timestamp, so a crashed worker's
   stale lock can be recovered (default >30 min old, or PID dead).

The pool / claim is deliberately a thin, opt-in helper -- the judge
scripts gain a couple of lines (``pool = build_pool(...)`` + ``with
claim_output(path):``) and the rest of their code is unchanged.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from typing import Iterator, List, Optional

from openai import OpenAI

# Reuse the bias/refusal/sycophancy schemas + retry logic from the existing
# OpenRouterClient so this module stays a *thin* facade.
from client import OpenRouterClient


# ---------------------------------------------------------------------------
# Endpoint discovery
# ---------------------------------------------------------------------------

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def parse_endpoints(
    base_urls_env: str = "JUDGE_BASE_URLS",
    base_url_env: str = "JUDGE_BASE_URL",
    api_keys_env: str = "JUDGE_API_KEYS",
    api_key_env: str = "JUDGE_API_KEY",
    default_base_url: Optional[str] = DEFAULT_BASE_URL,
    default_api_key: Optional[str] = None,
) -> List[tuple]:
    """Return a list of (base_url, api_key) tuples to round-robin over.

    Precedence:
      * JUDGE_BASE_URLS (comma separated) -> list, paired with
        JUDGE_API_KEYS (same length) if set, otherwise every endpoint
        reuses JUDGE_API_KEY / default_api_key.
      * JUDGE_BASE_URL (single) -> 1-element list (legacy).
      * neither -> [default_base_url] with default_api_key.

    Empty strings inside the comma list are filtered out so trailing
    commas don't add a phantom endpoint.
    """
    urls_raw = os.environ.get(base_urls_env, "").strip()
    keys_raw = os.environ.get(api_keys_env, "").strip()
    single_url = os.environ.get(base_url_env, "").strip()
    single_key = os.environ.get(api_key_env, "").strip() or default_api_key

    if urls_raw:
        urls = [u.strip() for u in urls_raw.split(",") if u.strip()]
    elif single_url:
        urls = [single_url]
    else:
        urls = [default_base_url] if default_base_url else []

    if keys_raw:
        keys = [k.strip() for k in keys_raw.split(",")]
        # Pad or truncate to len(urls): if mismatched, fall back to
        # broadcasting the single key for unspecified slots.
        if len(keys) < len(urls):
            keys = keys + [single_key] * (len(urls) - len(keys))
        else:
            keys = keys[: len(urls)]
    else:
        keys = [single_key] * len(urls)

    return list(zip(urls, keys))


# ---------------------------------------------------------------------------
# Round-robin judge pool
# ---------------------------------------------------------------------------


class JudgePool:
    """Thread-safe round-robin pool of OpenAI-compatible clients.

    Two flavours are exposed because the existing judge scripts split
    along this line:

      * ``next_openai_client()`` returns a raw ``openai.OpenAI`` (used
        by evaluate_opinion_only.py / _roleplay.py which call the
        chat-completions API directly).

      * ``next_router_client(model, ...)`` returns an
        ``OpenRouterClient`` (used by evaluate.py via
        ``client.generate_with_schema``). One ``OpenRouterClient`` is
        cached per endpoint+model+effort+max_length tuple so we don't
        rebuild the OpenAI wrapper on every call.
    """

    def __init__(self, endpoints: List[tuple]):
        if not endpoints:
            raise ValueError("JudgePool requires >=1 endpoint")
        self._endpoints = list(endpoints)
        # Pre-build one OpenAI client per endpoint so the HTTP connection
        # pools are warm and we don't pay client-construction cost on
        # the hot path.
        self._openai_clients = [
            OpenAI(base_url=url, api_key=key) for url, key in self._endpoints
        ]
        # OpenRouterClient cache keyed by (idx, model, effort, max_length).
        self._router_cache: dict = {}
        self._router_lock = threading.Lock()
        # itertools.count is the lock-free counter; modulo at use site.
        self._counter = itertools.count()

    def __len__(self) -> int:
        return len(self._endpoints)

    @property
    def endpoints(self) -> List[tuple]:
        return list(self._endpoints)

    def _next_index(self) -> int:
        return next(self._counter) % len(self._endpoints)

    def next_openai_client(self) -> OpenAI:
        idx = self._next_index()
        return self._openai_clients[idx]

    def next_router_client(
        self,
        model: str,
        temperature: float = 0.0,
        effort: str = "none",
        max_length: int = 4096,
    ) -> OpenRouterClient:
        idx = self._next_index()
        url, key = self._endpoints[idx]
        cache_key = (idx, model, effort, max_length, temperature)
        client = self._router_cache.get(cache_key)
        if client is None:
            with self._router_lock:
                client = self._router_cache.get(cache_key)
                if client is None:
                    client = OpenRouterClient(
                        model=model,
                        temperature=temperature,
                        max_length=max_length,
                        effort=effort,
                        base_url=url,
                        api_key=key,
                    )
                    self._router_cache[cache_key] = client
        return client


def build_pool(
    default_base_url: Optional[str] = DEFAULT_BASE_URL,
    default_api_key: Optional[str] = None,
) -> JudgePool:
    """Convenience: parse env, instantiate the pool, print a summary."""
    endpoints = parse_endpoints(
        default_base_url=default_base_url, default_api_key=default_api_key
    )
    pool = JudgePool(endpoints)
    # Mask the key on the summary line -- the URL is what matters for ops.
    for i, (url, _key) in enumerate(endpoints):
        print(f"[judge_pool] endpoint[{i}] = {url}", flush=True)
    return pool


# ---------------------------------------------------------------------------
# Atomic per-file claim
# ---------------------------------------------------------------------------

# How old (seconds) a claim must be before we consider it stale. 30 min is
# longer than the slowest per-file judge run we have (gemma-4-31b ~3-4 min
# per field) so a *live* worker is never preempted, but a crashed worker's
# lock doesn't persist forever.
STALE_CLAIM_SECONDS = 30 * 60


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it -> count as alive.
        return True
    except OSError:
        return False
    return True


def _read_claim_meta(claim_dir: str) -> dict:
    meta_path = os.path.join(claim_dir, "meta.json")
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _claim_is_stale(claim_dir: str, max_age: float = STALE_CLAIM_SECONDS) -> bool:
    """A claim is stale if (a) its owning PID is no longer alive, or
    (b) the claim is older than ``max_age`` seconds. We use the meta
    file when present and fall back to the dir's mtime when meta is
    unreadable."""
    meta = _read_claim_meta(claim_dir)
    pid = int(meta.get("pid", 0) or 0)
    started = float(meta.get("started_at", 0) or 0)

    # If we have a PID and it's a different host's PID we conservatively
    # rely on age alone.
    host = meta.get("host", "")
    same_host = (not host) or host == os.uname().nodename

    if same_host and pid and not _pid_alive(pid):
        return True

    if not started:
        # Fall back to dir mtime.
        try:
            started = os.path.getmtime(claim_dir)
        except OSError:
            return False  # can't tell; be conservative

    return (time.time() - started) > max_age


def _try_acquire(claim_dir: str, meta: dict) -> bool:
    """Try to mkdir(claim_dir) and write meta. Return True on success."""
    try:
        os.makedirs(claim_dir)  # not exist_ok -> raises if someone else won
    except FileExistsError:
        return False
    # We won. Drop the meta.json non-atomically; readers tolerate missing/
    # truncated meta (treated as old-style claim with no PID info).
    try:
        with open(os.path.join(claim_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
    except OSError:
        pass
    return True


@contextmanager
def claim_output(
    result_path: str,
    label: str = "",
    stale_seconds: float = STALE_CLAIM_SECONDS,
) -> Iterator[bool]:
    """Atomically claim ``result_path`` for this worker.

    Yields True if this worker owns the shard and should run; yields
    False if another worker either (a) already finished it (the result
    file exists) or (b) currently owns the claim and is still alive.

    On success the claim dir is removed in the finally block. If the
    body raises *before* writing the result file, the claim is also
    removed (so the next worker can retry rather than hitting a
    permanent lock).

    Usage::

        with claim_output(out_path, label=f"{model}/{field}") as owned:
            if not owned:
                return
            ... compute ...
            with open(out_path, "w") as f: json.dump(records, f)
    """
    # Fast path: file already present, nothing to claim.
    if os.path.exists(result_path):
        yield False
        return

    claim_dir = result_path + ".claim"
    meta = {
        "pid": os.getpid(),
        "host": os.uname().nodename,
        "started_at": time.time(),
        "label": label,
        "result_path": result_path,
    }

    acquired = _try_acquire(claim_dir, meta)
    if not acquired:
        # Someone else owns it. If their claim is stale, try to steal.
        if os.path.isdir(claim_dir) and _claim_is_stale(claim_dir, stale_seconds):
            print(
                f"[claim] stale claim at {claim_dir} -> reclaiming",
                flush=True,
            )
            try:
                shutil.rmtree(claim_dir)
            except OSError:
                pass
            acquired = _try_acquire(claim_dir, meta)

    if not acquired:
        # Lost the race fair and square. Skip.
        if label:
            print(f"[claim] busy -> skip {label}", flush=True)
        yield False
        return

    try:
        yield True
    finally:
        # Always drop the claim, even on exception. If the body wrote the
        # result file successfully, future workers will hit the fast path
        # (file exists). If the body failed, the claim is gone so they
        # can retry.
        try:
            shutil.rmtree(claim_dir)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Convenience for scripts that want to inspect their environment.
# ---------------------------------------------------------------------------


def summarize_env() -> str:
    endpoints = parse_endpoints()
    return "; ".join(f"{i}:{url}" for i, (url, _) in enumerate(endpoints))


if __name__ == "__main__":
    # Tiny self-check / smoke when run directly.
    print("Endpoints:", summarize_env())
    pool = build_pool()
    print("Pool size:", len(pool))
    for _ in range(min(5, 2 * len(pool))):
        print(" next ->", pool.next_openai_client().base_url)
