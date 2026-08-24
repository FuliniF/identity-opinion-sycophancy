"""End-to-end smoke test for the multi-endpoint judge pool + atomic claim.

Run with: .venv/bin/python src/test_judge_pool_smoke.py

What it checks:

  T1  parse_endpoints honours JUDGE_BASE_URLS, JUDGE_BASE_URL, and the
      legacy fallback.
  T2  JudgePool round-robins next_openai_client() across endpoints.
  T3  claim_output() is mutually exclusive across processes: of two
      workers racing on the same out_path, exactly one writes the file.
  T4  claim_output() recovers from a stale claim (PID dead).
  T5  Multi-endpoint integration: spin up TWO stub HTTP servers that
      mimic an OpenAI-compatible chat-completions endpoint and serve
      structured JSON; run the JudgePool against them with N>=4 calls
      and verify each server saw at least one request (round-robin
      reached every endpoint).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import judge_pool  # noqa: E402


# ---------------------------------------------------------------------------
# Test 1: parse_endpoints
# ---------------------------------------------------------------------------


def t1_parse_endpoints():
    # Save / restore env so we don't pollute the shell.
    saved = {k: os.environ.get(k) for k in
             ("JUDGE_BASE_URLS", "JUDGE_BASE_URL",
              "JUDGE_API_KEYS", "JUDGE_API_KEY")}
    try:
        for k in saved:
            os.environ.pop(k, None)

        # default
        eps = judge_pool.parse_endpoints(default_api_key="dummy")
        assert eps == [("https://openrouter.ai/api/v1", "dummy")], eps

        # single
        os.environ["JUDGE_BASE_URL"] = "http://a/v1"
        os.environ["JUDGE_API_KEY"] = "kA"
        eps = judge_pool.parse_endpoints()
        assert eps == [("http://a/v1", "kA")], eps

        # multi, broadcast key
        os.environ["JUDGE_BASE_URLS"] = "http://a/v1, http://b/v1 ,http://c/v1"
        eps = judge_pool.parse_endpoints()
        assert [u for u, _ in eps] == ["http://a/v1", "http://b/v1", "http://c/v1"], eps
        assert all(k == "kA" for _, k in eps), eps

        # multi, per-endpoint keys
        os.environ["JUDGE_API_KEYS"] = "k1,k2,k3"
        eps = judge_pool.parse_endpoints()
        assert [k for _, k in eps] == ["k1", "k2", "k3"], eps

        # mismatched key list -> pad with single
        os.environ["JUDGE_API_KEYS"] = "k1,k2"
        eps = judge_pool.parse_endpoints()
        assert [k for _, k in eps] == ["k1", "k2", "kA"], eps

        print("T1 parse_endpoints OK")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Test 2: round-robin
# ---------------------------------------------------------------------------


def t2_round_robin():
    endpoints = [("http://a/v1", "k"), ("http://b/v1", "k"),
                 ("http://c/v1", "k")]
    pool = judge_pool.JudgePool(endpoints)
    seen = [pool.next_openai_client().base_url for _ in range(9)]
    # OpenAI client normalises base_url; just check the host ordering.
    hosts = [str(u).split("//")[1].split("/")[0] for u in seen]
    expected = ["a", "b", "c"] * 3
    assert hosts == expected, hosts
    print("T2 round-robin OK")


# ---------------------------------------------------------------------------
# Test 3: claim_output mutual exclusion across processes
# ---------------------------------------------------------------------------


def _claim_worker(out_path, barrier_ms, log_path):
    """Sleep to align, then try to claim and write the file."""
    # Crude barrier: align workers to within ~50 ms.
    target = float(os.environ.get("CLAIM_RACE_T0", 0))
    while time.time() < target:
        time.sleep(0.001)
    import sys as _sys
    _sys.path.insert(0, HERE)
    import judge_pool as jp
    with jp.claim_output(out_path, label=f"smoke-{os.getpid()}") as owned:
        if not owned:
            with open(log_path, "a") as f:
                f.write(f"{os.getpid()} skip\n")
            return
        # Simulate work that takes a moment, so the *other* worker is
        # still in its claim attempt window.
        time.sleep(0.4)
        with open(out_path, "w") as f:
            json.dump({"by": os.getpid()}, f)
        with open(log_path, "a") as f:
            f.write(f"{os.getpid()} wrote\n")


def t3_claim_mutual_exclusion():
    tmp = tempfile.mkdtemp(prefix="judgepool_t3_")
    try:
        out_path = os.path.join(tmp, "result.json")
        log_path = os.path.join(tmp, "log.txt")
        os.environ["CLAIM_RACE_T0"] = str(time.time() + 0.3)
        N = 6
        procs = [mp.Process(target=_claim_worker,
                            args=(out_path, 50, log_path))
                 for _ in range(N)]
        for p in procs: p.start()
        for p in procs: p.join()

        # Exactly one worker should have written; others all skipped.
        with open(log_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        wrote = [ln for ln in lines if ln.endswith("wrote")]
        skip = [ln for ln in lines if ln.endswith("skip")]
        assert len(wrote) == 1, f"expected 1 writer, got {wrote} (all={lines})"
        assert len(skip) == N - 1, f"expected {N-1} skips, got {skip}"
        # File must exist with content from the writer.
        with open(out_path) as f:
            data = json.load(f)
        assert "by" in data
        # Claim dir must be gone.
        assert not os.path.exists(out_path + ".claim"), \
            "claim dir leaked after success"
        print(f"T3 claim mutual exclusion OK  ({wrote[0]})")
    finally:
        os.environ.pop("CLAIM_RACE_T0", None)
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: stale-claim recovery
# ---------------------------------------------------------------------------


def t4_stale_recovery():
    tmp = tempfile.mkdtemp(prefix="judgepool_t4_")
    try:
        out_path = os.path.join(tmp, "result.json")
        claim_dir = out_path + ".claim"
        # Hand-craft a stale claim: PID definitely dead (use 999999), and
        # started_at way in the past.
        os.makedirs(claim_dir)
        with open(os.path.join(claim_dir, "meta.json"), "w") as f:
            json.dump({
                "pid": 999999,
                "host": os.uname().nodename,
                "started_at": time.time() - 60 * 60,  # 1h ago
                "label": "stale",
            }, f)

        # Now a fresh worker should reclaim and successfully write.
        with judge_pool.claim_output(out_path, label="t4") as owned:
            assert owned, "should reclaim stale lock"
            with open(out_path, "w") as f:
                json.dump({"ok": True}, f)
        assert os.path.exists(out_path)
        assert not os.path.exists(claim_dir)
        print("T4 stale claim recovery OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 5: multi-endpoint integration with stub HTTP servers
# ---------------------------------------------------------------------------


# Per-server hit counter (keyed by server port).
_HIT_COUNTER = {}
_HIT_LOCK = threading.Lock()


class _StubHandler(BaseHTTPRequestHandler):
    # Quiet the noisy default access log.
    def log_message(self, fmt, *args):  # noqa: A003
        return

    def do_POST(self):  # noqa: N802  -- BaseHTTPRequestHandler convention
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)  # discard body
        with _HIT_LOCK:
            _HIT_COUNTER[self.server.server_port] = (
                _HIT_COUNTER.get(self.server.server_port, 0) + 1
            )
        # Minimal OpenAI-shaped response with structured JSON content.
        body = json.dumps({
            "id": "stub-1",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": json.dumps({"refusal": False})},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                      "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _spin_stub(port_holder):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    port_holder.append(srv.server_port)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv


def t5_multi_endpoint_integration():
    portsA, portsB = [], []
    srvA = _spin_stub(portsA)
    srvB = _spin_stub(portsB)
    try:
        urls = (f"http://127.0.0.1:{portsA[0]}/v1,"
                f"http://127.0.0.1:{portsB[0]}/v1")
        os.environ["JUDGE_BASE_URLS"] = urls
        os.environ["JUDGE_API_KEY"] = "stub"

        # Force re-parse + new pool (avoid any cached state).
        pool = judge_pool.build_pool()
        assert len(pool) == 2, len(pool)

        # Use the eval_opinion_only.judge() function so we exercise the
        # *actual* code path the real script uses.
        sys.path.insert(0, HERE)
        from evaluate_opinion_only import judge, SCHEMA_REFUSAL

        N = 8
        for _ in range(N):
            out = judge(pool, "stub-model", "hello", SCHEMA_REFUSAL,
                        max_tokens=16, reasoning_effort=None)
            assert out.get("refusal") is False, out

        a = _HIT_COUNTER.get(portsA[0], 0)
        b = _HIT_COUNTER.get(portsB[0], 0)
        assert a + b == N, f"expected {N} total hits, got A={a} B={b}"
        assert a >= 1 and b >= 1, \
            f"round-robin did not reach both endpoints: A={a} B={b}"
        # With even round-robin, A and B should each be N/2 (4).
        assert abs(a - b) <= 1, f"imbalanced: A={a} B={b}"
        print(f"T5 multi-endpoint integration OK  (A={a} B={b})")
    finally:
        srvA.shutdown()
        srvB.shutdown()
        for k in ("JUDGE_BASE_URLS", "JUDGE_API_KEY"):
            os.environ.pop(k, None)


# ---------------------------------------------------------------------------
# Test 6: backward compat -- single endpoint via JUDGE_BASE_URL still works
# ---------------------------------------------------------------------------


def t6_single_endpoint_compat():
    portsA = []
    srvA = _spin_stub(portsA)
    try:
        # Clear MULTI env, set SINGLE env.
        os.environ.pop("JUDGE_BASE_URLS", None)
        os.environ["JUDGE_BASE_URL"] = f"http://127.0.0.1:{portsA[0]}/v1"
        os.environ["JUDGE_API_KEY"] = "stub"

        pool = judge_pool.build_pool()
        assert len(pool) == 1, len(pool)

        sys.path.insert(0, HERE)
        from evaluate_opinion_only import judge, SCHEMA_REFUSAL

        before = _HIT_COUNTER.get(portsA[0], 0)
        for _ in range(3):
            out = judge(pool, "stub-model", "hello", SCHEMA_REFUSAL,
                        max_tokens=16, reasoning_effort=None)
            assert out.get("refusal") is False
        after = _HIT_COUNTER.get(portsA[0], 0)
        assert after - before == 3, after - before
        print(f"T6 single-endpoint backward-compat OK  (delta={after - before})")
    finally:
        srvA.shutdown()
        for k in ("JUDGE_BASE_URL", "JUDGE_API_KEY"):
            os.environ.pop(k, None)


if __name__ == "__main__":
    t1_parse_endpoints()
    t2_round_robin()
    t3_claim_mutual_exclusion()
    t4_stale_recovery()
    t5_multi_endpoint_integration()
    t6_single_endpoint_compat()
    print("ALL SMOKE TESTS PASSED")
