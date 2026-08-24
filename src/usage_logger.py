"""Append-only, thread-safe logger for per-call LLM token usage."""
import json, os, threading, time

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_DIR = os.path.join(os.path.dirname(_SRC_DIR), "logs")
_LOG_PATH = os.path.join(_LOG_DIR, "token_usage.jsonl")
_LOCK = threading.Lock()


def log_usage(model, stage, usage, *, field="", extra=None):
    """Append one record. `usage` is the SDK response.usage object (or None)."""
    if usage is None:
        return
    rec = {
        "ts": time.time(), "model": model, "stage": stage, "field": field,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        # OpenRouter reports the actual USD cost of the call on the usage
        # object; capturing it makes cost projection exact rather than
        # token-rate-estimated. Absent (e.g. plain OpenAI) -> 0.
        "cost": getattr(usage, "cost", 0) or 0,
    }
    if extra:
        rec.update(extra)
    os.makedirs(_LOG_DIR, exist_ok=True)
    with _LOCK:
        with open(_LOG_PATH, "a") as f:
            f.write(json.dumps(rec) + "\n")
