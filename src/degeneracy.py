"""Degeneracy classifier for anchor-event generation outputs.

The #17 audit detected only empty / missing / ERROR-sentinel cells. The
2026-05-19 re-audit found that NON-empty degenerate responses also pervade the
dataset; this module is the single source of truth for classifying them, used
by both ``rerun_failed_generation.py`` (cell repair) and the 14-model
re-audit.

Degeneracy classes (``classify`` returns exactly one):
  ok               a usable response
  empty            missing / empty / whitespace-only
  error_sentinel   the "ERROR WHEN GENERATING RESPONSE" sentinel
  gateway_refusal  a Chinese-host OpenRouter provider's moderation gateway
                   returned a fixed Chinese canned refusal
                   ("作为一个人工智能语言模型，我还没学习如何回答…")
  cjk_heavy        a mostly-CJK response where an English answer was expected
                   (other gateway refusals / a mid-response language switch)
  reasoning_leak   a reasoning model's planning channel contaminated
                   ``content`` -- instruction echo, repetition loops, tag spam
  truncation       a response cut off mid-sentence: short AND not ending in
                   terminal punctuation. A complete short answer (ends in
                   punctuation) is intentionally kept as ``ok``.
"""

import re

ERROR_SENTINEL = "ERROR WHEN GENERATING RESPONSE"
# A Chinese-host provider's moderation gateway returns this fixed boilerplate
# ("As an AI language model, I haven't learned how to answer this question…").
GATEWAY_REFUSAL = "作为一个人工智能语言模型"
TERMINAL_PUNCT = ".!?\"')’”"
SHORT_LEN = 250
CJK_HEAVY_FRACTION = 0.30

# A reasoning model leaking its planning channel into ``content``: the response
# opens with first-person planning meta-text instead of an opinion. Calibrated
# on the re-audit -- nemotron-3-super only; 0 hits in every other model.
_REASONING_TELL = re.compile(
    r"^\s*(We need to |We have to |We should |We are |"
    r"Let me |Let's |I need to |First,? I |"
    r"Okay,?\s+(the user|let'?s|let me|I |we )|"
    r"The user (wants|is asking|provided|writes|has ))",
    re.IGNORECASE)

CLASSES = ["ok", "empty", "error_sentinel", "gateway_refusal", "cjk_heavy",
           "reasoning_leak", "truncation"]
# Every class except "ok" is a defect that needs regeneration.
DEFECT_CLASSES = [c for c in CLASSES if c != "ok"]


def cjk_fraction(s):
    """Fraction of CJK-ideograph characters in a string."""
    if not s:
        return 0.0
    return sum(1 for c in s if "一" <= c <= "鿿") / len(s)


def classify(resp):
    """Return the degeneracy class of one response string (see module doc)."""
    if not isinstance(resp, str) or not resp.strip():
        return "empty"
    if ERROR_SENTINEL in resp:
        return "error_sentinel"
    s = resp.strip()
    if GATEWAY_REFUSAL in s:
        return "gateway_refusal"
    if cjk_fraction(s) > CJK_HEAVY_FRACTION:
        return "cjk_heavy"
    if _REASONING_TELL.match(s):
        return "reasoning_leak"
    if len(s) < SHORT_LEN and s[-1] not in TERMINAL_PUNCT:
        return "truncation"
    return "ok"


def is_degenerate(resp):
    """True if a response needs regenerating."""
    return classify(resp) != "ok"
