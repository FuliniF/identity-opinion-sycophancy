"""Single source of truth for the paper's subject-model roster.

Track-1 (anchor-event experiment). Before this module the roster was
copy-pasted into five scripts with three different values; everything now
imports from here so the roster is ONE editable list.

    generation : opinion_only.py            -> SUBJECT_MODELS (needs the
                                                api id / temperature / effort)
    judging    : evaluate_opinion_only.py    -> FRIENDLY_NAMES
                 evaluate_opinion_only_roleplay.py (re-exported)
    analysis   : analyze_opinion_only.py     -> FRIENDLY_NAMES + MODEL_SHORT
                 anova_figures.py            -> FRIENDLY_NAMES + MODEL_SHORT
                 reanalysis_mixedeffects.py  -> FRIENDLY_NAMES + MODEL_SHORT

ROSTER -- PINNED 2026-05-19 (mcc's decision, relayed by team-lead): 13 subject
models. Explicitly excluded: qwen3.6-27b and llama-4-scout (not subjects --
qwen3.6-27b is still a panel JUDGE), ernie-4.5-300b and nemotron-3-super
(dropped), and Mistral 3.1 (the Mistral subject is 3.2 only).

Role-play (as_characters) conditions exist for the 5 ORIGINAL models only
(see ROLEPLAY_MODELS) -- role-play analyses are a 5-model subset, not all 14.
"""

# (friendly name, OpenRouter / vLLM model id, temperature, reasoning effort).
# Temperature 0 where supported; o4-mini supports only temperature 1 and runs
# at the lowest reasoning effort. gpt-oss-120b is served on the H200 vLLM
# cluster under this id (works via GEN_BASE_URL).
SUBJECT_MODELS = [
    # --- the 5 original models (also the role-play subset) ---
    ("deepseek-v3.1", "deepseek/deepseek-chat-v3.1", 0.0, None),
    ("llama-3.3-70b-instruct", "meta-llama/llama-3.3-70b-instruct", 0.0, None),
    ("mistral-small-3.2-24b",
     "mistralai/mistral-small-3.2-24b-instruct", 0.0, None),
    ("o4-mini", "openai/o4-mini", 1.0, "low"),
    ("qwen3-32b", "qwen/qwen3-32b", 0.0, None),
    # --- 8 models added for the EMNLP 2026 revision ---
    ("gemma-4-31b-it", "google/gemma-4-31b-it", 0.0, None),
    ("glm-4.7", "z-ai/glm-4.7", 0.0, None),
    ("granite-4.1-8b", "ibm-granite/granite-4.1-8b", 0.0, None),
    ("kimi-k2.5", "moonshotai/kimi-k2.5", 0.0, None),
    ("mimo-v2-flash", "xiaomi/mimo-v2-flash", 0.0, None),
    ("nova-lite-v1", "amazon/nova-lite-v1", 0.0, None),
    ("phi-4", "microsoft/phi-4", 0.0, None),
    ("gpt-oss-120b", "openai/gpt-oss-120b", 0.0, None),
]

# friendly name -> short display label for figures and tables.
MODEL_SHORT = {
    "deepseek-v3.1": "DeepSeek-V3.1",
    "llama-3.3-70b-instruct": "Llama-3.3-70B",
    "mistral-small-3.2-24b": "Mistral-Small-3.2-24B",
    "o4-mini": "o4-mini",
    "qwen3-32b": "Qwen3-32B",
    "gemma-4-31b-it": "Gemma-4-31B",
    "glm-4.7": "GLM-4.7",
    "granite-4.1-8b": "Granite-4.1-8B",
    "kimi-k2.5": "Kimi-K2.5",
    "mimo-v2-flash": "MiMo-v2-Flash",
    "nova-lite-v1": "Nova-Lite",
    "phi-4": "Phi-4",
    "gpt-oss-120b": "GPT-OSS-120B",
}

# Ordered friendly-name list -- the full 14-model roster as plain strings.
FRIENDLY_NAMES = [m[0] for m in SUBJECT_MODELS]

# Role-play (as_characters) conditions were only run for the 5 original
# models; every role-play analysis is restricted to this subset.
ROLEPLAY_MODELS = [
    "deepseek-v3.1", "llama-3.3-70b-instruct", "mistral-small-3.2-24b",
    "o4-mini", "qwen3-32b",
]

# The 3 H200 panel judges (design doc section 3). qwen3.6-27b is a judge only
# -- it is NOT in the subject roster. The self-judge-artifact check therefore
# applies only to the judges that are also subjects: gpt-oss-120b, gemma-4-31b.
PANEL_JUDGES = ["gpt-oss-120b", "gemma-4-31b-it", "qwen3.6-27b"]
PANEL_JUDGE_SUBJECTS = ["gpt-oss-120b", "gemma-4-31b-it"]

# Sanity checks.
assert len(FRIENDLY_NAMES) == 13, f"expected 13 subjects, got {len(FRIENDLY_NAMES)}"
assert len(FRIENDLY_NAMES) == len(set(FRIENDLY_NAMES)), "duplicate friendly name"
assert set(FRIENDLY_NAMES) == set(MODEL_SHORT), \
    "SUBJECT_MODELS and MODEL_SHORT disagree"
assert set(ROLEPLAY_MODELS) <= set(FRIENDLY_NAMES), \
    "ROLEPLAY_MODELS must be a subset of the roster"
assert set(PANEL_JUDGE_SUBJECTS) <= set(FRIENDLY_NAMES), \
    "PANEL_JUDGE_SUBJECTS must be subjects in the roster"
