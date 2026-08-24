import json
import os
import time

import dotenv
import requests
from openai import OpenAI

from usage_logger import log_usage

dotenv.load_dotenv()

SCENARIO_SYNTHESIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "scenario_synthesis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "objective_anchor": {"type": "string"},
                "mapping_A": {"type": "string"},
                "mapping_B": {"type": "string"},
            },
            "required": ["objective_anchor", "mapping_A", "mapping_B"],
            "additionalProperties": False,
        },
    },
}

BIAS_EVALUATION_SCHEMA_GLOBAL = {
    "type": "json_schema",
    "json_schema": {
        "name": "bias_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "overall_stance_score": {"type": "number"},
                "fact_usage_balance": {
                    "type": "object",
                    "properties": {
                        "mentions_fact_a": {"type": "boolean"},
                        "mentions_fact_b": {"type": "boolean"},
                        "imbalance_description": {"type": "string"},
                    },
                    "required": [
                        "mentions_fact_a",
                        "mentions_fact_b",
                        "imbalance_description",
                    ],
                    "additionalProperties": False,
                },
                "structural_bias": {"type": "boolean"},
                "framing_bias": {"type": "boolean"},
                "selection_bias": {"type": "boolean"},
                "normative_bias": {"type": "boolean"},
            },
            "required": [
                "reasoning",
                "overall_stance_score",
                "fact_usage_balance",
                "structural_bias",
                "framing_bias",
                "selection_bias",
                "normative_bias",
            ],
            "additionalProperties": False,
        },
    },
}

BIAS_EVALUATION_SCHEMA_GLOBAL_SYCOPHANCY = {
    "type": "json_schema",
    "json_schema": {
        "name": "bias_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "overall_stance_score": {"type": "number"},
                "fact_usage_balance": {
                    "type": "object",
                    "properties": {
                        "mentions_fact_a": {"type": "boolean"},
                        "mentions_fact_b": {"type": "boolean"},
                        "imbalance_description": {"type": "string"},
                    },
                    "required": [
                        "mentions_fact_a",
                        "mentions_fact_b",
                        "imbalance_description",
                    ],
                    "additionalProperties": False,
                },
                "structural_bias": {"type": "boolean"},
                "framing_bias": {"type": "boolean"},
                "selection_bias": {"type": "boolean"},
                "normative_bias": {"type": "boolean"},
                "sycophantic_alignment": {"type": "boolean"},
            },
            "required": [
                "reasoning",
                "overall_stance_score",
                "fact_usage_balance",
                "structural_bias",
                "framing_bias",
                "selection_bias",
                "normative_bias",
                "sycophantic_alignment",
            ],
            "additionalProperties": False,
        },
    },
}

BIAS_EVALUATION_SCHEMA_REFUSAL = {
    "type": "json_schema",
    "json_schema": {
        "name": "bias_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "refusal": {"type": "boolean"},
            },
            "required": [
                "refusal",
            ],
            "additionalProperties": False,
        },
    },
}

# Anchor-free ATP-redesign judge schema (design doc section 4). The probe has
# no anchor event, so there is no fixed evidence set: `fact_usage_balance` is
# dropped and `selection_bias` is redefined judgmentally (does the response
# selectively foreground considerations favouring one side). `overall_stance_
# score` + `structural_bias` / `framing_bias` / `normative_bias` survive
# unchanged. The pilot schemas above are kept intact for the old scripts.
BIAS_EVALUATION_SCHEMA_ATP = {
    "type": "json_schema",
    "json_schema": {
        "name": "bias_evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string"},
                "overall_stance_score": {"type": "number"},
                "structural_bias": {"type": "boolean"},
                "framing_bias": {"type": "boolean"},
                "selection_bias": {"type": "boolean"},
                "normative_bias": {"type": "boolean"},
            },
            "required": [
                "reasoning",
                "overall_stance_score",
                "structural_bias",
                "framing_bias",
                "selection_bias",
                "normative_bias",
            ],
            "additionalProperties": False,
        },
    },
}


class OpenRouterClient:
    def __init__(
        self,
        model: str,
        temperature: float,
        effort: str = "none",
        max_length: int = 4096,
        base_url=None,
        api_key=None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_length = max_length
        self.effort = effort

        # base_url/api_key default to OpenRouter; the judge scripts pass an
        # override (e.g. the self-hosted vLLM endpoint) so generation stays
        # on OpenRouter untouched.
        self.client = OpenAI(
            base_url=base_url or "https://openrouter.ai/api/v1",
            api_key=api_key or dotenv.get_key(".env", "OPENROUTER_API_KEY"),
        )

        self.SCENARIO_SYNTHESIS_SCHEMA = SCENARIO_SYNTHESIS_SCHEMA
        self.BIAS_EVALUATION_SCHEMA_GLOBAL = BIAS_EVALUATION_SCHEMA_GLOBAL
        self.BIAS_EVALUATION_SCHEMA_GLOBAL_SYCOPHANCY = (
            BIAS_EVALUATION_SCHEMA_GLOBAL_SYCOPHANCY
        )
        self.BIAS_EVALUATION_SCHEMA_REFUSAL = BIAS_EVALUATION_SCHEMA_REFUSAL
        self.BIAS_EVALUATION_SCHEMA_ATP = BIAS_EVALUATION_SCHEMA_ATP

    def generate(self, system_prompt: str, user_prompt: str,
                 stage: str = "unknown", max_retries: int = 5) -> str:
        # Retries with backoff: the identity-arm scripts call this under a
        # thread pool, so a transient 429/network error must not abort the
        # whole run. On persistent failure we return the sentinel string
        # ("ERROR WHEN GENERATING RESPONSE: ...") that the verification grep
        # looks for, rather than raising.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_err = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_length,
                    # extra_body={"reasoning": {"effort": self.effort}},
                )
                log_usage(self.model, stage, getattr(response, "usage", None))
                if not response.choices:
                    raise RuntimeError(
                        f"no choices: {getattr(response, 'error', '')}")
                text = response.choices[0].message.content
                if not text or not text.strip():
                    raise RuntimeError("empty content")
                return text
            except Exception as e:  # noqa: BLE001 - broad on purpose, we retry
                last_err = e
                time.sleep(2 * (attempt + 1))
        return f"ERROR WHEN GENERATING RESPONSE: {last_err}"

    def generate_with_schema(
        self, system_prompt: str, user_prompt: str, schema_type: str,
        stage: str = "unknown", max_retries: int = 5
    ) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if schema_type == "global":
            json_schema = self.BIAS_EVALUATION_SCHEMA_GLOBAL
        elif schema_type == "global_sycophancy":
            json_schema = self.BIAS_EVALUATION_SCHEMA_GLOBAL_SYCOPHANCY
        elif schema_type == "anchor":
            json_schema = self.SCENARIO_SYNTHESIS_SCHEMA
        elif schema_type == "refusal":
            json_schema = self.BIAS_EVALUATION_SCHEMA_REFUSAL
        elif schema_type == "atp":
            json_schema = self.BIAS_EVALUATION_SCHEMA_ATP
        else:
            raise ValueError(f"Unsupported schema type: {schema_type}")

        # Reasoning judges (the self-hosted gpt-oss endpoint) need an explicit
        # effort hint; skipped when effort is unset/"none" (OpenRouter default).
        extra = {}
        if self.effort and self.effort not in ("none",):
            extra = {"extra_body": {"reasoning_effort": self.effort}}

        # Retries with backoff: judge calls run under a thread pool, and a
        # transient error or a malformed-JSON completion must not abort the run.
        last_err = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_length,
                    response_format=json_schema,
                    **extra,
                )
                log_usage(self.model, stage, getattr(response, "usage", None))
                # Self-hosted vLLM judge endpoints served non-thinking but
                # with a reasoning parser still active mis-file the structured
                # answer into the reasoning channel, leaving content null. The
                # guided-decoding JSON is intact there -- fall back to it.
                # content-first, so gpt-oss / gemma are unaffected.
                _m = response.choices[0].message.model_dump()
                _text = (_m.get("content") or _m.get("reasoning_content")
                         or _m.get("reasoning"))
                return json.loads(_text)
            except Exception as e:  # noqa: BLE001 - broad on purpose, we retry
                last_err = e
                time.sleep(2 * (attempt + 1))
        print(f"generate_with_schema failed after {max_retries} retries: "
              f"{last_err}")
        return {"error": "Failed to parse JSON", "raw": str(last_err)}


class OpenAIClient:
    def __init__(
        self,
        model: str,
        temperature: float,
        effort: str = "none",
        max_length: int = 4096,
    ):
        self.model = model
        self.temperature = temperature
        self.max_length = max_length
        self.effort = effort

        self.client = OpenAI(
            api_key=dotenv.get_key(".env", "OPENAI_API_KEY"),
        )

        self.SCENARIO_SYNTHESIS_SCHEMA = SCENARIO_SYNTHESIS_SCHEMA
        self.BIAS_EVALUATION_SCHEMA_GLOBAL = BIAS_EVALUATION_SCHEMA_GLOBAL
        self.BIAS_EVALUATION_SCHEMA_GLOBAL_SYCOPHANCY = (
            BIAS_EVALUATION_SCHEMA_GLOBAL_SYCOPHANCY
        )
        self.BIAS_EVALUATION_SCHEMA_ATP = BIAS_EVALUATION_SCHEMA_ATP

    def generate(self, system_prompt: str, user_prompt: str,
                 stage: str = "unknown") -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            # temperature=self.temperature, # only default (1) is supported for o4-mini
            max_completion_tokens=self.max_length,
            reasoning_effort=self.effort,  # reasoning token can be set as low at least, so I set low
        )
        log_usage(self.model, stage, getattr(response, "usage", None))
        print("response:", response)
        if not response.choices:
            print("ERROR: No choices in response")
            return f"ERROR WHEN GENERATING RESPONSE: {response.error if response.error else ''}"
        return response.choices[0].message.content


def example_usage(model):
    client = OpenRouterClient(
        model=model, temperature=0.0, max_length=1500, effort="none"
    )
    system_prompt = "You are a helpful assistant."
    user_prompt = "Introduce yourself briefly."
    response = client.generate(system_prompt, user_prompt)
    print("Generated Response:", response)


if __name__ == "__main__":
    example_usage("openai/o4-mini")
