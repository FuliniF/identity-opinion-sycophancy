import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the shared subject roster lives in src/
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from subject_models import FRIENDLY_NAMES, ROLEPLAY_MODELS  # noqa: E402

# model name: deepseek-v3.1, o4-mini, llama-3.3-70b-instruct, qwen3-32b, mistral-small-3.1-24b-instruct
# model_name = "qwen3-32b"
eval_results_dir = ""
response_dir = ""
output_dir = ""
characters = [
    "ambivalent right",
    "committed conservatives",
    "democratic mainstays",
    "establishment liberals",
    "faith and flag conservatives",
    "outsider left",
    "populist right",
    "progressive left",
    "stressed sideliners",
]
responded_as_characters = [
    "faith_and_flag_conservatives",
    "outsider_left",
    "stressed_sideliners",
]


def save_baseline_csv(
    response_path,
    anchor_path,
    evaluate_result_path,
    output_path,
    modelname=None,
    as_characters=False,
):
    with open(response_path, "r") as f:
        data = [json.loads(line) for line in f]
    with open(anchor_path, "r") as f:
        anchors = json.load(f)
    if not as_characters:
        with open(evaluate_result_path, "r") as f:
            evaluate_results = json.load(f)
    else:
        evaluate_results = []
        for respond_as_char in responded_as_characters:
            with open(
                os.path.join(
                    eval_results_dir,
                    f"baseline_global_{field}_responded_as_{respond_as_char}.json",
                ),
                "r",
            ) as f:
                evaluate_results += json.load(f)

    # id, model, anchor_event, response, overall_score
    with open(output_path, "w", newline="") as csvfile:
        fieldnames = [
            "id",
            "model",
            "anchor_event",
            "response",
            "overall_score",
            "mentions_fact_a",
            "mentions_fact_b",
            "imbalance_description",
            "structural_bias",
            "framing_bias",
            "selection_bias",
            "normative_bias",
            "was_responded_as",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        if not as_characters:
            for i, (d, a) in enumerate(zip(data, anchors)):
                fub = evaluate_results[i].get("fact_usage_balance", {}) or {}
                writer.writerow(
                    {
                        "id": i,
                        "model": model_name,
                        "anchor_event": a["objective_anchor"],
                        "response": d["response"],
                        "overall_score": evaluate_results[i]["overall_stance_score"],
                        "mentions_fact_a": fub.get("mentions_fact_a", False),
                        "mentions_fact_b": fub.get("mentions_fact_b", False),
                        "imbalance_description": fub.get("imbalance_description", ""),
                        "structural_bias": evaluate_results[i].get(
                            "structural_bias", ""
                        ),
                        "framing_bias": evaluate_results[i].get("framing_bias", ""),
                        "selection_bias": evaluate_results[i].get("selection_bias", ""),
                        "normative_bias": evaluate_results[i].get("normative_bias", ""),
                    }
                )
        else:
            for res in evaluate_results:
                if "overall_stance_score" not in res:
                    # Malformed judge output for this entry; skip rather than block.
                    continue
                fub = res.get("fact_usage_balance", {}) or {}
                writer.writerow(
                    {
                        "id": res["id"],
                        "model": model_name,
                        "anchor_event": anchors[int(res["id"])]["objective_anchor"],
                        "response": f'"{data[int(res["id"])]["response"]}"',
                        "overall_score": res["overall_stance_score"],
                        "mentions_fact_a": fub.get("mentions_fact_a", False),
                        "mentions_fact_b": fub.get("mentions_fact_b", False),
                        "imbalance_description": fub.get("imbalance_description", ""),
                        "structural_bias": res.get("structural_bias", ""),
                        "framing_bias": res.get("framing_bias", ""),
                        "selection_bias": res.get("selection_bias", ""),
                        "normative_bias": res.get("normative_bias", ""),
                        "was_responded_as": res["was_responded_as"],
                    }
                )


def save_stereotype_csv(
    response_path,
    anchor_path,
    evaluate_result_path,
    output_path,
    modelname=None,
    as_characters=False,
):
    with open(response_path, "r") as f:
        data = json.load(f)
    with open(anchor_path, "r") as f:
        anchors = json.load(f)
    if not as_characters:
        with open(evaluate_result_path, "r") as f:
            evaluate_results = json.load(f)
    else:
        evaluate_results = []
        for respond_as_char in responded_as_characters:
            if not os.path.exists(
                os.path.join(
                    eval_results_dir,
                    f"stereotype_global_{field}_responded_as_{respond_as_char}.json",
                )
            ):
                continue
            with open(
                os.path.join(
                    eval_results_dir,
                    f"stereotype_global_{field}_responded_as_{respond_as_char}.json",
                ),
                "r",
            ) as f:
                evaluate_results += json.load(f)

    # id, model, anchor_event, response, overall_score
    with open(output_path, "w", newline="") as csvfile:
        fieldnames = [
            "id",
            "model",
            "anchor_event",
            "character",
            "response",
            "overall_score",
            "mentions_fact_a",
            "mentions_fact_b",
            "imbalance_description",
            "structural_bias",
            "framing_bias",
            "selection_bias",
            "normative_bias",
            "was_responded_as",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        if not as_characters:
            for i, (d, a) in enumerate(zip(data, anchors)):
                for character in characters:
                    if character not in d:
                        continue
                    fub = evaluate_results[i][character].get("fact_usage_balance", {}) or {}
                    writer.writerow(
                        {
                            "id": i,
                            "model": model_name,
                            "anchor_event": a["objective_anchor"],
                            "character": character,
                            "response": d[character],
                            "overall_score": evaluate_results[i][character][
                                "overall_stance_score"
                            ],
                            "mentions_fact_a": fub.get("mentions_fact_a", False),
                            "mentions_fact_b": fub.get("mentions_fact_b", False),
                            "imbalance_description": fub.get("imbalance_description", ""),
                            "structural_bias": evaluate_results[i][character].get(
                                "structural_bias", ""
                            ),
                            "framing_bias": evaluate_results[i][character].get(
                                "framing_bias", ""
                            ),
                            "selection_bias": evaluate_results[i][character].get(
                                "selection_bias", ""
                            ),
                            "normative_bias": evaluate_results[i][character].get(
                                "normative_bias", ""
                            ),
                        }
                    )
        else:
            for res in evaluate_results:
                for character in characters:
                    if character not in data[int(res["id"])]:
                        continue
                    if character not in res:
                        continue
                    fub = res[character].get("fact_usage_balance", {}) or {}
                    writer.writerow(
                        {
                            "id": res["id"],
                            "model": model_name,
                            "anchor_event": anchors[int(res["id"])]["objective_anchor"],
                            "character": character,
                            "response": f'"{data[int(res["id"])][character]}"',
                            "overall_score": res[character]["overall_stance_score"],
                            "mentions_fact_a": fub.get("mentions_fact_a", False),
                            "mentions_fact_b": fub.get("mentions_fact_b", False),
                            "imbalance_description": fub.get("imbalance_description", ""),
                            "structural_bias": res[character]["structural_bias"],
                            "framing_bias": res[character]["framing_bias"],
                            "selection_bias": res[character]["selection_bias"],
                            "normative_bias": res[character]["normative_bias"],
                            "was_responded_as": res[character]["was_responded_as"],
                        }
                    )


def save_sycophancy_csv(
    response_path,
    anchor_path,
    evaluate_result_path,
    output_path,
    modelname=None,
    as_characters=False,
):
    with open(response_path, "r") as f:
        data = json.load(f)
    with open(anchor_path, "r") as f:
        anchors = json.load(f)
    if not as_characters:
        with open(evaluate_result_path, "r") as f:
            evaluate_results = json.load(f)
    else:
        evaluate_results = []
        for respond_as_char in responded_as_characters:
            with open(
                os.path.join(
                    eval_results_dir,
                    f"sycophancy_global_{field}_responded_as_{respond_as_char}.json",
                ),
                "r",
            ) as f:
                evaluate_results += json.load(f)

    # id, model, anchor_event, response, overall_score
    with open(output_path, "w", newline="") as csvfile:
        fieldnames = [
            "id",
            "model",
            "anchor_event",
            "character",
            "side",
            "response",
            "overall_score",
            "mentions_fact_a",
            "mentions_fact_b",
            "imbalance_description",
            "structural_bias",
            "framing_bias",
            "selection_bias",
            "normative_bias",
            "sycophantic_alignment",
            "was_responded_as",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        if not as_characters:
            for i, (d, a) in enumerate(zip(data, anchors)):
                for character in characters:
                    if character not in d:
                        continue
                    fub_A = evaluate_results[i][character]["A"].get("fact_usage_balance", {}) or {}
                    fub_B = evaluate_results[i][character]["B"].get("fact_usage_balance", {}) or {}
                    writer.writerow(
                        {
                            "id": i,
                            "model": model_name,
                            "anchor_event": a["objective_anchor"],
                            "character": character,
                            "side": "A",
                            "response": d[character]["response_A"],
                            "overall_score": evaluate_results[i][character]["A"][
                                "overall_stance_score"
                            ],
                            "mentions_fact_a": fub_A.get("mentions_fact_a", False),
                            "mentions_fact_b": fub_A.get("mentions_fact_b", False),
                            "imbalance_description": fub_A.get("imbalance_description", ""),
                            "structural_bias": evaluate_results[i][character]["A"].get(
                                "structural_bias", ""
                            ),
                            "framing_bias": evaluate_results[i][character]["A"].get(
                                "framing_bias", ""
                            ),
                            "selection_bias": evaluate_results[i][character]["A"].get(
                                "selection_bias", ""
                            ),
                            "normative_bias": evaluate_results[i][character]["A"].get(
                                "normative_bias", ""
                            ),
                            "sycophantic_alignment": evaluate_results[i][character][
                                "A"
                            ].get("sycophantic_alignment", ""),
                        }
                    )
                    writer.writerow(
                        {
                            "id": i,
                            "model": model_name,
                            "anchor_event": a["objective_anchor"],
                            "character": character,
                            "side": "B",
                            "response": d[character]["response_B"],
                            "overall_score": evaluate_results[i][character]["B"][
                                "overall_stance_score"
                            ],
                            "mentions_fact_a": fub_B.get("mentions_fact_a", False),
                            "mentions_fact_b": fub_B.get("mentions_fact_b", False),
                            "imbalance_description": fub_B.get("imbalance_description", ""),
                            "structural_bias": evaluate_results[i][character]["B"].get(
                                "structural_bias", ""
                            ),
                            "framing_bias": evaluate_results[i][character]["B"].get(
                                "framing_bias", ""
                            ),
                            "selection_bias": evaluate_results[i][character]["B"].get(
                                "selection_bias", ""
                            ),
                            "normative_bias": evaluate_results[i][character]["B"].get(
                                "normative_bias", ""
                            ),
                            "sycophantic_alignment": evaluate_results[i][character][
                                "B"
                            ].get("sycophantic_alignment", ""),
                        }
                    )
        else:
            for res in evaluate_results:
                for character in characters:
                    if character not in data[int(res["id"])]:
                        continue
                    if character not in res:
                        continue
                    fub_A = res[character]["A"].get("fact_usage_balance", {}) or {}
                    fub_B = res[character]["B"].get("fact_usage_balance", {}) or {}
                    writer.writerow(
                        {
                            "id": res["id"],
                            "model": model_name,
                            "anchor_event": anchors[int(res["id"])]["objective_anchor"],
                            "character": character,
                            "side": "A",
                            "response": f'"{data[int(res["id"])][character]["response_A"]}"',
                            "overall_score": res[character]["A"][
                                "overall_stance_score"
                            ],
                            "mentions_fact_a": fub_A.get("mentions_fact_a", False),
                            "mentions_fact_b": fub_A.get("mentions_fact_b", False),
                            "imbalance_description": fub_A.get("imbalance_description", ""),
                            "structural_bias": res[character]["A"]["structural_bias"],
                            "framing_bias": res[character]["A"]["framing_bias"],
                            "selection_bias": res[character]["A"]["selection_bias"],
                            "normative_bias": res[character]["A"]["normative_bias"],
                            "sycophantic_alignment": res[character]["A"][
                                "sycophantic_alignment"
                            ],
                            "was_responded_as": res[character]["was_responded_as"],
                        }
                    )
                    writer.writerow(
                        {
                            "id": res["id"],
                            "model": model_name,
                            "anchor_event": anchors[int(res["id"])]["objective_anchor"],
                            "character": character,
                            "side": "B",
                            "response": f'"{data[int(res["id"])][character]["response_B"]}"',
                            "overall_score": res[character]["B"][
                                "overall_stance_score"
                            ],
                            "mentions_fact_a": fub_B.get("mentions_fact_a", False),
                            "mentions_fact_b": fub_B.get("mentions_fact_b", False),
                            "imbalance_description": fub_B.get("imbalance_description", ""),
                            "structural_bias": res[character]["B"]["structural_bias"],
                            "framing_bias": res[character]["B"]["framing_bias"],
                            "selection_bias": res[character]["B"]["selection_bias"],
                            "normative_bias": res[character]["B"]["normative_bias"],
                            "sycophantic_alignment": res[character]["B"][
                                "sycophantic_alignment"
                            ],
                            "was_responded_as": res[character]["was_responded_as"],
                        }
                    )


def plot_histogram(overall_scores, field, prefix=None):
    plt.hist(overall_scores, bins=20)
    median = sorted(overall_scores)[len(overall_scores) // 2]
    plt.axvline(
        median, color="red", linestyle="--", linewidth=2, label=f"Median: {median:.2f}"
    )
    plt.xlabel("Overall Stance Score")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Overall Stance Scores for {model_name} on {field}")
    plt.legend()
    plt.savefig(
        f"{eval_results_dir}/{prefix}_{model_name}_on_gemini-2.5-flash_result_{field}.png"
    )
    plt.close()


def save_baseline(field, as_characters):
    save_baseline_csv(
        response_path=os.path.join(
            response_dir,
            field,
            f"{'deepseek-chat-v3.1' if model_name == 'deepseek-v3.1' else model_name}{'_as_characters' if as_characters else ''}_new_baseline_response.jsonl",
        ),
        anchor_path=os.path.join(REPO_ROOT, "data/scenarios/final", field, "anchor_events.json"),
        evaluate_result_path=os.path.join(
            eval_results_dir, f"baseline_global_{field}.json"
        ),
        output_path=os.path.join(output_dir, f"baseline_global_{field}.csv"),
        as_characters=as_characters,
    )


def save_stereotype(field, as_characters):
    save_stereotype_csv(
        response_path=os.path.join(
            response_dir,
            field,
            f"{'deepseek-chat-v3.1' if model_name == 'deepseek-v3.1' else model_name}{'_as_characters' if as_characters else ''}_stereotyping_response.json",
        ),
        anchor_path=os.path.join(REPO_ROOT, "data/scenarios/final", field, "anchor_events.json"),
        evaluate_result_path=os.path.join(
            eval_results_dir,
            f"stereotype_global_{field}.json",
        ),
        output_path=os.path.join(output_dir, f"stereotype_global_{field}.csv"),
        as_characters=as_characters,
    )


def save_sycophancy(field, as_characters):
    save_sycophancy_csv(
        response_path=os.path.join(
            response_dir,
            field,
            f"{'deepseek-chat-v3.1' if model_name == 'deepseek-v3.1' else model_name}{'_as_characters' if as_characters else ''}_sycophancy_response.json",
        ),
        anchor_path=os.path.join(REPO_ROOT, "data/scenarios/final", field, "anchor_events.json"),
        evaluate_result_path=os.path.join(
            eval_results_dir,
            f"sycophancy_global_{field}.json",
        ),
        output_path=os.path.join(output_dir, f"sycophancy_global_{field}.csv"),
        as_characters=as_characters,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten the global-bias eval JSON into the per-model "
                    "CSVs the analysis scripts read.")
    parser.add_argument(
        "--eval_root", type=str,
        default=os.path.join(REPO_ROOT, "evaluations"),
        help="evaluation tree root. Default evaluations/; pass "
             "evaluations/consensus to flatten the aggregated 3-judge panel.")
    parser.add_argument(
        "--as_characters", action="store_true",
        help="flatten the role-play (as_characters) evals instead.")
    parser.add_argument(
        "--model", type=str, default="all",
        help="'all' (the shared subject roster) or one friendly model name.")
    args = parser.parse_args()
    as_characters = args.as_characters

    # Role-play data exists for the 5-model subset only; the no-role-play arm
    # uses the full roster. Roster comes from the shared subject_models.py.
    roster = ROLEPLAY_MODELS if as_characters else FRIENDLY_NAMES
    if args.model != "all":
        roster = [args.model]

    # mcc: prefer "as_characters" (judge trees use this name); fall back to
    # "as_characters_new" for backward-compat with the old single-judge layout.
    # in_sub = "as_characters_new" if as_characters else "default"  # mcc: original
    _ac_new = "as_characters_new"
    _ac = "as_characters"
    if as_characters:
        in_sub = _ac if os.path.isdir(os.path.join(args.eval_root, _ac)) else _ac_new
    else:
        in_sub = "default"
    out_sub = "csvs_as_characters" if as_characters else "csvs"
    fields = ["diplomatic", "economy", "energy", "technology", "welfare"]

    done = skipped = 0
    for model_name in roster:
        eval_results_dir = os.path.join(args.eval_root, in_sub, model_name)
        output_dir = os.path.join(args.eval_root, out_sub, model_name)
        response_dir = os.path.join(REPO_ROOT, "responses", model_name)
        # Missing-data guard: a run may have eval JSON for only some models.
        if not os.path.isdir(eval_results_dir):
            print(f"[skip] {model_name}: no eval JSON at {eval_results_dir}")
            skipped += 1
            continue
        os.makedirs(output_dir, exist_ok=True)
        for field in fields:
            save_baseline(field, as_characters)
            save_stereotype(field, as_characters)
            save_sycophancy(field, as_characters)
        done += 1
        print(f"[done] {model_name} -> {output_dir}")
    print(f"flattened {done} model(s), skipped {skipped} "
          f"({'role-play' if as_characters else 'no-role-play'} arm).")
