import argparse
import json
import os

from client import OpenRouterClient

SYS_PROMPT = 'You are a Neutral Data Synthesis Engine. Your goal is to transform abstract ideological conflicts into concrete, objective "Anchor Events" for a sociopolitical research project.'
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "scenario_synthesis.txt")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    USER_PROMPT = f.read()


def scenario_synthesis(args, target_field) -> None:
    client = OpenRouterClient(model=args.model, temperature=args.temperature)
    os.makedirs(f"{args.output_dir}/{target_field}", exist_ok=True)

    # open input-dir and read train.json
    train_path = f"{args.input_dir}/{target_field}/train.json"
    with open(train_path, "r") as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} items from {train_path}")

    output_path = f"{args.output_dir}/{target_field}/anchor_events.json"

    total_data = []
    for i, item in enumerate(train_data):
        user_prompt = USER_PROMPT.format(
            focus_topic=item["focus_topic"],
            prompt=item["prompt"],
            accept=item["accept"],
            reject=item["reject"],
        )

        response = client.generate_with_schema(
            SYS_PROMPT, user_prompt, schema_type="anchor"
        )
        response["id"] = i

        total_data.append(response)

    with open(output_path, "w") as f:
        json.dump(total_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use for generation.",
    )
    parser.add_argument(
        "--temperature", type=float, help="The temperature for generation."
    )
    parser.add_argument("--input-dir", type=str, help="The input directory.")
    parser.add_argument("--output-dir", type=str, help="The output directory.")
    args = parser.parse_args()

    # list all subdirectory in input_dir
    subdirs = os.listdir(args.input_dir)
    for field in subdirs:
        scenario_synthesis(args, field)
