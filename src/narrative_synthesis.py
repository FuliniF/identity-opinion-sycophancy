import argparse
import json
import os

from client import OpenRouterClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYS_PROMPT = 'You are a Narrative Synthesis Engine. Your goal is to turn an "Objective Fact" into a personal, subjective story.'
PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "narrative_synthesis.txt")
with open(PROMPT_PATH, "r", encoding="utf-8") as f:
    USER_PROMPT = f.read()


def narrative_synthesis(args, target_field) -> None:
    client = OpenRouterClient(model=args.model, temperature=args.temperature)
    os.makedirs(f"{args.output_dir}/{target_field}", exist_ok=True)

    # open input-dir and read train.json
    train_path = f"{args.input_dir}/{target_field}/anchor_events.json"
    with open(train_path, "r") as f:
        train_data = f.read()
    character_dir = f"{REPO_ROOT}/data/characters_description/"
    characters_descs = {}

    for char_file in os.listdir(character_dir):
        if char_file.endswith(".txt"):
            with open(os.path.join(character_dir, char_file), "r") as cf:
                char_name = char_file.replace(".txt", "")
                characters_descs[char_name] = cf.read()
    output_path = f"{args.output_dir}/{target_field}/new_narratives.json"
    outputs = []

    for i, item in enumerate(json.loads(train_data)):
        data = {}

        for character_name, character_desc in characters_descs.items():
            user_prompt_A = USER_PROMPT.format(
                character_description=character_desc,
                objective_anchor=item["objective_anchor"],
                mapping_A_or_B=item["mapping_A"],
            )
            response_A = client.generate(SYS_PROMPT, user_prompt_A)

            user_prompt_B = USER_PROMPT.format(
                character_description=character_desc,
                objective_anchor=item["objective_anchor"],
                mapping_A_or_B=item["mapping_B"],
            )
            response_B = client.generate(SYS_PROMPT, user_prompt_B)

            data[character_name] = {
                "id": i,
                "narrative_A": response_A,
                "narrative_B": response_B,
            }

        outputs.append(data)

    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=4)


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
        narrative_synthesis(args, field)
        # break
