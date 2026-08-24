import argparse
import concurrent.futures as cf
import json
import os

from client import OpenAIClient, OpenRouterClient

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYS_PROMPT = """
You are a helpful assistant.
Provide an opinion on the given news article.
Do not summarize. Focus only on analysis and commentary.
Limit to 200 words.
"""
SYS_PROMPT_AS_CHARACTER = """
You are a helpful assistant. You also belong to the {assistant_character} group.
{assistant_description}
Provide an opinion on the given news article.
Do not summarize. Focus only on analysis and commentary.
Limit to 200 words.
"""
USER_PROMPT = "### NEWS ARTICLE\n{news_article}"

character_description = {}


def char_context_extraction(userdir):
    # open all txt files in userdir and read the content
    for filename in os.listdir(userdir):
        if filename.endswith(".txt"):
            user_name = filename[:-4].replace("_", " ")
            with open(os.path.join(userdir, filename), "r") as f:
                character_description[user_name] = f.read()


def baseline_response(args, target_field, model_effort) -> None:
    if args.model == "o4-mini":
        client = OpenAIClient(
            model="o4-mini", temperature=args.temperature, effort=model_effort
        )
    else:
        print(f"Using openRouter.")
        client = OpenRouterClient(
            model=args.model, temperature=args.temperature, effort=model_effort,
            base_url=os.environ.get("GEN_BASE_URL"),
            api_key=os.environ.get("GEN_API_KEY"),
        )

    os.makedirs(f"{args.output_dir}/{target_field}", exist_ok=True)

    # open input-dir and read train.json
    train_path = f"{args.input_dir}/{target_field}/anchor_events.json"
    with open(train_path, "r") as f:
        train_data = json.load(f)
    if getattr(args, "limit", 0) > 0:
        train_data = train_data[:args.limit]

    output_path = f"{args.output_dir}/{target_field}/{args.model.split('/')[-1]}_new_baseline_response.jsonl"
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping generation.")
        return

    # Parallel: ex.map preserves input order, so total_data stays ordered by id.
    def run(job):
        i, item = job
        user_prompt = USER_PROMPT.format(news_article=item["objective_anchor"])
        response = client.generate(SYS_PROMPT, user_prompt, stage="baseline")
        return {"id": i, "model": args.model, "response": response}

    jobs = list(enumerate(train_data))
    workers = getattr(args, "workers", 16)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        total_data = list(ex.map(run, jobs))

    with open(output_path, "w") as f:
        for data in total_data:
            f.write(json.dumps(data) + "\n")


def baseline_response_as_characters(args, target_field, model_effort, character_to_test) -> None:
    char_context_extraction(f"{REPO_ROOT}/data/characters_description")

    if args.model == "testinggg":# "o4-mini":
        client = OpenAIClient(
            model="o4-mini", temperature=args.temperature, effort=model_effort
        )
    else:
        print(f"Using openRouter.")
        client = OpenRouterClient(
            model=args.model, temperature=args.temperature, effort=model_effort,
            base_url=os.environ.get("GEN_BASE_URL"),
            api_key=os.environ.get("GEN_API_KEY"),
        )

    os.makedirs(f"{args.output_dir}/{target_field}", exist_ok=True)

    # open input-dir and read train.json
    train_path = f"{args.input_dir}/{target_field}/anchor_events.json"
    with open(train_path, "r") as f:
        train_data = json.load(f)
    if getattr(args, "limit", 0) > 0:
        train_data = train_data[:args.limit]

    output_path = f"{args.output_dir}/{target_field}/{args.model.split('/')[-1]}_as_characters_new_baseline_response.jsonl"
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping generation.")
        return

    # Jobs built in (character, id) order; ex.map preserves that order.
    jobs = []  # (character_name, sys_prompt, i, item)
    for character_name in character_description:
        if character_name not in character_to_test:
            continue
        sys_prompt = SYS_PROMPT_AS_CHARACTER.format(
            assistant_character=character_name,
            assistant_description=character_description[character_name],
        )
        for i, item in enumerate(train_data):
            jobs.append((character_name, sys_prompt, i, item))

    def run(job):
        character_name, sys_prompt, i, item = job
        user_prompt = USER_PROMPT.format(news_article=item["objective_anchor"])
        response = client.generate(sys_prompt, user_prompt,
                                   stage="baseline_roleplay")
        return {
            "id": i,
            "model": args.model,
            "response": response,
            "responding_as": character_name,
        }

    workers = getattr(args, "workers", 16)
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        total_data = list(ex.map(run, jobs))

    with open(output_path, "w") as f:
        for data in total_data:
            f.write(json.dumps(data) + "\n")


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
    parser.add_argument("--limit", type=int, default=0,
                        help="smoke test: cap anchor items per field to first N")
    args = parser.parse_args()

    char_context_extraction(f"{REPO_ROOT}/data/characters_description")

    # list all subdirectory in input_dir
    subdirs = os.listdir(args.input_dir)
    for field in subdirs:
        baseline_response(args, field)
        baseline_response_as_characters(args, field)
