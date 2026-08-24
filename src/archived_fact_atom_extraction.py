import os

from client import OpenRouterClient

SYS_PROMPT = 'You are an expert data analyst specializing in Information Extraction (IE). Your task is to identify the "ground truth" of a narrative by stripping away all emotional coloring, metaphors, and subjective interpretations.'
USER_PROMPT = 'Extract 5 strictly Objective Fact Atoms (OFAs) from the provided Scenario.\
    Input Data: \
        - Scenario: {skeleton_scenario} \
    Guidelines: \
        - No Adjectives: Remove words like "tense," "uneasy,", or "sunlit." Focus only on nouns and verbs. \
        - Observable Actions: Only include things that can be seen, heard, or measured. \
        - Neutral Identity: Describe parties by their role in the event, not their perceived power (e.g., "API provider" and "API user" instead of "Tech giant" and "Small startup"). \
        - No Motives: Do not assume why someone did something. State what was done. \
        - Mutual Agreement: Each atom must be a statement that all characters in the story would agree is "the event that took place." \
    Output Format: A numbered list of 5 short, discrete sentences. No introduction or conclusion.'


def fact_atom_extraction(args) -> None:
    client = OpenRouterClient(model=args.model, temperature=args.temperature)
    os.makedirs(args.output_dir, exist_ok=True)

    # open input-dir and read temp_scenario.txt
    with open(f"{args.input_dir}/temp_scenario.txt", "r") as f:
        scenario_data = f.read()
    scenario_data = scenario_data.split("\n\n", 2)[-1]

    user_prompt = USER_PROMPT.format(skeleton_scenario=scenario_data)

    response = client.generate(SYS_PROMPT, user_prompt)

    # Save the response to output-dir
    output_path = f"{args.output_dir}/temp_fact_atom.txt"
    with open(output_path, "w") as out_f:
        out_f.write(scenario_data + "\n\n" + response)


if __name__ == "__main__":
    import argparse

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

    fact_atom_extraction(args)
