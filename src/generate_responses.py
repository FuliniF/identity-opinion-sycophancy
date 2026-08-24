import argparse
import os

from baseline import baseline_response, baseline_response_as_characters
from stereotype import stereotyping_response, stereotyping_response_as_characters
from sycophancy import sycophancy_response, sycophancy_response_as_characters


def main(args):
    subdirs = [d for d in os.listdir(args.input_dir)
               if os.path.isdir(os.path.join(args.input_dir, d))]
    character_to_test = ['faith and flag conservatives', 'outsider left', 'stressed sideliners']
    for field in subdirs:
        print(f"Processing field: {field}", flush=True)
        print("Generating baseline responses...", flush=True)
        baseline_response(args, field, model_effort=args.effort)
        if not args.skip_roleplay:
            baseline_response_as_characters(args, field, model_effort=args.effort, character_to_test=character_to_test)
        print("Generating stereotyping responses...", flush=True)
        stereotyping_response(args, field, model_effort=args.effort)
        if not args.skip_roleplay:
            stereotyping_response_as_characters(args, field, model_effort=args.effort, character_to_test=character_to_test)
        print("Generating sycophancy responses...", flush=True)
        sycophancy_response(args, field, model_effort=args.effort)
        if not args.skip_roleplay:
            sycophancy_response_as_characters(args, field, model_effort=args.effort, character_to_test=character_to_test)
        print(f"Finished processing field: {field}", flush=True)
        print("-----------------------------------", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="The model to use for generation.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="The temperature for generation."
    )
    parser.add_argument("--input-dir", type=str, help="The input directory.")
    parser.add_argument("--output-dir", type=str, help="The output directory.")
    parser.add_argument(
        "--effort",
        type=str,
        default="none",
        help="The reasoning effort level (e.g., none, low, medium, high).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="smoke test: cap anchor items per field to the first N.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="thread-pool size for concurrent generation calls.",
    )
    parser.add_argument(
        "--skip-roleplay",
        action="store_true",
        help="skip the *_as_characters role-play arms (the 15-model "
             "dissociation only needs the non-role-play conditions).",
    )
    args = parser.parse_args()
    main(args)
