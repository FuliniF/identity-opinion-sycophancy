# How Identity and Opinion Shape Political Sycophancy in LLMs

This repository contains the scripts and prompt templates for reproducing the
experimental pipeline described in *How Identity and Opinion Shape Political
Sycophancy in LLMs*, including the opinion-only workflow. It intentionally
contains **no experiment inputs or outputs**:

- no benchmark data or persona-description files;
- no generated model responses;
- no LLM-judge evaluations, derived CSVs, figures, or statistical results;
- no notebooks, which may embed output cells.

The repository has a fresh Git history, separate from the private research
repository.

## Setup

Create an environment and install the dependencies:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

Set credentials in a local `.env` file. The clients support OpenRouter-style
generation endpoints and OpenAI-compatible judge endpoints; see `src/client.py`
and `src/judge_pool.py` for the environment variables used by each workflow.
Never commit `.env`.

## Private/local input layout

Obtain the authorized benchmark data and prepare local inputs in the following
ignored layout. Do not commit these files to this repository.

```text
data/
├── benchmark/dataset_original/<domain>/train.json
├── characters_description/<persona>.txt
└── scenarios/final/<domain>/
    ├── anchor_events.json
    └── new_narratives.json
```

`<domain>` is one of the benchmark domains (for example, `economy` or
`diplomatic`). `anchor_events.json` and `new_narratives.json` can be generated
with the supplied synthesis scripts. Persona files are local prompt inputs used
by the role-play arms.

## Reproduction workflow

Run scripts from `src/`, because the original pipeline imports its modules as
top-level files:

```bash
cd src

# 1. Create anchor events from the authorized benchmark train split.
python scenario_synthesis.py --model <generator-model> --temperature 0.5 \
  --input-dir ../data/benchmark/dataset_original \
  --output-dir ../data/scenarios/final

# 2. Create persona-conditioned narratives.
python narrative_synthesis.py --model <generator-model> --temperature 0.5 \
  --input-dir ../data/scenarios/final --output-dir ../data/scenarios/final

# 3. Generate subject-model responses. Use local, ignored output paths.
python generate_responses.py --model <subject-model> --temperature 0 \
  --input-dir ../data/scenarios/final --output-dir ../responses/<subject-model>

# 4. Evaluate a response set with an LLM judge.
python evaluate.py --eval_model <judge-model> --response_model <subject-model> \
  --datapath ../responses/<subject-model> \
  --result_dir ../evaluations/<judge-model>/<subject-model>
```

For the additional opinion-only condition, use `narrative_synthesis_neutral.py`,
`opinion_only.py`, `opinion_only_roleplay.py`, `evaluate_opinion_only.py`, and
`evaluate_opinion_only_roleplay.py`. `subject_models.py` is the single source
of truth for the model roster. The remaining scripts support retries,
multi-judge aggregation, audits, analysis, and plotting, all over local output
paths.

## Repository contents

- `src/`: generation, evaluation, opinion-only, ATP, retry, audit, analysis,
  and plotting code.
- `src/prompts/`: all prompt templates used by the pipeline.
- `scripts/`: batch-serving helpers and post-processing utilities.

All local inputs and outputs are protected by `.gitignore`. Before publishing,
run `git status --ignored` and verify that only source files, prompt templates,
and documentation are staged.
