# Arguinas

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Then edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-proj-...
```

You only need to set the key(s) for the model family you intend to run (Anthropic for Claude models, OpenAI for GPT models).

## Usage

Run the pipeline with default arguments:

```bash
python run_GAAR.py
```

This is equivalent to:

```bash
python run_GAAR.py \
  --data_path ./data/Sample \
  --data_filename sample.json \
  --use_general_reconstruction True \
  --use_specific_reconstruction False \
  --save_path ./output \
  --prompt_path ./prompts/GAAR \
  --subset sample \
  --model_name claude-sonnet-4-5-20250929 \
  --max_num_recon 10 \
  --max_num_debug 5 \
  --max_attempts 5
```

Outputs are written to `./output/reconstruction_<subset>_<model_name>.json`.

## Prompts

All prompt templates used by each stage of the pipeline (fallacy detection, reconstruction, validity checking, streamlining, faithfulness checking, program debugging) live under [`prompts/GAAR/`](./prompts/GAAR). Refer to these files to see or modify the instructions given to the LLM at each step.

Two reconstruction variants are provided:
- **General** (`reconstruction_general_*.txt`) — classifies reasoning into 4 broad types (deductive / inductive / analogical / abductive).
- **Specific** (`reconstruction_60_types_*.txt`) — classifies reasoning into 60 fine-grained Walton-style argumentation schemes.

Toggle between them with the `--use_general_reconstruction` / `--use_specific_reconstruction` flags.

## Data

Our train and test datasets live in [`data/`](./data). See [`data/README.md`](./data/README.md) for the full data format (top-level columns, `fallacy_info`, `sections`, etc.).

### Expected input format for `run_GAAR.py`

`run_GAAR.py` only reads three fields from each entry in the input JSON:

| Field | Type | Description |
|---|---|---|
| `title` | `string` | The debate topic. |
| `background` | `string` | Background context (`"None"` if absent). |
| `argument` | `string` | The raw argument text to reconstruct. |

See [`data/Sample/sample.json`](./data/Sample/sample.json) for a minimal working example, and [`output/reconstruction_sample_claude-sonnet-4-5-20250929.json`](./output/reconstruction_sample_claude-sonnet-4-5-20250929.json) for a corresponding sample output produced by the pipeline.

To run on your own data, place a JSON file with the same schema under any directory and point `--data_path` / `--data_filename` to it.
