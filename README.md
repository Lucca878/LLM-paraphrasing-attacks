# LLM Paraphrasing Attacks

Fully reproducible pipeline for adversarial paraphrasing attacks on a deception classifier using OpenRouter-hosted LLMs.

## Overview

A ModernBERT classifier trained on the Hippocorpus dataset (accuracy/F1 = 0.78) labels autobiographical statements as **truthful** or **deceptive**. This pipeline replicates the attack procedure used by human participants: each LLM receives a statement and must rewrite it so the classifier flips its prediction, receiving feedback after every attempt (up to 10 per sequence).

Three architectures are compared:

| Architecture | OpenRouter model |
|---|---|
| `gemma4` | `google/gemma-4-31b-it` |
| `llama3.3` | `meta-llama/llama-3.3-70b-instruct` |
| `qwen3` | `qwen/qwen3-235b-a22b` |

Each architecture runs **n = 325** paraphrasing sequences using the same statement sample (for comparability with human participants) and independently sampled temperatures drawn uniformly from [0.1, 1.0]. The most effective and efficient architecture will be selected for the human vs. LLM comparison.

---

## Project structure

```
llmAttacks/
├── data/
│   └── hippocorpus_test_truncated_80_100.csv   # hold-out pool (n=262 statements)
├── model/
│   └── modernbert_trained/                      # ModernBERT weights (not committed)
├── results/                                     # output CSVs written here
├── src/
│   ├── config.py        # all tuneable constants (n, temperature range, model IDs, paths)
│   ├── dao.py           # AttackSequence / AttemptResult data classes
│   ├── llm_client.py    # OpenRouter SDK client
│   ├── utility.py       # classifier, prompt builder, CSV writer
│   └── app.py           # entry point
├── .env.example         # copy to .env and fill in your key
├── environment.yml      # conda environment
└── requirements.txt     # pip-only alternative
```

---

## Setup

### 1 — Clone and create the conda environment

```bash
git clone <repo-url> llmAttacks
cd llmAttacks
conda env create -f environment.yml
conda activate llm-attacks
```

### 2 — Add the model weights

Place the trained ModernBERT checkpoint at:

```
model/modernbert_trained/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
```

### 3 — Configure OpenRouter

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
OPENROUTER_API_KEY=your_key_here
```

Sign up at <https://openrouter.ai>.

This project uses OpenRouter's official Python SDK (`openrouter`).

Optional API controls:

```dotenv
LLM_API_TIMEOUT_SECONDS=120
LLM_API_MAX_RETRIES=1
```

Optional API controls:

```dotenv
LLM_API_TIMEOUT_SECONDS=120
LLM_API_MAX_RETRIES=1
```

---

## Running the pipeline

```bash
# Run all three architectures (n=325 sequences each)
python src/app.py

# Run a single architecture
python src/app.py --architecture llama3.3

# Quick local smoke test (10 sequences, max 3 attempts each)
python src/app.py --architecture llama3.3 --test

# Override number of sequences (e.g. quick smoke-test)
python src/app.py --architecture llama3.3 --n-sequences 5

# Resume after a crash (skip the first 47 completed sequences)
python src/app.py --architecture llama3.3 --start-index 47
```

Results are written to `results/<architecture>_<timestamp>.csv`.
In `--test` mode, output is written to `results/test/<architecture>_<timestamp>.csv`.

---

## Output format

The CSV mirrors the human `all_sessions.csv` structure. Human-only columns (prolific_id, attention checks, UI timing, motivation/difficulty ratings) are omitted. Added columns:

| Column | Description |
|---|---|
| `attack_modality` | Always `"llm"` |
| `llm_architecture` | `gemma4` / `llama3.3` / `qwen3` |
| `temperature` | Sampled temperature for this sequence |
| `rewrite{1-10}_prompt` | Full prompt sent to the LLM for each attempt |

All other columns (`session_id`, `statement_id`, `original_text`, `original_label`, `original_confidence`, `attempts_used`, `rewrite{n}_text/label/confidence/duration_ms`, `strategies`, `session_start/end`, `total_duration_ms`) are identical to the human format.

---

## Reproducibility

- Statement sampling uses a fixed seed (`RANDOM_SEED = 42` in `config.py`) — all architectures attack the same 325 statements.
- Temperatures are drawn per-architecture from a separate seeded RNG (`ARCH_SEEDS` in `config.py`).
- Zero-shot prompting only (no few-shot examples).
- To change model variants or any other parameter, edit `src/config.py` — no other file needs changing.
