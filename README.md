# LLM Paraphrasing Attacks

Fully reproducible pipeline for adversarial paraphrasing attacks on a deception classifier using open-source LLMs.

## Overview

A ModernBERT classifier trained on the Hippocorpus dataset (accuracy/F1 = 0.78) labels autobiographical statements as **truthful** or **deceptive**. This pipeline replicates the attack procedure used by human participants: each LLM receives a statement and must rewrite it so the classifier flips its prediction, receiving feedback after every attempt (up to 10 per sequence).

Three architectures are compared:

| Architecture | Provider | Largest variant |
|---|---|---|
| `gemma4` | Google DeepMind | gemma4:31b-cloud (Ollama) / gemma-3-27b-it (Together AI) |
| `llama3.1` | Meta | llama3.1:8b (Ollama) / Meta-Llama-3.1-405B-Instruct-Turbo (Together AI) |
| `qwen3` | Alibaba Cloud | qwen3:8b (Ollama) / Qwen3-235B-A22B (Together AI) |

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
│   ├── llm_client.py    # OpenAI-compatible client for Ollama or Together AI
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

### 3 — Configure your LLM provider

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# "ollama" for local inference, "together" for Together AI cloud
LLM_PROVIDER=ollama
```

**Together AI** is recommended for the full-size variants (405B, 235B). Sign up at <https://api.together.ai>.

For **Together AI** instead of local Ollama, set:

```dotenv
LLM_PROVIDER=together
TOGETHER_API_KEY=your_key_here
```

### 4 - Local Ollama setup and model management

Install and start Ollama:

```bash
brew install --cask ollama
open -a Ollama
which ollama
ollama --version
```

Pull only the local models you need for testing:

```bash
ollama pull llama3.1:8b
ollama pull qwen3:8b
ollama pull gemma4:31b-cloud
```

List local models:

```bash
ollama list
```

Where models are stored on macOS:

```text
~/.ollama/models
```

Remove one model when no longer needed:

```bash
ollama rm llama3.1:8b
```

Remove all local Ollama model data (careful):

```bash
rm -rf ~/.ollama/models
```

---

## Running the pipeline

```bash
# Run all three architectures (n=325 sequences each)
python src/app.py

# Run a single architecture
python src/app.py --architecture llama3.1

# Quick local smoke test (10 sequences, max 2 attempts each)
python src/app.py --architecture llama3.1 --test

# Override number of sequences (e.g. quick smoke-test)
python src/app.py --architecture llama3.1 --n-sequences 5

# Resume after a crash (skip the first 47 completed sequences)
python src/app.py --architecture llama3.1 --start-index 47
```

Results are written to `results/<architecture>_<timestamp>.csv`.
In `--test` mode, output is written to `results/test/<architecture>_<timestamp>.csv`.

---

## Output format

The CSV mirrors the human `all_sessions.csv` structure. Human-only columns (prolific_id, attention checks, UI timing, motivation/difficulty ratings) are omitted. Added columns:

| Column | Description |
|---|---|
| `attack_modality` | Always `"llm"` |
| `llm_architecture` | `gemma4` / `llama3.1` / `qwen3` |
| `temperature` | Sampled temperature for this sequence |
| `rewrite{1-10}_prompt` | Full prompt sent to the LLM for each attempt |

All other columns (`session_id`, `statement_id`, `original_text`, `original_label`, `original_confidence`, `attempts_used`, `rewrite{n}_text/label/confidence/duration_ms`, `strategies`, `session_start/end`, `total_duration_ms`) are identical to the human format.

---

## Reproducibility

- Statement sampling uses a fixed seed (`RANDOM_SEED = 42` in `config.py`) — all architectures attack the same 325 statements.
- Temperatures are drawn per-architecture from a separate seeded RNG (`ARCH_SEEDS` in `config.py`).
- Zero-shot prompting only (no few-shot examples).
- To change model variants or any other parameter, edit `src/config.py` — no other file needs changing.
