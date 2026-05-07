import os
from dotenv import load_dotenv

# Load .env from project root before reading any env vars
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

# ---------------------------------------------------------------------------
# Model IDs (OpenRouter only)
# OpenRouter: https://openrouter.ai/models
# Verify / update these IDs as new model versions are released.
# ---------------------------------------------------------------------------
LLM_ARCHITECTURES = {
    "gemma4": {
        "model": "google/gemma-4-31b-it",
    },
    "llama3.3": {
        "model": "meta-llama/llama-3.3-70b-instruct",
    },
    "qwen3": {
        "model": "qwen/qwen3-next-80b-a3b-instruct",
    },
}

# Architecture-specific RNG seeds (deterministic but unique per architecture)
ARCH_SEEDS = {
    "gemma4":   43,
    "llama3.3": 44,
    "qwen3":    45,
}

# ---------------------------------------------------------------------------
# Attack parameters
# ---------------------------------------------------------------------------
MAX_ATTEMPTS       = 10
TEMPERATURE_MIN    = 0.1
TEMPERATURE_MAX    = 1.0
N_ATTACK_SEQUENCES = 325   # total sequences per architecture, to match human n
RANDOM_SEED        = 42    # for statement sampling (same statements across archs)
WORD_TOLERANCE     = 20    # rewritten text must stay within +/- this many tokens
MAX_LENGTH_REPROMPTS = 3   # retry count within the same attempt for length violations

# Token-cap tuning per architecture
QWEN_MAXTOK_MULTIPLIER = 1.10
QWEN_MAXTOK_ADDON      = 8

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = "data/hippocorpus_test_truncated_80_100.csv"
RESULTS_DIR  = "results"
MODEL_PATH   = "model/modernbert_trained"
