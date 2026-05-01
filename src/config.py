import os
from dotenv import load_dotenv

# Load .env from project root before reading any env vars
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

# ---------------------------------------------------------------------------
# LLM provider
# Set LLM_PROVIDER="together" in .env to use Together AI cloud (recommended
# for the largest model variants). Default is "ollama" for local inference.
# ---------------------------------------------------------------------------
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")

# ---------------------------------------------------------------------------
# Model IDs per provider
# Ollama: https://ollama.com/library
# Together AI: https://api.together.ai/models
# Verify / update these IDs as new model versions are released.
# ---------------------------------------------------------------------------
LLM_ARCHITECTURES = {
    "gemma4": {
        "ollama_model":   "gemma4:31b-cloud",
        "together_model": "google/gemma-4-31B-it",                          # serverless
    },
    "llama3.1": {
        "ollama_model":   "llama3.1:8b",
        "together_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",        # serverless (Llama 3.3 70B, closest available)
    },
    "qwen3": {
        "ollama_model":   "qwen3:8b",
        "together_model": "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",        # serverless
    },
}

# Architecture-specific RNG seeds (deterministic but unique per architecture)
ARCH_SEEDS = {
    "gemma4":   43,
    "llama3.1": 44,
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
WORD_TOLERANCE     = 20    # rewritten text must stay within +/- this many words
MAX_LENGTH_REPROMPTS = 3   # retry count within the same attempt for length violations

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_PATH = "data/hippocorpus_test_truncated_80_100.csv"
RESULTS_DIR  = "results"
MODEL_PATH   = "model/modernbert_trained"
