import os

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

from config import MODEL_PATH, MAX_ATTEMPTS, RESULTS_DIR
from dao import AttackSequence, AttemptResult

# Resolve .env relative to project root (one level above src/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

# ---------------------------------------------------------------------------
# Classifier (loaded once at import time)
# ---------------------------------------------------------------------------
_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
_clf_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH, use_safetensors=True
)
_clf_model.eval()


def classify(text):
    """Run ModernBERT classifier. Returns (label: int, confidence_pct: float).

    The raw model outputs: 0 = truthful, 1 = deceptive.
    We normalize to the study convention (matching the dataset and JS interface):
      0 = deceptive, 1 = truthful
    """
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = _clf_model(**inputs)
    probs = outputs.logits.softmax(dim=-1)
    raw_label = int(probs.argmax().item())
    confidence = float(probs[0, raw_label].item()) * 100
    # Flip raw label to study convention: raw 0 (truthful) -> 1, raw 1 (deceptive) -> 0
    label = 1 - raw_label
    return label, confidence


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(path):
    return pd.read_csv(path).reset_index(drop=True)


def sample_statements(df, n, seed):
    """Sample n rows with replacement for reproducibility."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(df), size=n)
    return df.iloc[indices].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

_LABEL_MAP = {0: "deceptive", 1: "truthful"}


def generate_attack_prompt(sequence):
    """Build the zero-shot attack prompt including full history of previous attempts."""
    original_label_str = _LABEL_MAP[sequence.original_label]
    target_label_str   = _LABEL_MAP[1 - sequence.original_label]
    original_length    = len(sequence.original_text.split())

    prompt = (
        "Your task is to rewrite a statement so that a machine learning model changes its prediction.\n"
        "You will see the original statement (taken from a previous study on truthful or deceptive "
        "autobiographical statements), the prediction of a machine learning model (whether the predicted "
        "label is truthful or deceptive) and the prediction confidence (i.e., the class probabilities of "
        "the statement belonging to the predicted class).\n"
        "You will also see any previous attempts (if there are any, as well as the prediction that "
        "resulted from these).\n"
        f"Original statement: {sequence.original_text}\n"
        f"Machine learning prediction (original statement): {original_label_str}\n"
        f"Prediction confidence (original statement): {sequence.original_confidence:.0f}%\n"
    )

    if sequence.attempts:
        prompt += f"\nPrevious attempts: {len(sequence.attempts)}\n"
        for idx, attempt in enumerate(sequence.attempts, 1):
            prompt += (
                f"\nStatement after attempt {idx}: {attempt.text}\n"
                f"Machine learning prediction (after attempt {idx}): {_LABEL_MAP[attempt.label]}\n"
                f"Prediction confidence (after attempt {idx}): {attempt.confidence:.0f}%\n"
            )

    prompt += (
        f"\nNow rewrite the original statement (taking into account any previous attempts listed above) "
        f"so that it appears more {target_label_str} to the machine learning classifier. "
        "Maintain the original statement's meaning, ensure it is grammatically correct, and appears "
        "natural (i.e., that it is readable, coherent, and fluent). "
        f"Ensure that the version is within +-20 words of the length of the original statement "
        f"({original_length} words).\n"
        "Your modification:"
    )
    return prompt


def generate_strategy_prompt(sequence):
    """Prompt asking the LLM to describe its strategy after completing the sequence."""
    target_label_str = _LABEL_MAP[1 - sequence.original_label]
    best = _find_best_attempt(sequence)
    return (
        f"You just completed a paraphrasing task. Your goal was to rewrite a statement to appear more "
        f"{target_label_str} to a machine learning deception classifier.\n"
        f"Your best modification was: \"{best.text}\"\n"
        "In 2-3 sentences, describe what strategy or strategies you used to achieve this goal."
    )


def _find_best_attempt(sequence):
    """Return the attempt that best achieved the flip goal."""
    target_label = 1 - sequence.original_label
    flipped = [a for a in sequence.attempts if a.label == target_label]
    if flipped:
        return max(flipped, key=lambda a: a.confidence)
    same_label = [a for a in sequence.attempts if a.label == sequence.original_label]
    if same_label:
        return min(same_label, key=lambda a: a.confidence)
    return sequence.attempts[-1]


def label_flipped(sequence, attempt):
    return attempt.label != sequence.original_label


# ---------------------------------------------------------------------------
# CSV output (wide format matching the human all_sessions.csv structure)
# ---------------------------------------------------------------------------

def _build_columns():
    cols = [
        "session_id", "session_start", "session_end", "total_duration_ms",
        "statement_id", "original_text", "original_label", "original_confidence",
        "attempts_used", "max_attempts",
    ]
    for i in range(1, MAX_ATTEMPTS + 1):
        cols += [
            f"rewrite{i}_text",
            f"rewrite{i}_label",
            f"rewrite{i}_confidence",
            f"rewrite{i}_duration_ms",
            f"rewrite{i}_prompt",
        ]
    cols += ["strategies", "received_at", "attack_modality", "llm_architecture", "temperature"]
    return cols


def init_results_csv(architecture, timestamp, subdir=None):
    """Create an empty results CSV and return its path."""
    out_dir = os.path.join(RESULTS_DIR, subdir) if subdir else RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{architecture}_{timestamp}.csv")
    pd.DataFrame(columns=_build_columns()).to_csv(path, index=False)
    return path


def append_sequence_to_csv(path, sequence):
    """Append one completed AttackSequence as a row to the results CSV."""
    row = {
        "session_id":          sequence.session_id,
        "session_start":       sequence.session_start,
        "session_end":         sequence.session_end,
        "total_duration_ms":   sequence.total_duration_ms,
        "statement_id":        sequence.statement_id,
        "original_text":       sequence.original_text,
        "original_label":      sequence.original_label,
        "original_confidence": round(sequence.original_confidence, 2),
        "attempts_used":       len(sequence.attempts),
        "max_attempts":        MAX_ATTEMPTS,
        "strategies":          sequence.strategies,
        "received_at":         sequence.session_end,
        "attack_modality":     "llm",
        "llm_architecture":    sequence.llm_architecture,
        "temperature":         round(sequence.temperature, 4),
    }

    for i, attempt in enumerate(sequence.attempts, 1):
        row[f"rewrite{i}_text"]        = attempt.text
        row[f"rewrite{i}_label"]       = attempt.label
        row[f"rewrite{i}_confidence"]  = round(attempt.confidence, 2)
        row[f"rewrite{i}_duration_ms"] = attempt.duration_ms
        row[f"rewrite{i}_prompt"]      = attempt.prompt

    for i in range(len(sequence.attempts) + 1, MAX_ATTEMPTS + 1):
        row[f"rewrite{i}_text"]        = ""
        row[f"rewrite{i}_label"]       = ""
        row[f"rewrite{i}_confidence"]  = ""
        row[f"rewrite{i}_duration_ms"] = ""
        row[f"rewrite{i}_prompt"]      = ""

    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=False)
