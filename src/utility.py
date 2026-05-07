import os

import tiktoken
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

from config import MODEL_PATH, MAX_ATTEMPTS, RESULTS_DIR, WORD_TOLERANCE
from dao import AttackSequence, AttemptResult

# Resolve .env relative to project root (one level above src/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))

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


def count_words(text):
    return len(text.split())


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------

_LABEL_MAP = {0: "deceptive", 1: "truthful"}


def generate_attack_prompt(sequence):
    """Building the attack prompt mirroring the human study instructions, including full history of previous attempts."""
    original_label_str = _LABEL_MAP[sequence.original_label]
    target_label_str   = _LABEL_MAP[1 - sequence.original_label]
    orig_tok  = _count_tokens(sequence.original_text)
    min_tok   = max(1, orig_tok - WORD_TOLERANCE)
    max_tok   = orig_tok + WORD_TOLERANCE

    prompt = (
        "In this study, you will read 1 short statement written by participants in another study. "
        "This statement is either truthful or deceptive. Truthful statements describe events that have "
        "actually happened, while deceptive statements are made up. Some statements have been shortened.\n\n"
        "You will also see the predictions of a state-of-the-art AI deception detection algorithm, which "
        "was trained on large datasets of truths and lies.\n\n"
        "These predictions show you whether the AI classified the statement as deceptive or truthful and "
        "the confidence of the classification. The closer the confidence score is to 100%, the higher the "
        "confidence of the model's prediction. Confidence values closer to 50% indicate uncertainty. "
        "A confidence of 50% implies the highest possible uncertainty about the decision (i.e., a truthful "
        "and deceptive judgement are equally possible).\n\n"
        "Your task is to modify the original statement. Specifically, we want you to try to paraphrase "
        "the statement so that the model's prediction changes. You will receive live feedback from the AI "
        "model. Your goal is to modify the statement so much that the AI model's original prediction is "
        f"reversed: if the original prediction was '{original_label_str}', your modification should render "
        f"the prediction now as '{target_label_str}'.\n\n"
        f"In total, you have {MAX_ATTEMPTS} attempts to lower the confidence score of the AI's original "
        "prediction as much as possible. If you lower the confidence enough so that the AI model reverses "
        "its original prediction, you will immediately move on to the end of the study.\n\n"
        "There is one very important condition to be met: You must maintain the meaning of the original "
        "statement. In other words, in your modification, core facts, events or the overall claim of the "
        "original statement should remain recognizable. Your modification should also be grammatically "
        "correct and appear natural.\n\n"
        "---\n\n"
        "Task Reminder\n\n"
        "What To Do\n"
        f"Try to flip the model prediction: Rewrite the statement to reduce the AI's confidence so much "
        f"that it changes its initial prediction.\n"
        "Keep the same core meaning: Key facts and events must remain recognizable in your modification.\n"
        "Write naturally: Your modified statement should be grammatically correct and sound natural.\n\n"
        "What Not To Do\n"
        "Do not make the statement obviously unrealistic or absurd just to change the AI prediction. For example:\n"
        "Changing 'my grandmother died at age 76' to 'my grandmother died at age 765'\n"
        "Adding unrealistic details such as 'aliens abducted me on the way home'\n\n"
        "---\n\n"
        "Main Task\n\n"
        f"Original statement\n\n{sequence.original_text}\n\n"
        "Original AI prediction\n\n"
        f"The AI classifies this statement as {original_label_str.upper()}.\n"
        f"Confidence score: {sequence.original_confidence:.2f}%\n\n"
        f"Task goal\n\nModify the statement so that it appears {target_label_str.upper()} to the AI "
        "while preserving meaning, grammar, and naturalness.\n\n"
        f"Your modification must stay within {min_tok} to {max_tok} tokens.\n"
        f"You have used {len(sequence.attempts)} of {MAX_ATTEMPTS} attempts.\n"
    )

    if sequence.attempts:
        prompt += "\nBelow you can see your previous modifications\n"
        for idx, attempt in enumerate(sequence.attempts, 1):
            prompt += (
                f"\nAttempt {idx}\n"
                f"Modified statement: {attempt.text}\n"
                f"AI prediction: {_LABEL_MAP[attempt.label].upper()}\n"
                f"Confidence score: {attempt.confidence:.2f}%\n"
            )

    prompt += "\nYour modification:\n"
    return prompt


def generate_strategy_prompt(sequence):
    """Prompt asking the LLM to report strategy using the same wording as the human question."""
    original_label_str = _LABEL_MAP[sequence.original_label].upper()
    target_label_str = _LABEL_MAP[1 - sequence.original_label].upper()
    best = _find_best_attempt(sequence) if sequence.attempts else None

    return (
        "You have completed the main task of the study. "
        "Your task was to paraphrase an original truthful or deceptive autobiographical statement so the AI would change its initial prediction. "
        "In your modification, you had to preserve the original meaning, grammar, and naturalness, and stay within a similar token length to the original.\n\n"
        f"In this sequence, the original statement was classified as {original_label_str}, and your goal was to make it appear {target_label_str} to the AI.\n\n"
        f"Original statement: {sequence.original_text}\n"
        f"Most successful modification: {best.text if best else ''}\n\n"
        "What strategy did you use so the AI would change its initial prediction?\n\n"
        "Describe your approach in 2-3 sentences:"
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
    cols += ["strategies", "strategy_prompt", "received_at", "attack_modality", "llm_architecture", "temperature"]
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
    # Build row in the same order as _build_columns() so pandas writes
    # values into the correct columns when appending without a header.
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

    # Trailing columns — must come after the rewrite columns (matches _build_columns order)
    row["strategies"]       = sequence.strategies
    row["strategy_prompt"]  = sequence.strategy_prompt
    row["received_at"]      = sequence.session_end
    row["attack_modality"]  = "llm"
    row["llm_architecture"] = sequence.llm_architecture
    row["temperature"]      = round(sequence.temperature, 4)

    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=False)
