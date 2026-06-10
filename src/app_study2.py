import argparse
import datetime
import os
import time
import uuid

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config import (
    ARCH_SEEDS,
    DATASET_PATH,
    LLM_ARCHITECTURES,
    RANDOM_SEED,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    WORD_TOLERANCE,
)
from dao import AttackSequence, AttemptResult
from llm_client import call_llm, words_to_token_range
from utility import (
    classify,
    label_flipped,
    load_dataset,
    sample_statements,
)

load_dotenv()

STUDY2_ARCHITECTURE = "llama3.3"
STUDY2_N_SEQUENCES = 330
STUDY2_MAX_ATTEMPTS = 1

_LABEL_MAP = {0: "deceptive", 1: "truthful"}


def _attack_system_prompt() -> str:
    return (
        "You are a participant in a study and have to paraphrase a given statement. "
    )


def _attack_developer_prompt(original_text: str) -> str:
    n = len(original_text.split())
    lo = max(1, n - WORD_TOLERANCE)
    hi = n + WORD_TOLERANCE - 10
    return (
        f"STRICT OUTPUT RULES:\n"
        f"- Output ONLY the requested modification. No preamble, no commentary, no word count.\n"
        f"- Use English only. Do not use non-Latin scripts or non-English words.\n"
        f"- The modification must be at least {lo} words and at most {hi} words.\n"
        f"- End with a complete sentence and terminal punctuation (. ! or ?).\n"
    )


def generate_study2_attack_prompt(sequence):
    """Study 2 prompt copy, isolated from Study 1 so it can be edited safely."""
    original_label_str = _LABEL_MAP[sequence.original_label]
    target_label_str = _LABEL_MAP[1 - sequence.original_label]
    orig_words = len(sequence.original_text.split())
    min_words = max(1, orig_words - WORD_TOLERANCE)
    max_words = orig_words + WORD_TOLERANCE - 10

    prompt = (
        "In this study, you will read 1 short statement written by participants in another study. "
        "Your task is to modify the original statement. Specifically, we want you to try to paraphrase the statement. "
        "There is one very important condition to be met: You must maintain the meaning of the original "
        "statement. In other words, in your modification, core facts, events, or the overall claim of the "
        "original statement should remain recognizable. Your modification should also be grammatically "
        "correct and appear natural.\n\n"
        f"Original statement:\n\n{sequence.original_text}\n\n"
        f"Task goal:\n\nRewrite the statement while preserving meaning, grammar, and naturalness.\n\n"
        "Language requirement: Write the modification in English only. Do not use non-Latin scripts.\n"
        f"Your modification must be at least {min_words} words and at most {max_words} words.\n"
    )

    prompt += "\nYour modification:\n"
    return prompt


def _study2_columns():
    return [
        "session_id",
        "session_start",
        "session_end",
        "total_duration_ms",
        "statement_id",
        "original_text",
        "original_label",
        "original_confidence",
        "attempts_used",
        "max_attempts",
        "rewrite1_text",
        "rewrite1_label",
        "rewrite1_confidence",
        "rewrite1_duration_ms",
        "rewrite1_prompt",
        "received_at",
        "attack_modality",
        "llm_architecture",
        "temperature",
    ]


def _init_study2_csv(timestamp: str) -> str:
    out_dir = os.path.join("results", "study2")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{STUDY2_ARCHITECTURE}_{timestamp}.csv")
    pd.DataFrame(columns=_study2_columns()).to_csv(path, index=False)
    return path


def _latest_study2_csv() -> str | None:
    out_dir = os.path.join("results", "study2")
    if not os.path.isdir(out_dir):
        return None
    files = [
        os.path.join(out_dir, f)
        for f in os.listdir(out_dir)
        if f.startswith(f"{STUDY2_ARCHITECTURE}_") and f.endswith(".csv")
    ]
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def _count_csv_rows(path: str) -> int:
    rows = 0
    with open(path, newline="") as fh:
        for _ in pd.read_csv(fh, chunksize=10000):
            rows += len(_)
    return rows


def _append_study2_sequence(path: str, sequence: AttackSequence):
    attempt = sequence.attempts[0] if sequence.attempts else None
    row = {
        "session_id": sequence.session_id,
        "session_start": sequence.session_start,
        "session_end": sequence.session_end,
        "total_duration_ms": sequence.total_duration_ms,
        "statement_id": sequence.statement_id,
        "original_text": sequence.original_text,
        "original_label": sequence.original_label,
        "original_confidence": round(sequence.original_confidence, 2),
        "attempts_used": len(sequence.attempts),
        "max_attempts": STUDY2_MAX_ATTEMPTS,
        "rewrite1_text": attempt.text if attempt else "",
        "rewrite1_label": attempt.label if attempt else "",
        "rewrite1_confidence": round(attempt.confidence, 2) if attempt else "",
        "rewrite1_duration_ms": attempt.duration_ms if attempt else "",
        "rewrite1_prompt": attempt.prompt if attempt else "",
        "received_at": sequence.session_end,
        "attack_modality": "llm",
        "llm_architecture": sequence.llm_architecture,
        "temperature": round(sequence.temperature, 4),
    }
    pd.DataFrame([row]).to_csv(path, mode="a", index=False, header=False)


def run_study2_sequence(statement_row, temperature, use_max_tokens=True):
    session_id = str(uuid.uuid4())
    start_time = time.time()
    start_iso = datetime.datetime.utcnow().isoformat() + "Z"

    sequence = AttackSequence(
        session_id=session_id,
        statement_id=int(statement_row["index"]),
        original_text=statement_row["text_truncated"],
        original_label=int(statement_row["modern_bert_label_numeric"]),
        original_confidence=float(statement_row["modern_bert_class_prob"]) * 100,
        llm_architecture=STUDY2_ARCHITECTURE,
        temperature=temperature,
    )
    sequence.session_start = start_iso

    _, max_tok = words_to_token_range(
        sequence.original_text,
        WORD_TOLERANCE,
        STUDY2_ARCHITECTURE,
    )

    prompt = generate_study2_attack_prompt(sequence)
    attack_sys = _attack_system_prompt()
    attack_dev = _attack_developer_prompt(sequence.original_text)

    attempt_start = time.time()
    rewrite_text = call_llm(
        STUDY2_ARCHITECTURE,
        prompt,
        temperature,
        attack_sys,
        developer_prompt=attack_dev,
        max_tokens=max_tok if use_max_tokens else None,
    )
    duration_ms = int((time.time() - attempt_start) * 1000)

    label, confidence = classify(rewrite_text)
    sequence.attempts.append(
        AttemptResult(
            text=rewrite_text,
            label=label,
            confidence=confidence,
            duration_ms=duration_ms,
            prompt=prompt,
        )
    )

    sequence.session_end = datetime.datetime.utcnow().isoformat() + "Z"
    sequence.total_duration_ms = int((time.time() - start_time) * 1000)
    return sequence


def main():
    parser = argparse.ArgumentParser(
        description="Study 2: one-attempt llama paraphrasing attack pipeline"
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=STUDY2_N_SEQUENCES,
        help=f"Number of attack sequences to run (default: {STUDY2_N_SEQUENCES})",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Resume from this sequence index (0-based). If > 0, appends to latest study2 CSV when available.",
    )
    parser.add_argument(
        "--no-max-tokens",
        action="store_true",
        help="Disable max_tokens cap for attack generation.",
    )
    args = parser.parse_args()

    if STUDY2_ARCHITECTURE not in LLM_ARCHITECTURES:
        raise ValueError(f"Missing architecture config: {STUDY2_ARCHITECTURE}")

    print(
        f"[STUDY 2] architecture=llama3.3 | n={args.n_sequences} | "
        "max_attempts=1 | strategy_prompt=off"
    )
    if args.no_max_tokens:
        print("[TOKEN CAP OFF] max_tokens is disabled for attack generation")

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    df = load_dataset(DATASET_PATH)
    sampled = sample_statements(df, args.n_sequences, seed=RANDOM_SEED)

    arch_rng = np.random.default_rng(ARCH_SEEDS[STUDY2_ARCHITECTURE])
    temperatures = arch_rng.uniform(TEMPERATURE_MIN, TEMPERATURE_MAX, size=args.n_sequences)

    existing_rows = 0
    if args.start_index > 0:
        resume_path = _latest_study2_csv()
        if resume_path:
            csv_path = resume_path
            existing_rows = _count_csv_rows(csv_path)
            print(f"Resuming -> {csv_path} (existing rows: {existing_rows})")
        else:
            csv_path = _init_study2_csv(timestamp)
            print(f"[resume warning] no existing file found, creating new -> {csv_path}")
    else:
        csv_path = _init_study2_csv(timestamp)
        print(f"Results -> {csv_path}")

    effective_start = max(args.start_index, existing_rows)
    if effective_start > args.start_index:
        print(
            f"[resume adjust] start-index={args.start_index} but file already has {existing_rows} rows; "
            f"continuing from index {effective_start}"
        )

    for i, (_, row) in enumerate(sampled.iterrows()):
        if i < effective_start:
            continue

        temperature = float(temperatures[i])
        print(
            f"  [{STUDY2_ARCHITECTURE}] {i + 1}/{args.n_sequences} | "
            f"statement_id={int(row['index'])} | "
            f"label={int(row['modern_bert_label_numeric'])} | "
            f"temp={temperature:.3f}"
        )

        sequence = run_study2_sequence(
            row,
            temperature,
            use_max_tokens=not args.no_max_tokens,
        )
        _append_study2_sequence(csv_path, sequence)

        flipped = label_flipped(sequence, sequence.attempts[-1]) if sequence.attempts else False
        print(
            f"    attempts={len(sequence.attempts)} | "
            f"flipped={flipped} | "
            f"duration={sequence.total_duration_ms}ms"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()