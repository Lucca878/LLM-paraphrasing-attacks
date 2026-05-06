import argparse
import datetime
import time
import uuid

import numpy as np
from dotenv import load_dotenv

from config import (
    LLM_ARCHITECTURES,
    ARCH_SEEDS,
    MAX_ATTEMPTS,
    TEMPERATURE_MIN,
    TEMPERATURE_MAX,
    N_ATTACK_SEQUENCES,
    RANDOM_SEED,
    DATASET_PATH,
    MAX_LENGTH_REPROMPTS,
    WORD_TOLERANCE,
)
from dao import AttackSequence, AttemptResult
from llm_client import call_llm, words_to_token_range
from utility import (
    classify,
    load_dataset,
    sample_statements,
    generate_attack_prompt,
    generate_strategy_prompt,
    build_length_reprompt_prompt,
    count_words,
    is_within_word_limit,
    label_flipped,
    init_results_csv,
    append_sequence_to_csv,
)
from config import WORD_TOLERANCE

load_dotenv()

def _attack_system_prompt(original_text: str) -> str:
    """System prompt that hard-enforces the word-length constraint for the current statement."""
    n = count_words(original_text)
    lo = max(1, n - WORD_TOLERANCE)
    hi = n + WORD_TOLERANCE
    return (
        "You are a participant in a study called 'Rewriting truths and lies'. "
        "Your task is to interact with an AI model that has been trained to identify statements "
        "written by human participants as truthful or deceptive. "
        "Only give me the requested paraphrase and nothing else. "
        "Do not add any explanation, preamble, or commentary. "
        f"Your response MUST be between {lo} and {hi} words (the original has {n} words). "
        "Strictly respect this word count — do not go shorter or longer."
    )
SYSTEM_PROMPT_STRATEGY = (
    "You are a participant in a study called 'Rewriting truths and lies'. "
    "Your task was to interact with an AI model that has been trained to identify statements "
    "written by human participants as truthful or deceptive. "
    "You just completed the main task and are now answering a follow-up question about your approach. "
    "Describe only the strategy used. "
    "Output exactly 2 to 3 complete sentences. "
    "Do not add any explanation, preamble, or commentary. "
)


def run_attack_sequence(architecture, statement_row, temperature, max_attempts=MAX_ATTEMPTS):
    session_id  = str(uuid.uuid4())
    start_time  = time.time()
    start_iso   = datetime.datetime.utcnow().isoformat() + "Z"

    sequence = AttackSequence(
        session_id=session_id,
        statement_id=int(statement_row["index"]),
        original_text=statement_row["text_truncated"],
        original_label=int(statement_row["modern_bert_label_numeric"]),
        original_confidence=float(statement_row["modern_bert_class_prob"]) * 100,
        llm_architecture=architecture,
        temperature=temperature,
    )
    sequence.session_start = start_iso

    # Token range derived from original statement length — passed to every attack call
    orig_word_count = count_words(sequence.original_text)
    min_tok, max_tok = words_to_token_range(orig_word_count, WORD_TOLERANCE)

    for _ in range(max_attempts):
        prompt = generate_attack_prompt(sequence)
        effective_prompt = prompt
        attack_sys = _attack_system_prompt(sequence.original_text)
        attempt_start = time.time()
        rewrite_text = call_llm(architecture, effective_prompt, temperature, attack_sys,
                                min_tokens=min_tok, max_tokens=max_tok)

        # Enforce length constraint by reprompting within the same attempt.
        length_reprompt_used = ""
        for _ in range(MAX_LENGTH_REPROMPTS):
            if is_within_word_limit(sequence.original_text, rewrite_text, tolerance=WORD_TOLERANCE):
                break
            effective_prompt = build_length_reprompt_prompt(
                sequence.original_text,
                rewrite_text,
                target_label_str={0: "deceptive", 1: "truthful"}[1 - sequence.original_label],
                attempts_used=len(sequence.attempts),
                max_attempts=max_attempts,
                tolerance=WORD_TOLERANCE,
            )
            length_reprompt_used = effective_prompt
            rewrite_text = call_llm(architecture, effective_prompt, temperature, attack_sys,
                                    min_tokens=min_tok, max_tokens=max_tok)

        duration_ms = int((time.time() - attempt_start) * 1000)

        if not is_within_word_limit(sequence.original_text, rewrite_text, tolerance=WORD_TOLERANCE):
            print(
                "    warning=length_constraint_not_met "
                f"orig_words={count_words(sequence.original_text)} "
                f"rewrite_words={count_words(rewrite_text)}"
            )

        label, confidence = classify(rewrite_text)
        attempt = AttemptResult(
            text=rewrite_text,
            label=label,
            confidence=confidence,
            duration_ms=duration_ms,
            prompt=prompt,  # always store the original attack prompt, not a reprompt
            length_reprompt=length_reprompt_used,
        )
        sequence.attempts.append(attempt)

        if label_flipped(sequence, attempt):
            break

    # Ask the LLM to reflect on its strategy (matches human study design)
    strategy_prompt          = generate_strategy_prompt(sequence)
    sequence.strategy_prompt = strategy_prompt
    sequence.strategies      = call_llm(
        architecture, strategy_prompt, sequence.temperature, SYSTEM_PROMPT_STRATEGY
    )

    sequence.session_end      = datetime.datetime.utcnow().isoformat() + "Z"
    sequence.total_duration_ms = int((time.time() - start_time) * 1000)
    return sequence


def main():
    parser = argparse.ArgumentParser(
        description="LLM paraphrasing attack pipeline"
    )
    parser.add_argument(
        "--architecture",
        choices=list(LLM_ARCHITECTURES.keys()) + ["all"],
        default="all",
        help="Which LLM architecture to run (default: all)",
    )
    parser.add_argument(
        "--n-sequences",
        type=int,
        default=N_ATTACK_SEQUENCES,
        help=f"Number of attack sequences per architecture (default: {N_ATTACK_SEQUENCES})",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Resume from this sequence index (0-based, for crash recovery)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick smoke-test: 10 sequences, max 2 attempts each, results in results/test/",
    )
    args = parser.parse_args()

    if args.test:
        args.n_sequences = 10

    test_max_attempts = 3 if args.test else MAX_ATTEMPTS
    results_subdir    = "test" if args.test else None

    if args.test:
        print("[TEST MODE] 10 sequences, max 3 attempts each, output -> results/test/")

    timestamp  = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    df         = load_dataset(DATASET_PATH)

    # Same statement sample across all architectures for fair comparison
    sampled = sample_statements(df, args.n_sequences, seed=RANDOM_SEED)

    archs = list(LLM_ARCHITECTURES.keys()) if args.architecture == "all" else [args.architecture]

    for architecture in archs:
        print(f"\n=== Architecture: {architecture} ===")

        # Independently sampled temperatures per architecture (reproducible)
        arch_rng     = np.random.default_rng(ARCH_SEEDS[architecture])
        temperatures = arch_rng.uniform(TEMPERATURE_MIN, TEMPERATURE_MAX, size=args.n_sequences)

        csv_path = init_results_csv(architecture, timestamp, subdir=results_subdir)
        print(f"Results -> {csv_path}")

        for i, (_, row) in enumerate(sampled.iterrows()):
            if i < args.start_index:
                continue

            temperature = float(temperatures[i])
            print(
                f"  [{architecture}] {i + 1}/{args.n_sequences} | "
                f"statement_id={int(row['index'])} | "
                f"label={int(row['modern_bert_label_numeric'])} | "
                f"temp={temperature:.3f}"
            )

            sequence = run_attack_sequence(architecture, row, temperature, max_attempts=test_max_attempts)
            append_sequence_to_csv(csv_path, sequence)

            flipped = (
                sequence.attempts[-1].label != sequence.original_label
                if sequence.attempts else False
            )
            print(
                f"    attempts={len(sequence.attempts)} | "
                f"flipped={flipped} | "
                f"duration={sequence.total_duration_ms}ms"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
