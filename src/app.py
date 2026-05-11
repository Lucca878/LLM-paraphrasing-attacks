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
    label_flipped,
    init_results_csv,
    append_sequence_to_csv,
)

load_dotenv()

def _attack_system_prompt() -> str:
    """System prompt for attack role/context."""
    return (
        "You are a participant in a study called 'Rewriting truths and lies'. "
        "Your task is to interact with an AI model that has been trained to identify statements "
        "written by human participants as truthful or deceptive."
    )


def _attack_developer_prompt(original_text: str) -> str:
    """Developer prompt that enforces output format and word-length constraints."""
    n = len(original_text.split())
    lo = max(1, n - WORD_TOLERANCE)
    hi = n + WORD_TOLERANCE - 5  # Keep slight headroom so outputs can end naturally.
    return (
        f"STRICT OUTPUT RULES:\n"
        f"- Output ONLY the requested modification. No preamble, no commentary, no word count.\n"
        f"- The modification must be at least {lo} words and at most {hi} words.\n"
        f"- End with a complete sentence and terminal punctuation (. ! or ?).\n"
    )


SYSTEM_PROMPT_STRATEGY = (
    "You are a participant in a study called 'Rewriting truths and lies'. "
    "Your task was to interact with an AI model that has been trained to identify statements "
    "written by human participants as truthful or deceptive. "
    "You just completed the main task and are now answering a follow-up question about your approach."
)
DEVELOPER_PROMPT_STRATEGY = (
    "Describe only the strategy used. "
    "Output exactly 2 to 3 complete sentences. "
    "Do not add any explanation, preamble, or commentary in your response. "
)


def run_attack_sequence(
    architecture,
    statement_row,
    temperature,
    max_attempts=MAX_ATTEMPTS,
    use_max_tokens=True,
):
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

    # Token range derived from shared cl100k token counting.
    min_tok, max_tok = words_to_token_range(
        sequence.original_text,
        WORD_TOLERANCE,
        architecture,
    )

    for _ in range(max_attempts):
        prompt = generate_attack_prompt(sequence)
        attack_sys = _attack_system_prompt()
        attack_dev = _attack_developer_prompt(sequence.original_text)
        attempt_start = time.time()
        rewrite_text = call_llm(
            architecture,
            prompt,
            temperature,
            attack_sys,
            developer_prompt=attack_dev,
            max_tokens=max_tok if use_max_tokens else None,
        )
        duration_ms = int((time.time() - attempt_start) * 1000)

        label, confidence = classify(rewrite_text)
        attempt = AttemptResult(
            text=rewrite_text,
            label=label,
            confidence=confidence,
            duration_ms=duration_ms,
            prompt=prompt,
        )
        sequence.attempts.append(attempt)

        if label_flipped(sequence, attempt):
            break

    # Ask the LLM to reflect on its strategy (matches human study design)
    strategy_prompt          = generate_strategy_prompt(sequence)
    sequence.strategy_prompt = strategy_prompt
    sequence.strategies      = call_llm(
        architecture,
        strategy_prompt,
        sequence.temperature,
        SYSTEM_PROMPT_STRATEGY,
        developer_prompt=DEVELOPER_PROMPT_STRATEGY,
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
        help="Run a quick smoke-test: 10 sequences, max 3 attempts each, results in results/test/",
    )
    parser.add_argument(
        "--no-max-tokens",
        action="store_true",
        help="Disable max_tokens cap for attack generations (useful for tokenizer mismatch tests).",
    )
    args = parser.parse_args()

    if args.test:
        args.n_sequences = 10

    test_max_attempts = 3 if args.test else MAX_ATTEMPTS
    results_subdir    = "test" if args.test else None

    if args.test:
        print("[TEST MODE] 10 sequences, max 3 attempts each, output -> results/test/")
    if args.no_max_tokens:
        print("[TOKEN CAP OFF] max_tokens is disabled for attack generations")

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

            sequence = run_attack_sequence(
                architecture,
                row,
                temperature,
                max_attempts=test_max_attempts,
                use_max_tokens=not args.no_max_tokens,
            )
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
