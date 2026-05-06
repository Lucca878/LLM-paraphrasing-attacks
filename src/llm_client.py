import os
import re

from openai import OpenAI
from dotenv import load_dotenv

from config import LLM_PROVIDER, LLM_ARCHITECTURES

# Resolve .env relative to project root (one level above src/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))


def _build_client() -> OpenAI:
    """Return an OpenAI-compatible client for the configured provider."""
    if LLM_PROVIDER == "together":
        return OpenAI(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
        )
    # Default: Ollama exposes an OpenAI-compatible endpoint
    return OpenAI(
        api_key="ollama",
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    )


def _get_model_id(architecture: str) -> str:
    key = "together_model" if LLM_PROVIDER == "together" else "ollama_model"
    return LLM_ARCHITECTURES[architecture][key]


def _clean_response(text: str) -> str:
    """Strip thinking blocks (e.g. Qwen3 <think>…</think>) and trim whitespace."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


# Approximate token count from word count (English prose: ~1.3 tokens/word)
_TOKENS_PER_WORD = 1.3


def words_to_token_range(word_count: int, tolerance: int) -> tuple[int, int]:
    """Convert a word-count ± tolerance into a (min_tokens, max_tokens) pair."""
    lo = max(1, word_count - tolerance)
    hi = word_count + tolerance
    return int(lo * _TOKENS_PER_WORD), int(hi * _TOKENS_PER_WORD)


def call_llm(
    architecture: str,
    prompt: str,
    temperature: float,
    system_prompt: str = None,
    developer_prompt: str = None,
    min_tokens: int = None,
    max_tokens: int = None,
    stop_sequences: list[str] = None,
) -> str:
    """Call the LLM and return the cleaned text response."""
    client = _build_client()
    model_id = _get_model_id(architecture)

    def _build_messages(use_developer_role: bool):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if developer_prompt:
            if use_developer_role:
                messages.append({"role": "developer", "content": developer_prompt})
            else:
                merged = developer_prompt if not system_prompt else f"{system_prompt}\n\n{developer_prompt}"
                messages = [{"role": "system", "content": merged}]
        messages.append({"role": "user", "content": prompt})
        return messages

    def _generation_variants():
        """Return ordered generation-parameter variants for provider compatibility."""
        variants = []
        max_key_options = ["max_tokens"]
        if max_tokens is not None:
            max_key_options.append("max_new_tokens")

        for max_key in max_key_options:
            variant = {}
            if max_tokens is not None:
                variant[max_key] = max_tokens
            if min_tokens is not None:
                variant["min_tokens"] = min_tokens
            if stop_sequences:
                variant["stop"] = stop_sequences
            variants.append(variant)

            # Common fallback: unsupported min_tokens.
            if min_tokens is not None:
                variant_no_min = dict(variant)
                variant_no_min.pop("min_tokens", None)
                variants.append(variant_no_min)

            # Common fallback: unsupported stop parameter.
            if stop_sequences:
                variant_no_stop = dict(variant)
                variant_no_stop.pop("stop", None)
                variants.append(variant_no_stop)

        # Last-resort fallback with no generation controls.
        variants.append({})

        deduped = []
        seen = set()
        for v in variants:
            key = tuple(
                sorted(
                    (k, tuple(val) if isinstance(val, list) else val)
                    for k, val in v.items()
                )
            )
            if key not in seen:
                deduped.append(v)
                seen.add(key)
        return deduped

    def _create_with_variants(messages):
        last_err = None
        base = dict(model=model_id, messages=messages, temperature=temperature)
        for params in _generation_variants():
            kwargs = dict(base)
            kwargs.update(params)
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as err:
                last_err = err
        raise last_err

    try:
        response = _create_with_variants(_build_messages(use_developer_role=True))
    except Exception:
        if developer_prompt is None:
            raise
        # Fallback for providers that do not accept role="developer".
        response = _create_with_variants(_build_messages(use_developer_role=False))

    return _clean_response(response.choices[0].message.content)
