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
    min_tokens: int = None,
    max_tokens: int = None,
) -> str:
    """Call the LLM and return the cleaned text response."""
    client = _build_client()
    model_id = _get_model_id(architecture)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(model=model_id, messages=messages, temperature=temperature)
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    # min_tokens is not reliably supported across providers; lower bound is
    # enforced via system prompt and reprompt loop instead.

    response = client.chat.completions.create(**kwargs)
    return _clean_response(response.choices[0].message.content)
