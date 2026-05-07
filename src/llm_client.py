import os
import re

from openrouter import OpenRouter
from dotenv import load_dotenv

from config import LLM_ARCHITECTURES


# Resolve .env relative to project root (one level above src/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))


def _build_client() -> OpenRouter:
    """Return an OpenRouter SDK client."""
    return OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])


def _get_model_id(architecture: str) -> str:
    return LLM_ARCHITECTURES[architecture]["model"]


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
    max_tokens: int = None,
) -> str:
    """Call OpenRouter and return the cleaned text response."""
    client = _build_client()
    model_id = _get_model_id(architecture)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if developer_prompt:
        messages.append({"role": "developer", "content": developer_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {"model": model_id, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    response = client.chat.send(**kwargs)
    return _clean_response(response.choices[0].message.content)
