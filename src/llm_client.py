import os
import re
import time

import tiktoken
from openrouter import OpenRouter
from openrouter import errors as openrouter_errors
from dotenv import load_dotenv

from config import (
    LLM_ARCHITECTURES,
    QWEN_MAXTOK_MULTIPLIER,
    QWEN_MAXTOK_ADDON,
    TOKENS_PER_WORD,
)


# Resolve .env relative to project root (one level above src/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))

# OpenRouter uses cl100k_base for token counting (OpenAI-compatible API layer)
_enc = tiktoken.get_encoding("cl100k_base")
def count_tokens_exact(text: str) -> int:
    """Count tokens with cl100k_base tokenizer."""
    return len(_enc.encode(text))


def _build_client() -> OpenRouter:
    """Return an OpenRouter SDK client."""
    return OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])


def _get_model_id(architecture: str) -> str:
    return LLM_ARCHITECTURES[architecture]["model"]


def _clean_response(text: str) -> str:
    """Strip model meta-output and return only the paraphrase text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    return text.strip()


def _is_transient_validation_error(exc: Exception) -> bool:
    """Detect OpenRouter validation wrappers for transient upstream errors."""
    msg = str(exc).lower()
    transient_markers = (
        "code': 429",
        'code": 429',
        "code': 502",
        'code": 502',
        "code': 503",
        'code": 503',
        "code': 504",
        'code": 504',
        "code': 524",
        'code": 524',
        "code': 529",
        'code": 529',
        "timed out",
        "request timeout",
        "provider overloaded",
        "provider aborted",
    )
    return any(marker in msg for marker in transient_markers)


def words_to_token_range(text: str, tolerance: int, architecture: str) -> tuple[int, int]:
    """Return (min_tokens, max_tokens) based on cl100k token count ± tolerance.

    The token tolerance is scaled by TOKENS_PER_WORD to match the word
    tolerance in the developer prompt.
    """
    orig = count_tokens_exact(text)
    tok_tolerance = round(tolerance * TOKENS_PER_WORD)
    min_tok = max(1, orig - tok_tolerance)
    max_tok = orig + tok_tolerance

    if architecture == "qwen3":
        max_tok = int(max_tok * QWEN_MAXTOK_MULTIPLIER) + QWEN_MAXTOK_ADDON

    return min_tok, max_tok


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
    combined_system = "\n\n".join(filter(None, [system_prompt, developer_prompt]))
    if combined_system:
        messages.append({"role": "system", "content": combined_system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {"model": model_id, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.send(**kwargs)
            return _clean_response(response.choices[0].message.content)
        except (
            openrouter_errors.TooManyRequestsResponseError,
            openrouter_errors.RequestTimeoutResponseError,
            openrouter_errors.BadGatewayResponseError,
            openrouter_errors.ServiceUnavailableResponseError,
            openrouter_errors.ProviderOverloadedResponseError,
            openrouter_errors.EdgeNetworkTimeoutResponseError,
        ):
            if attempt == max_retries - 1:
                raise
            wait_seconds = 2 ** attempt
            print(f"    [transient] provider/network issue; retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)
        except openrouter_errors.ResponseValidationError as exc:
            if not _is_transient_validation_error(exc) or attempt == max_retries - 1:
                raise
            wait_seconds = 2 ** attempt
            print(f"    [transient-parse] upstream error wrapped by SDK; retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)
