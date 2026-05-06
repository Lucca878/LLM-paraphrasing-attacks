import os
import re

from openai import OpenAI
from dotenv import load_dotenv

from config import LLM_PROVIDER, LLM_ARCHITECTURES

# Cache successful compatibility modes so we do not re-probe on every request.
_DEVELOPER_ROLE_CACHE: dict[tuple[str, str], bool] = {}
_GEN_CAP_CACHE: dict[tuple[str, str], dict] = {}

# Provider-aware defaults to avoid expensive first-call probing.
_DEFAULT_DEVELOPER_ROLE_BY_PROVIDER = {
    "together": False,
    "ollama": True,
}


def _client_timeout_seconds() -> float:
    """Return request timeout in seconds, overridable via env."""
    default = "120" if LLM_PROVIDER == "together" else "300"
    raw = os.environ.get("LLM_API_TIMEOUT_SECONDS", default)
    try:
        val = float(raw)
        return val if val > 0 else float(default)
    except ValueError:
        return float(default)


def _client_max_retries() -> int:
    """Return SDK retry count, overridable via env."""
    raw = os.environ.get("LLM_API_MAX_RETRIES", "1")
    try:
        val = int(raw)
        return max(0, val)
    except ValueError:
        return 1

# Resolve .env relative to project root (one level above src/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"))


def _build_client() -> OpenAI:
    """Return an OpenAI-compatible client for the configured provider."""
    if LLM_PROVIDER == "together":
        return OpenAI(
            api_key=os.environ["TOGETHER_API_KEY"],
            base_url="https://api.together.xyz/v1",
            timeout=_client_timeout_seconds(),
            max_retries=_client_max_retries(),
        )
    # Default: Ollama exposes an OpenAI-compatible endpoint
    return OpenAI(
        api_key="ollama",
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        timeout=_client_timeout_seconds(),
        max_retries=_client_max_retries(),
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
    cache_key = (LLM_PROVIDER, model_id)

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

    def _params_from_capability(capability: dict) -> dict:
        params = {}
        max_key = capability.get("max_key")
        if max_key and max_tokens is not None:
            params[max_key] = max_tokens
        if capability.get("use_min") and min_tokens is not None:
            params["min_tokens"] = min_tokens
        if capability.get("use_stop") and stop_sequences:
            params["stop"] = stop_sequences
        return params

    def _generation_variants():
        """Return ordered generation-parameter variants for provider compatibility."""
        variants = []

        cached = _GEN_CAP_CACHE.get(cache_key)
        if cached is not None:
            variants.append(_params_from_capability(cached))

        if LLM_PROVIDER == "together":
            # Together's OpenAI-compatible chat endpoint reliably supports max_tokens;
            # probing max_new_tokens/min_tokens often adds avoidable latency.
            default_caps = [
                {"max_key": "max_tokens", "use_min": False, "use_stop": True},
                {"max_key": "max_tokens", "use_min": False, "use_stop": False},
                {"max_key": None, "use_min": False, "use_stop": False},
            ]
        else:
            default_caps = [
                {"max_key": "max_tokens", "use_min": True, "use_stop": True},
                {"max_key": "max_tokens", "use_min": False, "use_stop": True},
                {"max_key": "max_tokens", "use_min": False, "use_stop": False},
                {"max_key": "max_new_tokens", "use_min": False, "use_stop": False},
                {"max_key": None, "use_min": False, "use_stop": False},
            ]
        for cap in default_caps:
            variants.append(_params_from_capability(cap))

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
                response = client.chat.completions.create(**kwargs)
                # Persist successful parameter capability for subsequent calls.
                capability = {
                    "max_key": "max_new_tokens" if "max_new_tokens" in params else ("max_tokens" if "max_tokens" in params else None),
                    "use_min": "min_tokens" in params,
                    "use_stop": "stop" in params,
                }
                _GEN_CAP_CACHE[cache_key] = capability
                return response
            except Exception as err:
                last_err = err
        raise last_err

    role_preference = _DEVELOPER_ROLE_CACHE.get(cache_key)
    if role_preference is None:
        role_preference = _DEFAULT_DEVELOPER_ROLE_BY_PROVIDER.get(LLM_PROVIDER, True)

    role_order = [role_preference] if developer_prompt else [False]
    if developer_prompt:
        role_order.append(not role_preference)
    if developer_prompt is None:
        role_order = [False]

    last_err = None
    response = None
    for use_developer_role in role_order:
        try:
            response = _create_with_variants(_build_messages(use_developer_role=use_developer_role))
            _DEVELOPER_ROLE_CACHE[cache_key] = use_developer_role
            break
        except Exception as err:
            last_err = err
            continue
    if response is None:
        raise last_err

    cleaned = _clean_response(response.choices[0].message.content)
    if cleaned:
        return cleaned

    # One lightweight retry for empty generations, which occasionally occur
    # on remote providers under load.
    response_retry = _create_with_variants(_build_messages(use_developer_role=_DEVELOPER_ROLE_CACHE.get(cache_key, False)))
    cleaned_retry = _clean_response(response_retry.choices[0].message.content)
    if cleaned_retry:
        return cleaned_retry
    raise RuntimeError("LLM returned an empty response twice")
