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


def call_llm(
    architecture: str,
    prompt: str,
    temperature: float,
    system_prompt: str = None,
) -> str:
    """Call the LLM and return the cleaned text response."""
    client = _build_client()
    model_id = _get_model_id(architecture)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
    )
    return _clean_response(response.choices[0].message.content)
