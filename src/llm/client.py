import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


DEFAULT_TIMEOUT_SECONDS = 90
DEFAULT_MAX_RETRIES = 1


def _get_int_env(
    name: str,
    default: int,
) -> int:
    """
    Read an integer environment variable safely.

    Falls back to the supplied default when the
    variable is missing or invalid.
    """

    raw_value = os.getenv(
        name,
        "",
    ).strip()

    if not raw_value:
        return default

    try:
        return int(
            raw_value
        )

    except ValueError:
        raise RuntimeError(
            f"{name} must be an integer."
        ) from None


def get_llm(
    temperature: float = 0.0,
) -> ChatOpenAI:
    """
    Create the shared OpenRouter-backed LLM client.

    Requests use a bounded timeout and limited retry
    policy so a slow or unavailable provider cannot
    block the pipeline indefinitely.
    """

    api_key = os.getenv(
        "OPENROUTER_API_KEY",
        "",
    ).strip()

    model = os.getenv(
        "LLM_MODEL",
        "",
    ).strip()

    base_url = os.getenv(
        "LLM_BASE_URL",
        "https://openrouter.ai/api/v1",
    ).strip()

    timeout_seconds = _get_int_env(
        "LLM_TIMEOUT_SECONDS",
        DEFAULT_TIMEOUT_SECONDS,
    )

    max_retries = _get_int_env(
        "LLM_MAX_RETRIES",
        DEFAULT_MAX_RETRIES,
    )

    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. "
            "Add it to the project .env file."
        )

    if not model:
        raise RuntimeError(
            "LLM_MODEL is missing."
        )

    if timeout_seconds <= 0:
        raise RuntimeError(
            "LLM_TIMEOUT_SECONDS must "
            "be greater than 0."
        )

    if max_retries < 0:
        raise RuntimeError(
            "LLM_MAX_RETRIES must "
            "be 0 or greater."
        )

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )
