"""
base_agent — shared LangChain LLM factory.

Replaces the custom OpenAI-client BaseAgent with LangChain's ChatOpenAI,
configured to point at OpenRouter so the rest of the codebase keeps using
the same models.
"""

from typing import Optional
from langchain_openai import ChatOpenAI
import config


def create_llm(
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> ChatOpenAI:
    """
    Return a ChatOpenAI instance wired to OpenRouter.

    Parameters
    ----------
    model       : override the default TEXT_MODEL from config
    temperature : sampling temperature
    max_tokens  : maximum tokens in the completion
    """
    if not config.OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Copy .env.example to .env and fill in your key."
        )
    return ChatOpenAI(
        model=model or config.TEXT_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )
