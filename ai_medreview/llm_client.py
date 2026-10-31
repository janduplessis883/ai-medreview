"""
LLM client for external self-hosted LLM API.
Supports OpenAI-compatible chat completions endpoint.

This module provides a unified interface for making LLM requests.
It can use either a self-hosted external LLM or fall back to Groq.
"""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


def get_llm_config() -> dict[str, Any]:
    """Get LLM configuration from secrets or environment variables."""
    return {
        "api_key": st.secrets.get("LLM_API_KEY", os.environ.get("LLM_API_KEY", "")),
        "base_url": st.secrets.get(
            "LLM_BASE_URL", os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        ),
        "model": st.secrets.get("LLM_MODEL", os.environ.get("LLM_MODEL", "auto")),
    }


def get_groq_config() -> dict[str, Any]:
    """Get Groq configuration from secrets or environment variables."""
    return {
        "api_key": st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", "")),
        "model": st.secrets.get("GROQ_MODEL", os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")),
    }


def call_llm(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    use_groq: bool = False,
) -> str:
    """
    Call an LLM chat completions endpoint.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        model: Model name (uses configured default if None)
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens in response
        use_groq: If True, use Groq instead of the external LLM

    Returns:
        The assistant's response content as a string

    Raises:
        requests.exceptions.HTTPError: If the API returns an error status
        ValueError: If API key is not configured
    """
    if use_groq:
        config = get_groq_config()
        api_key = config["api_key"]
        model = model or config["model"]
        base_url = "https://api.groq.com/openai/v1"
    else:
        config = get_llm_config()
        api_key = config["api_key"]
        base_url = config["base_url"]
        model = model or config["model"]

    if not api_key:
        provider = "GROQ_API_KEY" if use_groq else "LLM_API_KEY"
        raise ValueError(
            f"{provider} not configured. Set it in .streamlit/secrets.toml or environment."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    response = requests.post(endpoint, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def ask_llm(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.7,
    system_prompt: str | None = None,
    use_groq: bool = False,
) -> str:
    """
    Simple interface to ask the LLM a question.

    Args:
        prompt: User prompt/question
        model: Model name (uses configured default if None)
        temperature: Sampling temperature
        system_prompt: Optional system message
        use_groq: If True, use Groq instead of the external LLM

    Returns:
        The assistant's response as a string
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return call_llm(messages=messages, model=model, temperature=temperature, use_groq=use_groq)


# Backward-compatible alias for existing code that uses ask_groq
def ask_groq(prompt: str, model: str = "openai/gpt-oss-120b") -> str:
    """
    Backward-compatible function for Groq API calls.
    This wraps ask_llm with use_groq=True.
    """
    return ask_llm(prompt=prompt, model=model, use_groq=True)
