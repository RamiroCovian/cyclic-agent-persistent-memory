"""Fábrica del LLM según el proveedor configurado."""

from __future__ import annotations

from typing import Any

from agent.config import Settings


def create_llm(settings: Settings) -> Any:
    """Crea el chat model correspondiente al proveedor de ``settings``.

    Args:
        settings: Configuración con proveedor, modelo y API keys.

    Returns:
        Instancia de chat model compatible con ``bind_tools()``.

    Raises:
        ValueError: Si el proveedor no está soportado.
        ImportError: Si falta el paquete del proveedor elegido.
    """
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
        )

    raise ValueError(f"Proveedor no soportado: {settings.llm_provider!r}")
