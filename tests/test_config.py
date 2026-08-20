"""Pruebas de carga y validación de Settings."""

from __future__ import annotations

import pytest

from agent.config import load_settings


def test_load_settings_gemini_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Con Gemini y GOOGLE_API_KEY debe construir Settings válidos."""
    monkeypatch.setattr("agent.config.load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setenv("RECURSION_LIMIT", "12")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    settings = load_settings()
    assert settings.llm_provider == "gemini"
    assert settings.llm_model == "gemini-2.0-flash"
    assert settings.google_api_key == "fake-key"
    assert settings.recursion_limit == 12


def test_load_settings_openai_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Con OpenAI y OPENAI_API_KEY debe construir Settings válidos."""
    monkeypatch.setattr("agent.config.load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    settings = load_settings()
    assert settings.llm_provider == "openai"
    assert settings.openai_api_key == "sk-test"


def test_load_settings_proveedor_invalido(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un proveedor desconocido debe lanzar ValueError."""
    monkeypatch.setattr("agent.config.load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "cohere")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")

    with pytest.raises(ValueError, match="LLM_PROVIDER inválido"):
        load_settings()


def test_load_settings_falta_api_key_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemini sin GOOGLE_API_KEY debe fallar."""
    monkeypatch.setattr("agent.config.load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        load_settings()
