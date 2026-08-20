"""Fixtures compartidas para las pruebas unitarias."""

from __future__ import annotations

import pytest

from agent.config import Settings


@pytest.fixture
def settings_gemini() -> Settings:
    """Settings de prueba con proveedor Gemini (sin llamar a la API)."""
    return Settings(
        llm_provider="gemini",
        llm_model="gemini-2.0-flash",
        google_api_key="test-google-key",
        openai_api_key=None,
        anthropic_api_key=None,
        sqlite_db_path="./data/test-checkpoints.db",
        recursion_limit=10,
    )


@pytest.fixture
def settings_openai() -> Settings:
    """Settings de prueba con proveedor OpenAI."""
    return Settings(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        google_api_key=None,
        openai_api_key="test-openai-key",
        anthropic_api_key=None,
        sqlite_db_path="./data/test-checkpoints.db",
        recursion_limit=8,
    )
