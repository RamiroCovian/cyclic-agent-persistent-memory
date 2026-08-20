"""Pruebas de la fábrica LLM y bind_tools (sin llamar a proveedores reales)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from agent.config import Settings
from agent.llm import bind_tools_to_llm, create_llm, create_llm_with_tools
from agent.tools import obtener_herramientas


def test_create_llm_openai(settings_openai: Settings) -> None:
    """create_llm debe instanciar ChatOpenAI para provider openai."""
    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        mock_cls.return_value = MagicMock(name="openai-llm")
        llm = create_llm(settings_openai)
        mock_cls.assert_called_once()
        assert llm is mock_cls.return_value


def test_create_llm_gemini(settings_gemini: Settings) -> None:
    """create_llm debe instanciar ChatGoogleGenerativeAI para gemini."""
    with patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_cls:
        mock_cls.return_value = MagicMock(name="gemini-llm")
        llm = create_llm(settings_gemini)
        mock_cls.assert_called_once()
        assert llm is mock_cls.return_value


def test_bind_tools_to_llm_invoca_bind_tools() -> None:
    """bind_tools_to_llm debe delegar en llm.bind_tools(tools)."""
    llm = MagicMock()
    tools = obtener_herramientas()
    bind_tools_to_llm(llm, tools)
    llm.bind_tools.assert_called_once_with(tools)


def test_create_llm_with_tools_encadena_create_y_bind(
    settings_openai: Settings,
) -> None:
    """create_llm_with_tools debe crear el LLM y vincular las tools."""
    fake_llm = MagicMock()
    fake_bound = MagicMock(name="bound")
    fake_llm.bind_tools.return_value = fake_bound

    with patch("agent.llm.create_llm", return_value=fake_llm) as mock_create:
        resultado = create_llm_with_tools(settings_openai)
        mock_create.assert_called_once_with(settings_openai)
        fake_llm.bind_tools.assert_called_once()
        assert resultado is fake_bound
