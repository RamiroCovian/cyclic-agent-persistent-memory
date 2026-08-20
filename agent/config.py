"""Carga de configuración desde variables de entorno."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    """Parámetros de ejecución del agente leídos desde el entorno.

    Attributes:
        llm_provider: Proveedor del modelo (``gemini``, ``openai`` o ``anthropic``).
        llm_model: Nombre del modelo a invocar.
        google_api_key: API key de Google Gemini (opcional).
        openai_api_key: API key de OpenAI (opcional).
        anthropic_api_key: API key de Anthropic (opcional).
        sqlite_db_path: Ruta del archivo SQLite del checkpointer.
        recursion_limit: Tope de pasos del grafo para evitar bucles infinitos.
    """

    llm_provider: str
    llm_model: str
    google_api_key: str | None
    openai_api_key: str | None
    anthropic_api_key: str | None
    sqlite_db_path: str
    recursion_limit: int


def load_settings() -> Settings:
    """Carga y valida la configuración desde ``.env`` / variables de entorno.

    Returns:
        Instancia inmutable ``Settings`` con los valores resueltos.

    Raises:
        ValueError: Si el proveedor no es soportado o falta la API key requerida.
    """
    load_dotenv()

    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    model = os.getenv("LLM_MODEL", "gemini-2.0-flash").strip()
    google_api_key = os.getenv("GOOGLE_API_KEY") or None
    openai_api_key = os.getenv("OPENAI_API_KEY") or None
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or None
    sqlite_db_path = os.getenv("SQLITE_DB_PATH", "./data/checkpoints.db").strip()
    recursion_limit = int(os.getenv("RECURSION_LIMIT", "10"))

    if provider not in {"gemini", "openai", "anthropic"}:
        raise ValueError(
            f"LLM_PROVIDER inválido: {provider!r}. "
            "Usá gemini | openai | anthropic."
        )

    if provider == "gemini" and not google_api_key:
        raise ValueError("Falta GOOGLE_API_KEY para LLM_PROVIDER=gemini.")
    if provider == "openai" and not openai_api_key:
        raise ValueError("Falta OPENAI_API_KEY para LLM_PROVIDER=openai.")
    if provider == "anthropic" and not anthropic_api_key:
        raise ValueError("Falta ANTHROPIC_API_KEY para LLM_PROVIDER=anthropic.")

    return Settings(
        llm_provider=provider,
        llm_model=model,
        google_api_key=google_api_key,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        sqlite_db_path=sqlite_db_path,
        recursion_limit=recursion_limit,
    )
