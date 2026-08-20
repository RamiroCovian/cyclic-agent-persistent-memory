"""Pruebas de helpers de persistencia (sin LLM)."""

from __future__ import annotations

from pathlib import Path

from agent.config import Settings
from agent.persistence import asegurar_directorio_db, build_run_config


def test_asegurar_directorio_db_crea_padre(tmp_path: Path) -> None:
    """Debe crear el directorio padre del archivo SQLite."""
    db_path = tmp_path / "sub" / "checkpoints.db"
    resolved = asegurar_directorio_db(str(db_path))
    assert resolved.parent.exists()
    assert resolved.name == "checkpoints.db"


def test_build_run_config_incluye_thread_y_limit(
    settings_openai: Settings,
) -> None:
    """El config debe llevar thread_id y recursion_limit."""
    config = build_run_config("sesion-1", settings=settings_openai)
    assert config["configurable"]["thread_id"] == "sesion-1"
    assert config["recursion_limit"] == 8


def test_build_run_config_extra_configurable(settings_gemini: Settings) -> None:
    """Debe fusionar claves extra en configurable."""
    config = build_run_config(
        "sesion-2",
        settings=settings_gemini,
        checkpoint_ns="ns-a",
    )
    assert config["configurable"]["checkpoint_ns"] == "ns-a"
