"""Persistencia de estado con SqliteSaver / AsyncSqliteSaver.

El checkpointer guarda el historial del grafo por ``thread_id``, permitiendo
retomar la misma sesión de razonamiento en invocaciones posteriores.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent.config import Settings, load_settings


def asegurar_directorio_db(db_path: str) -> Path:
    """Crea el directorio padre del archivo SQLite si no existe.

    Args:
        db_path: Ruta al archivo ``.db`` del checkpointer.

    Returns:
        ``Path`` absoluto del archivo de base de datos.
    """
    path = Path(db_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def create_sqlite_saver(db_path: str | None = None) -> Any:
    """Abre un context manager de ``SqliteSaver`` síncrono.

    Args:
        db_path: Ruta SQLite. Si es ``None``, usa ``SQLITE_DB_PATH`` del entorno.

    Returns:
        Context manager de ``SqliteSaver`` (usar con ``with``).

    Example:
        ``with create_sqlite_saver() as checkpointer: ...``
    """
    settings = load_settings() if db_path is None else None
    resolved = db_path or (settings.sqlite_db_path if settings else "./data/checkpoints.db")
    path = asegurar_directorio_db(resolved)
    return SqliteSaver.from_conn_string(str(path))


@asynccontextmanager
async def open_async_checkpointer(
    db_path: str | None = None,
) -> AsyncIterator[AsyncSqliteSaver]:
    """Abre un ``AsyncSqliteSaver`` como context manager asíncrono.

    Args:
        db_path: Ruta SQLite. Si es ``None``, usa ``SQLITE_DB_PATH`` del entorno.

    Yields:
        Checkpointer async compatible con ``ainvoke`` / ``astream``.
    """
    settings = load_settings() if db_path is None else None
    resolved = db_path or (settings.sqlite_db_path if settings else "./data/checkpoints.db")
    path = asegurar_directorio_db(resolved)

    async with AsyncSqliteSaver.from_conn_string(str(path)) as checkpointer:
        yield checkpointer


def build_run_config(
    thread_id: str,
    settings: Settings | None = None,
    **extra_configurable: Any,
) -> dict[str, Any]:
    """Arma el ``config`` de invocación con ``thread_id`` y ``recursion_limit``.

    Args:
        thread_id: Identificador de sesión; el mismo valor reanuda el historial.
        settings: Configuración. Si es ``None``, se carga desde el entorno.
        **extra_configurable: Claves adicionales para ``configurable``.

    Returns:
        Diccionario usable en ``invoke`` / ``ainvoke``.
    """
    resolved = settings or load_settings()
    configurable: dict[str, Any] = {"thread_id": thread_id, **extra_configurable}
    return {
        "configurable": configurable,
        "recursion_limit": resolved.recursion_limit,
    }
