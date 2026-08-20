"""Punto de entrada del agente.

Muestra topología, vínculo LLM y una demo de memoria de sesión con
``AsyncSqliteSaver`` + ``thread_id`` (dos turnos en la misma conversación).
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage

from agent.config import load_settings
from agent.graph import build_graph, describe_graph
from agent.llm import describir_vinculo_tools
from agent.persistence import build_run_config, open_async_checkpointer


def mostrar_topologia() -> None:
    """Imprime la estructura del grafo sin invocar al LLM."""
    print(describe_graph())


def mostrar_vinculo_llm() -> None:
    """Imprime proveedor/modelo y tools vinculadas con ``bind_tools``."""
    print("\n--- LLM + bind_tools ---")
    print(describir_vinculo_tools())


async def ejecutar_consulta(
    pregunta: str,
    *,
    thread_id: str,
    app: Any | None = None,
) -> list[Any]:
    """Ejecuta una consulta preservando la sesión vía ``thread_id``.

    Args:
        pregunta: Texto del usuario a enviar como ``HumanMessage``.
        thread_id: ID de sesión; el mismo valor reutiliza el historial persistido.
        app: Grafo ya compilado con checkpointer. Si es ``None``, se abre uno temporal.

    Returns:
        Lista de mensajes del estado final tras la invocación.
    """
    settings = load_settings()
    config = build_run_config(thread_id=thread_id, settings=settings)

    if app is not None:
        result = await app.ainvoke(
            {"messages": [HumanMessage(content=pregunta)]},
            config=config,
        )
        return list(result["messages"])

    async with open_async_checkpointer(settings.sqlite_db_path) as checkpointer:
        compiled = build_graph(settings, checkpointer=checkpointer)
        result = await compiled.ainvoke(
            {"messages": [HumanMessage(content=pregunta)]},
            config=config,
        )
        return list(result["messages"])


async def demostrar_memoria_sesion(thread_id: str = "demo-cliente-102") -> None:
    """Corre dos turnos con el mismo ``thread_id`` para probar persistencia.

    1. Pregunta por cantidad/total del cliente 102.
    2. Pregunta "¿y el último?" — el agente debe recordar el cliente_id.

    Args:
        thread_id: Identificador de la sesión de demo.
    """
    settings = load_settings()
    print(f"\n--- Persistencia SQLite + thread_id={thread_id!r} ---")
    print(f"DB: {settings.sqlite_db_path}")

    async with open_async_checkpointer(settings.sqlite_db_path) as checkpointer:
        app = build_graph(settings, checkpointer=checkpointer)

        primera = "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?"
        print(f"\n[Turno 1] {primera}")
        mensajes_1 = await ejecutar_consulta(primera, thread_id=thread_id, app=app)
        for message in mensajes_1:
            message.pretty_print()

        segunda = "¿y el último?"
        print(f"\n[Turno 2] {segunda}")
        mensajes_2 = await ejecutar_consulta(segunda, thread_id=thread_id, app=app)
        # Solo mostrar mensajes nuevos del segundo turno
        nuevos = mensajes_2[len(mensajes_1) :]
        for message in nuevos:
            message.pretty_print()


def main() -> None:
    """Muestra topología/vínculo LLM y demo de memoria de sesión."""
    mostrar_topologia()

    try:
        load_settings()
    except ValueError as error:
        print(f"\nOmitiendo vínculo/invocación LLM: {error}")
        return

    mostrar_vinculo_llm()
    asyncio.run(demostrar_memoria_sesion())


if __name__ == "__main__":
    main()
