"""Punto de entrada del agente.

Incluye demo multi-paso (≥2 tool calls), ``recursion_limit``, persistencia
por ``thread_id`` y exportación de traza ReAct a ``traces/``.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from langchain_core.messages import HumanMessage

from agent.config import load_settings
from agent.graph import build_graph, describe_graph
from agent.llm import describir_vinculo_tools
from agent.persistence import build_run_config, open_async_checkpointer
from agent.trace import construir_traza, contar_tool_calls, guardar_traza


PREGUNTA_MULTIPASO = (
    "Del cliente 102, ¿cuántos pedidos tuvo, cuál fue el total "
    "y cuál es el detalle del último pedido?"
)


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


async def demostrar_multipaso(
    thread_id: str | None = None,
    pregunta: str = PREGUNTA_MULTIPASO,
) -> dict[str, Any]:
    """Ejecuta una prueba que requiere ≥2 tool calls y guarda la traza.

    Args:
        thread_id: Sesión a usar. Si es ``None``, se genera un ID único.
        pregunta: Prompt diseñado para forzar razonamiento multi-paso.

    Returns:
        Payload de traza generado tras la corrida.

    Raises:
        RuntimeError: Si el agente no invoca al menos dos herramientas.
    """
    settings = load_settings()
    session_id = thread_id or f"multipaso-{uuid.uuid4().hex[:8]}"

    print("\n--- Demo multi-paso (>=2 tool calls) ---")
    print(f"thread_id: {session_id}")
    print(f"recursion_limit: {settings.recursion_limit}")
    print(f"pregunta: {pregunta}")

    async with open_async_checkpointer(settings.sqlite_db_path) as checkpointer:
        app = build_graph(settings, checkpointer=checkpointer)
        mensajes = await ejecutar_consulta(pregunta, thread_id=session_id, app=app)

    for message in mensajes:
        message.pretty_print()

    cantidad = contar_tool_calls(mensajes)
    print(f"\nTool calls detectados: {cantidad}")
    if cantidad < 2:
        raise RuntimeError(
            f"Se esperaban >=2 tool calls y hubo {cantidad}. "
            "Revisá el prompt o los docstrings de las tools."
        )

    traza = construir_traza(
        mensajes=mensajes,
        thread_id=session_id,
        recursion_limit=settings.recursion_limit,
        pregunta=pregunta,
    )
    ruta_json, ruta_log = guardar_traza(traza)
    print(f"Traza JSON: {ruta_json}")
    print(f"Traza LOG:  {ruta_log}")
    return traza


async def demostrar_memoria_sesion(thread_id: str | None = None) -> None:
    """Corre dos turnos con el mismo ``thread_id`` para probar persistencia.

    Args:
        thread_id: Identificador de la sesión de demo.
    """
    settings = load_settings()
    session_id = thread_id or f"sesion-{uuid.uuid4().hex[:8]}"
    print(f"\n--- Persistencia SQLite + thread_id={session_id!r} ---")
    print(f"DB: {settings.sqlite_db_path}")
    print(f"recursion_limit: {settings.recursion_limit}")

    async with open_async_checkpointer(settings.sqlite_db_path) as checkpointer:
        app = build_graph(settings, checkpointer=checkpointer)

        primera = "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?"
        print(f"\n[Turno 1] {primera}")
        mensajes_1 = await ejecutar_consulta(primera, thread_id=session_id, app=app)
        for message in mensajes_1:
            message.pretty_print()

        segunda = "¿y el último?"
        print(f"\n[Turno 2] {segunda}")
        mensajes_2 = await ejecutar_consulta(segunda, thread_id=session_id, app=app)
        nuevos = mensajes_2[len(mensajes_1) :]
        for message in nuevos:
            message.pretty_print()


def main() -> None:
    """Muestra topología, vínculo LLM y demo multi-paso con traza."""
    mostrar_topologia()

    try:
        load_settings()
    except ValueError as error:
        print(f"\nOmitiendo vínculo/invocación LLM: {error}")
        print("Podés revisar traces/react-multipaso.json si ya fue generado.")
        return

    mostrar_vinculo_llm()
    asyncio.run(demostrar_multipaso())


if __name__ == "__main__":
    main()
