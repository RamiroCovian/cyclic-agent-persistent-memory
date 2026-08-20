"""Punto de entrada del agente.

Muestra la topología del StateGraph, el vínculo LLM + ``bind_tools`` y,
si hay API key, ejecuta una consulta de ejemplo.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage

from agent.config import load_settings
from agent.graph import build_graph, describe_graph
from agent.llm import describir_vinculo_tools


def mostrar_topologia() -> None:
    """Imprime la estructura del grafo sin invocar al LLM."""
    print(describe_graph())


def mostrar_vinculo_llm() -> None:
    """Imprime proveedor/modelo y tools vinculadas con ``bind_tools``."""
    print("\n--- LLM + bind_tools ---")
    print(describir_vinculo_tools())


async def ejecutar_consulta(pregunta: str) -> None:
    """Ejecuta una consulta de prueba a través del grafo asíncrono.

    Args:
        pregunta: Texto del usuario a enviar como ``HumanMessage``.
    """
    settings = load_settings()
    app = build_graph(settings)
    result = await app.ainvoke(
        {"messages": [HumanMessage(content=pregunta)]},
        config={"recursion_limit": settings.recursion_limit},
    )
    for message in result["messages"]:
        message.pretty_print()


def main() -> None:
    """Muestra topología y vínculo LLM; si hay API key, corre una consulta."""
    mostrar_topologia()

    try:
        load_settings()
    except ValueError as error:
        print(f"\nOmitiendo vínculo/invocación LLM: {error}")
        return

    mostrar_vinculo_llm()

    pregunta = (
        "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?"
    )
    print(f"\nEjecutando consulta: {pregunta}")
    asyncio.run(ejecutar_consulta(pregunta))


if __name__ == "__main__":
    main()
