"""Punto de entrada del agente.

En esta fase muestra la topología del StateGraph (modelo, herramientas
y ``tools_condition``). La ejecución con LLM requiere variables de entorno.
"""

from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage

from agent.config import load_settings
from agent.graph import build_graph, describe_graph


def mostrar_topologia() -> None:
    """Imprime la estructura del grafo sin invocar al LLM."""
    print(describe_graph())


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
    """Muestra la topología; si hay API key, corre una consulta de ejemplo."""
    mostrar_topologia()

    try:
        load_settings()
    except ValueError as error:
        print(f"\nOmitiendo invocación LLM: {error}")
        return

    pregunta = (
        "¿Cuántos pedidos tuvo el cliente 102 y cuál fue el total?"
    )
    print(f"\nEjecutando consulta: {pregunta}")
    asyncio.run(ejecutar_consulta(pregunta))


if __name__ == "__main__":
    main()
