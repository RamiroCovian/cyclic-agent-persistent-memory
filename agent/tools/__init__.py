"""Herramientas LangChain (@tool) para consulta de pedidos.

El LLM elige qué herramienta invocar basándose en los docstrings.
Las descripciones deben ser explícitas sobre cuándo usar cada una.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.tools import tool

from agent.tools.db import (
    buscar_pedido_por_id,
    listar_pedidos_cliente,
    obtener_ultimo_pedido,
    resumir_pedidos,
)


def _serializar(resultado: dict[str, Any]) -> str:
    """Convierte un diccionario de resultado a JSON legible para el LLM.

    Args:
        resultado: Payload de respuesta de la herramienta.

    Returns:
        Cadena JSON con ``ensure_ascii=False`` para preservar acentos.
    """
    return json.dumps(resultado, ensure_ascii=False)


@tool
def buscar_pedidos(cliente_id: int) -> str:
    """Consulta en la base de pedidos cuántos pedidos tiene un cliente y el monto total.

    Usá esta herramienta cuando el usuario pregunte por la cantidad de pedidos,
    el total gastado, el resumen de compras o el historial agregado de un cliente
    identificado por su ``cliente_id`` numérico (por ejemplo 102).

    No sirve para obtener el detalle de un pedido puntual ni el último pedido;
    para eso usá ``obtener_detalle_ultimo_pedido`` o ``buscar_pedido``.

    Args:
        cliente_id: ID numérico del cliente a consultar (ej. 101, 102, 103).

    Returns:
        JSON con ``cliente_id``, ``pedidos`` (cantidad) y ``total`` (suma de montos).
        Si el cliente no existe, ``pedidos`` será 0 y ``total`` 0.
    """
    pedidos = listar_pedidos_cliente(cliente_id)
    resumen = resumir_pedidos(pedidos)
    return _serializar(
        {
            "cliente_id": cliente_id,
            "pedidos": resumen["pedidos"],
            "total": resumen["total"],
        }
    )


@tool
def obtener_detalle_ultimo_pedido(cliente_id: int) -> str:
    """Obtiene el detalle del pedido más reciente de un cliente.

    Usá esta herramienta cuando el usuario pida el último pedido, el pedido más
    reciente, o el detalle del pedido actual/en curso de un cliente. Requiere
    el ``cliente_id`` numérico.

    Args:
        cliente_id: ID numérico del cliente (ej. 102).

    Returns:
        JSON con el pedido más reciente (``pedido_id``, ``fecha``, ``monto``,
        ``estado``) o un mensaje de error si el cliente no tiene pedidos.
    """
    pedidos = listar_pedidos_cliente(cliente_id)
    ultimo = obtener_ultimo_pedido(pedidos)
    if ultimo is None:
        return _serializar(
            {
                "error": "cliente_sin_pedidos",
                "cliente_id": cliente_id,
                "mensaje": f"No se encontraron pedidos para el cliente {cliente_id}.",
            }
        )
    return _serializar({"cliente_id": cliente_id, "ultimo_pedido": ultimo})


@tool
def buscar_pedido(cliente_id: int, pedido_id: int) -> str:
    """Busca un pedido específico de un cliente por su ``pedido_id``.

    Usá esta herramienta cuando el usuario indique un número de pedido concreto
    y quiera conocer su fecha, monto o estado. Necesita ambos identificadores.

    Args:
        cliente_id: ID numérico del cliente dueño del pedido.
        pedido_id: ID numérico del pedido a recuperar (ej. 6003).

    Returns:
        JSON con el pedido encontrado, o un error si no existe esa combinación.
    """
    pedido = buscar_pedido_por_id(cliente_id, pedido_id)
    if pedido is None:
        return _serializar(
            {
                "error": "pedido_no_encontrado",
                "cliente_id": cliente_id,
                "pedido_id": pedido_id,
                "mensaje": (
                    f"No existe el pedido {pedido_id} para el cliente {cliente_id}."
                ),
            }
        )
    return _serializar({"cliente_id": cliente_id, "pedido": pedido})


def obtener_herramientas() -> list[Any]:
    """Devuelve la lista de herramientas registradas para el agente.

    Returns:
        Lista de instancias ``BaseTool`` listas para ``llm.bind_tools()``.
    """
    return [buscar_pedidos, obtener_detalle_ultimo_pedido, buscar_pedido]
