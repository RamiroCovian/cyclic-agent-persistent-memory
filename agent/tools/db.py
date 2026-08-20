"""Datos simulados de pedidos para las herramientas del agente.

Simula una base de datos en memoria con clientes y pedidos,
sin conexiones externas ni API keys.
"""

from __future__ import annotations

from typing import Any


PEDIDOS_DB: dict[int, list[dict[str, Any]]] = {
    101: [
        {"pedido_id": 5001, "fecha": "2026-01-10", "monto": 3200.0, "estado": "entregado"},
        {"pedido_id": 5002, "fecha": "2026-02-03", "monto": 1800.0, "estado": "entregado"},
    ],
    102: [
        {"pedido_id": 6001, "fecha": "2026-01-15", "monto": 4500.0, "estado": "entregado"},
        {"pedido_id": 6002, "fecha": "2026-02-20", "monto": 5200.0, "estado": "entregado"},
        {"pedido_id": 6003, "fecha": "2026-03-08", "monto": 4800.0, "estado": "en_camino"},
    ],
    103: [
        {"pedido_id": 7001, "fecha": "2026-03-01", "monto": 990.0, "estado": "cancelado"},
    ],
}


def listar_pedidos_cliente(cliente_id: int) -> list[dict[str, Any]]:
    """Devuelve la lista de pedidos asociados a un cliente.

    Args:
        cliente_id: Identificador numérico del cliente.

    Returns:
        Lista de pedidos del cliente. Si no existe, lista vacía.
    """
    return list(PEDIDOS_DB.get(cliente_id, []))


def resumir_pedidos(pedidos: list[dict[str, Any]]) -> dict[str, Any]:
    """Calcula cantidad de pedidos y monto total a partir de una lista.

    Args:
        pedidos: Colección de pedidos con campo ``monto``.

    Returns:
        Diccionario con ``pedidos`` (cantidad) y ``total`` (suma de montos).
    """
    total = sum(float(pedido.get("monto", 0)) for pedido in pedidos)
    return {"pedidos": len(pedidos), "total": total}


def obtener_ultimo_pedido(pedidos: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Obtiene el pedido más reciente ordenando por fecha descendente.

    Args:
        pedidos: Colección de pedidos con campo ``fecha`` (YYYY-MM-DD).

    Returns:
        El pedido más reciente, o ``None`` si la lista está vacía.
    """
    if not pedidos:
        return None
    return max(pedidos, key=lambda pedido: str(pedido.get("fecha", "")))


def buscar_pedido_por_id(cliente_id: int, pedido_id: int) -> dict[str, Any] | None:
    """Busca un pedido concreto de un cliente por su identificador.

    Args:
        cliente_id: Identificador numérico del cliente.
        pedido_id: Identificador numérico del pedido.

    Returns:
        El pedido encontrado, o ``None`` si no existe.
    """
    for pedido in listar_pedidos_cliente(cliente_id):
        if int(pedido.get("pedido_id", -1)) == pedido_id:
            return pedido
    return None
