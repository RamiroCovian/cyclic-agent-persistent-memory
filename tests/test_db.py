"""Pruebas de la capa de datos simulada de pedidos."""

from __future__ import annotations

from agent.tools.db import (
    buscar_pedido_por_id,
    listar_pedidos_cliente,
    obtener_ultimo_pedido,
    resumir_pedidos,
)


def test_listar_pedidos_cliente_102_tiene_tres() -> None:
    """El cliente 102 de la consigna debe tener 3 pedidos."""
    pedidos = listar_pedidos_cliente(102)
    assert len(pedidos) == 3


def test_listar_pedidos_cliente_inexistente() -> None:
    """Un cliente desconocido debe devolver lista vacía."""
    assert listar_pedidos_cliente(999) == []


def test_resumir_pedidos_cliente_102() -> None:
    """El resumen del cliente 102 coincide con la consigna (3 / 14500)."""
    resumen = resumir_pedidos(listar_pedidos_cliente(102))
    assert resumen == {"pedidos": 3, "total": 14500.0}


def test_resumir_pedidos_lista_vacia() -> None:
    """Sin pedidos el total y la cantidad deben ser cero."""
    assert resumir_pedidos([]) == {"pedidos": 0, "total": 0}


def test_obtener_ultimo_pedido_por_fecha() -> None:
    """El último pedido del cliente 102 es el 6003."""
    ultimo = obtener_ultimo_pedido(listar_pedidos_cliente(102))
    assert ultimo is not None
    assert ultimo["pedido_id"] == 6003


def test_obtener_ultimo_pedido_vacio() -> None:
    """Sin pedidos, el helper debe devolver None."""
    assert obtener_ultimo_pedido([]) is None


def test_buscar_pedido_por_id_encontrado() -> None:
    """Debe encontrar un pedido existente por IDs."""
    pedido = buscar_pedido_por_id(102, 6001)
    assert pedido is not None
    assert pedido["monto"] == 4500.0


def test_buscar_pedido_por_id_no_encontrado() -> None:
    """Debe devolver None si la combinación no existe."""
    assert buscar_pedido_por_id(102, 9999) is None
