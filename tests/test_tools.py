"""Pruebas de las herramientas LangChain (@tool)."""

from __future__ import annotations

import json

from agent.tools import (
    buscar_pedido,
    buscar_pedidos,
    obtener_detalle_ultimo_pedido,
    obtener_herramientas,
)


def test_obtener_herramientas_registra_tres() -> None:
    """El agente expone exactamente tres tools con nombre."""
    tools = obtener_herramientas()
    assert len(tools) == 3
    nombres = {tool.name for tool in tools}
    assert nombres == {
        "buscar_pedidos",
        "obtener_detalle_ultimo_pedido",
        "buscar_pedido",
    }


def test_buscar_pedidos_cliente_102() -> None:
    """buscar_pedidos(102) debe devolver el resumen de la consigna."""
    raw = buscar_pedidos.invoke({"cliente_id": 102})
    data = json.loads(raw)
    assert data["cliente_id"] == 102
    assert data["pedidos"] == 3
    assert data["total"] == 14500.0


def test_buscar_pedidos_cliente_sin_datos() -> None:
    """Un cliente inexistente debe devolver ceros."""
    data = json.loads(buscar_pedidos.invoke({"cliente_id": 999}))
    assert data == {"cliente_id": 999, "pedidos": 0, "total": 0}


def test_obtener_detalle_ultimo_pedido() -> None:
    """Debe devolver el pedido más reciente del cliente 102."""
    data = json.loads(obtener_detalle_ultimo_pedido.invoke({"cliente_id": 102}))
    assert data["cliente_id"] == 102
    assert data["ultimo_pedido"]["pedido_id"] == 6003


def test_obtener_detalle_ultimo_pedido_sin_datos() -> None:
    """Sin pedidos debe responder con error tipado."""
    data = json.loads(obtener_detalle_ultimo_pedido.invoke({"cliente_id": 999}))
    assert data["error"] == "cliente_sin_pedidos"


def test_buscar_pedido_encontrado() -> None:
    """buscar_pedido debe devolver el detalle del pedido pedido."""
    data = json.loads(
        buscar_pedido.invoke({"cliente_id": 102, "pedido_id": 6001})
    )
    assert data["pedido"]["pedido_id"] == 6001
    assert data["pedido"]["estado"] == "entregado"


def test_buscar_pedido_no_encontrado() -> None:
    """Si no existe el pedido, debe devolver error."""
    data = json.loads(
        buscar_pedido.invoke({"cliente_id": 102, "pedido_id": 9999})
    )
    assert data["error"] == "pedido_no_encontrado"


def test_herramientas_tienen_docstring_descriptivo() -> None:
    """Los docstrings deben existir: el LLM decide con esa descripción."""
    for tool in obtener_herramientas():
        assert tool.description
        assert len(tool.description) > 40
