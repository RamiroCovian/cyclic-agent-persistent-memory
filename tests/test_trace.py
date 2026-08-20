"""Pruebas de serialización y exportación de trazas ReAct."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agent.trace import (
    construir_traza,
    contar_tool_calls,
    guardar_traza,
    serializar_mensaje,
    traza_a_log,
)


def test_contar_tool_calls_multipaso() -> None:
    """Debe sumar todos los tool_calls del historial."""
    mensajes = [
        SimpleNamespace(tool_calls=[{"name": "a"}, {"name": "b"}]),
        SimpleNamespace(tool_calls=[]),
        SimpleNamespace(),
    ]
    assert contar_tool_calls(mensajes) == 2


def test_serializar_mensaje_con_tool_calls() -> None:
    """serializar_mensaje debe incluir name/args/id de cada call."""
    message = SimpleNamespace(
        type="ai",
        content="",
        tool_calls=[
            {"name": "buscar_pedidos", "args": {"cliente_id": 102}, "id": "c1"}
        ],
    )
    data = serializar_mensaje(message)
    assert data["type"] == "ai"
    assert data["tool_calls"][0]["name"] == "buscar_pedidos"
    assert data["tool_calls"][0]["args"]["cliente_id"] == 102


def test_construir_traza_marca_multipaso() -> None:
    """Con >=2 tool calls, multipaso debe ser True."""
    mensajes = [
        SimpleNamespace(
            type="ai",
            content="",
            tool_calls=[
                {"name": "buscar_pedidos", "args": {"cliente_id": 102}, "id": "1"},
                {
                    "name": "obtener_detalle_ultimo_pedido",
                    "args": {"cliente_id": 102},
                    "id": "2",
                },
            ],
        )
    ]
    traza = construir_traza(
        mensajes=mensajes,
        thread_id="t-1",
        recursion_limit=10,
        pregunta="demo",
    )
    assert traza["tool_calls_count"] == 2
    assert traza["multipaso"] is True
    assert traza["recursion_limit"] == 10


def test_traza_a_log_incluye_pasos_react() -> None:
    """El log debe mencionar herramientas y respuesta."""
    traza = {
        "thread_id": "t-1",
        "recursion_limit": 10,
        "tool_calls_count": 1,
        "multipaso": False,
        "pregunta": "¿cuántos pedidos?",
        "messages": [
            {"type": "human", "content": "¿cuántos pedidos?"},
            {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "name": "buscar_pedidos",
                        "args": {"cliente_id": 102},
                        "id": "1",
                    }
                ],
            },
            {
                "type": "tool",
                "name": "buscar_pedidos",
                "content": '{"pedidos": 3}',
            },
            {"type": "ai", "content": "El cliente tuvo 3 pedidos."},
        ],
    }
    log = traza_a_log(traza)
    assert "buscar_pedidos" in log
    assert "El cliente tuvo 3 pedidos." in log
    assert "recursion_limit: 10" in log


def test_guardar_traza_crea_json_y_log(tmp_path: Path) -> None:
    """guardar_traza debe escribir ambos archivos en el directorio."""
    traza = construir_traza(
        mensajes=[],
        thread_id="t-tmp",
        recursion_limit=5,
        pregunta="hola",
    )
    ruta_json, ruta_log = guardar_traza(traza, directorio=tmp_path, prefijo="test")
    assert ruta_json.exists()
    assert ruta_log.exists()
    payload = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert payload["thread_id"] == "t-tmp"
