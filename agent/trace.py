"""Serialización de trazas ReAct a JSON y log legible."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _contenido_texto(content: Any) -> str:
    """Normaliza el content de un mensaje a texto plano.

    Args:
        content: Content del mensaje (str, list de bloques, u otro).

    Returns:
        Representación textual usable en la traza.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes: list[str] = []
        for bloque in content:
            if isinstance(bloque, dict) and bloque.get("type") == "text":
                partes.append(str(bloque.get("text", "")))
            else:
                partes.append(str(bloque))
        return " ".join(parte for parte in partes if parte)
    return str(content)


def serializar_mensaje(message: Any) -> dict[str, Any]:
    """Convierte un mensaje LangChain a un dict JSON-serializable.

    Args:
        message: Mensaje del historial del grafo.

    Returns:
        Diccionario con tipo, contenido y tool_calls si aplica.
    """
    tipo = getattr(message, "type", message.__class__.__name__)
    entrada: dict[str, Any] = {
        "type": tipo,
        "content": _contenido_texto(getattr(message, "content", "")),
    }

    name = getattr(message, "name", None)
    if name:
        entrada["name"] = name

    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        entrada["tool_calls"] = [
            {
                "name": call.get("name"),
                "args": call.get("args", {}),
                "id": call.get("id"),
            }
            for call in tool_calls
        ]

    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        entrada["tool_call_id"] = tool_call_id

    return entrada


def contar_tool_calls(mensajes: list[Any]) -> int:
    """Cuenta cuántas invocaciones a herramientas hay en el historial.

    Args:
        mensajes: Lista de mensajes del estado del grafo.

    Returns:
        Cantidad total de ``tool_calls`` emitidos por el modelo.
    """
    total = 0
    for message in mensajes:
        tool_calls = getattr(message, "tool_calls", None) or []
        total += len(tool_calls)
    return total


def construir_traza(
    *,
    mensajes: list[Any],
    thread_id: str,
    recursion_limit: int,
    pregunta: str,
) -> dict[str, Any]:
    """Arma el payload JSON de una ejecución ReAct.

    Args:
        mensajes: Historial completo al final de la corrida.
        thread_id: ID de sesión usado en el checkpointer.
        recursion_limit: Tope de pasos configurado en la invocación.
        pregunta: Prompt original del usuario.

    Returns:
        Diccionario listo para guardar como ``.json``.
    """
    tool_calls = contar_tool_calls(mensajes)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "recursion_limit": recursion_limit,
        "pregunta": pregunta,
        "tool_calls_count": tool_calls,
        "multipaso": tool_calls >= 2,
        "messages": [serializar_mensaje(message) for message in mensajes],
    }


def traza_a_log(traza: dict[str, Any]) -> str:
    """Renderiza la traza en formato log legible estilo consigna.

    Args:
        traza: Payload generado por ``construir_traza``.

    Returns:
        Texto multilínea listo para ``.log``.
    """
    lineas = [
        f"thread_id: {traza['thread_id']}",
        f"recursion_limit: {traza['recursion_limit']}",
        f"tool_calls_count: {traza['tool_calls_count']}",
        f"multipaso: {traza['multipaso']}",
        "",
        f'Usuario: "{traza["pregunta"]}"',
    ]

    for message in traza["messages"]:
        tipo = message["type"]
        if tipo == "human":
            continue
        if tipo == "ai" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                args = json.dumps(call.get("args", {}), ensure_ascii=False)
                lineas.append(
                    f"-> El agente decide usar la herramienta: "
                    f"{call.get('name')}({args})"
                )
        elif tipo == "tool":
            lineas.append(
                f"-> La herramienta {message.get('name')} devuelve: "
                f"{message.get('content')}"
            )
        elif tipo == "ai" and message.get("content"):
            lineas.append(f'Respuesta: "{message["content"]}"')

    return "\n".join(lineas) + "\n"


def guardar_traza(
    traza: dict[str, Any],
    directorio: str | Path = "traces",
    prefijo: str = "react-multipaso",
) -> tuple[Path, Path]:
    """Persiste la traza como ``.json`` y ``.log`` en el directorio indicado.

    Args:
        traza: Payload de la ejecución.
        directorio: Carpeta destino (se crea si no existe).
        prefijo: Prefijo del nombre de archivo.

    Returns:
        Tupla ``(ruta_json, ruta_log)``.
    """
    carpeta = Path(directorio)
    carpeta.mkdir(parents=True, exist_ok=True)

    ruta_json = carpeta / f"{prefijo}.json"
    ruta_log = carpeta / f"{prefijo}.log"

    ruta_json.write_text(
        json.dumps(traza, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ruta_log.write_text(traza_a_log(traza), encoding="utf-8")
    return ruta_json, ruta_log
