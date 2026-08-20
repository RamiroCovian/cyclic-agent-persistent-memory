"""Fábrica del LLM y vínculo con herramientas vía ``bind_tools``.

Cumple el entregable de la consigna: configurar un LLM (OpenAI o Anthropic,
también Gemini) y enlazarlo a las tools con ``llm.bind_tools()`` para que el
modelo decida autónomamente cuándo invocarlas.
"""

from __future__ import annotations

from typing import Any, Sequence

from agent.config import Settings, load_settings
from agent.tools import obtener_herramientas


def create_llm(settings: Settings) -> Any:
    """Crea el chat model correspondiente al proveedor de ``settings``.

    Args:
        settings: Configuración con proveedor, modelo y API keys.

    Returns:
        Instancia de chat model compatible con ``bind_tools()``.

    Raises:
        ValueError: Si el proveedor no está soportado.
        ImportError: Si falta el paquete del proveedor elegido.
    """
    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )

    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
        )

    raise ValueError(f"Proveedor no soportado: {settings.llm_provider!r}")


def bind_tools_to_llm(llm: Any, tools: Sequence[Any] | None = None) -> Any:
    """Vincula herramientas al LLM con ``llm.bind_tools()``.

    El modelo recibe el esquema de cada tool (nombre, docstring, args) y puede
    emitir ``tool_calls`` en su respuesta. No ejecuta las tools: eso lo hace
    el ``ToolNode`` del grafo.

    Args:
        llm: Chat model sin tools (salida de ``create_llm``).
        tools: Herramientas a vincular. Si es ``None``, usa las del agente.

    Returns:
        Runnable del LLM con tools bound (mismo contrato que ``bind_tools``).
    """
    resolved_tools = list(tools) if tools is not None else obtener_herramientas()
    return llm.bind_tools(resolved_tools)


def create_llm_with_tools(
    settings: Settings | None = None,
    tools: Sequence[Any] | None = None,
) -> Any:
    """Crea el LLM del entorno y lo vincula a las herramientas del agente.

    Equivale a: ``create_llm(settings).bind_tools(tools)``.

    Args:
        settings: Configuración. Si es ``None``, se carga desde el entorno.
        tools: Tools a bindear. Si es ``None``, usa ``obtener_herramientas()``.

    Returns:
        LLM listo para el nodo ``model`` del StateGraph.
    """
    resolved = settings or load_settings()
    llm = create_llm(resolved)
    return bind_tools_to_llm(llm, tools)


def describir_vinculo_tools(settings: Settings | None = None) -> str:
    """Resume proveedor, modelo y tools vinculadas (sin invocar la API).

    Args:
        settings: Configuración. Si es ``None``, se carga desde el entorno.

    Returns:
        Texto legible para verificar el contrato ``bind_tools``.
    """
    resolved = settings or load_settings()
    tools = obtener_herramientas()
    llm_with_tools = create_llm_with_tools(resolved, tools)

    nombres = [getattr(tool, "name", str(tool)) for tool in tools]
    kwargs = getattr(llm_with_tools, "kwargs", {}) or {}
    bound = kwargs.get("tools")
    cantidad_bound = len(bound) if bound is not None else "n/d"

    return (
        f"Proveedor: {resolved.llm_provider}\n"
        f"Modelo: {resolved.llm_model}\n"
        f"Tools del agente: {', '.join(nombres)}\n"
        f"Tools en bind_tools: {cantidad_bound}\n"
        "Contrato: llm.bind_tools(tools) -> el modelo puede emitir tool_calls"
    )
