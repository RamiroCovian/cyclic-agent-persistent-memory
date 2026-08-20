"""Definición del StateGraph ReAct: modelo ↔ herramientas.

Usa ``MessagesState`` como base del estado, un nodo de modelo con
``bind_tools``, un ``ToolNode`` y la arista condicional ``tools_condition``.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.config import Settings, load_settings
from agent.llm import create_llm
from agent.tools import obtener_herramientas

SYSTEM_PROMPT = (
    "Sos un asistente de consultas de pedidos. "
    "Usá las herramientas disponibles para obtener datos reales antes de responder. "
    "Si falta información (por ejemplo el cliente_id), pedí aclaración. "
    "Si una herramienta falla o devuelve error, reintentá con otros parámetros "
    "o pedí más detalles al usuario. Respondé siempre en español."
)


class AgentState(MessagesState):
    """Estado del agente basado en la lista acumulada de mensajes.

    Hereda de ``MessagesState`` (reducer ``add_messages``) para mantener
    el historial de la conversación entre nodos del grafo.
    """


def _compile_graph(llm_with_tools: Any, tools: list[Any]) -> CompiledStateGraph:
    """Ensambla nodos, aristas y compila el StateGraph.

    Args:
        llm_with_tools: Modelo ya vinculado con ``bind_tools``.
        tools: Lista de herramientas para el ``ToolNode``.

    Returns:
        Grafo compilado con ciclo modelo ↔ herramientas.
    """

    async def call_model(state: AgentState) -> dict[str, Any]:
        """Nodo del modelo: decide responder o invocar herramientas.

        Args:
            state: Estado actual con el historial de mensajes.

        Returns:
            Diccionario con el mensaje de respuesta del LLM (posible tool call).
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode(tools))

    graph.add_edge(START, "model")
    graph.add_conditional_edges(
        "model",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "model")

    return graph.compile()


def build_graph(settings: Settings | None = None) -> CompiledStateGraph:
    """Construye y compila el grafo modelo ↔ herramientas.

    Args:
        settings: Configuración del LLM. Si es ``None``, se carga desde el entorno.

    Returns:
        Grafo compilado listo para ``invoke`` / ``ainvoke``.
    """
    resolved = settings or load_settings()
    tools = obtener_herramientas()
    llm_with_tools = create_llm(resolved).bind_tools(tools)
    return _compile_graph(llm_with_tools, tools)


def build_graph_topology() -> CompiledStateGraph:
    """Compila el grafo con un LLM stub solo para inspeccionar la topología.

    Returns:
        Grafo compilado sin requerir API keys reales.
    """
    from langchain_core.messages import AIMessage
    from langchain_core.runnables import RunnableLambda

    async def _stub_response(_messages: list[Any]) -> AIMessage:
        """Respuesta falsa usada únicamente para compilar la topología.

        Args:
            _messages: Historial recibido (ignorado en el stub).

        Returns:
            ``AIMessage`` vacío sin tool calls.
        """
        return AIMessage(content="ok")

    tools = obtener_herramientas()
    stub_llm = RunnableLambda(_stub_response)
    return _compile_graph(stub_llm, tools)


def describe_graph(compiled: CompiledStateGraph | None = None) -> str:
    """Devuelve un resumen textual de nodos y aristas del grafo.

    Args:
        compiled: Grafo ya compilado. Si es ``None``, se usa topología sin API key.

    Returns:
        Descripción legible de la topología (útil para verificación local).
    """
    app = compiled or build_graph_topology()
    drawable = app.get_graph()
    nodes = ", ".join(sorted(drawable.nodes.keys()))
    edges = [f"{edge.source} -> {edge.target}" for edge in drawable.edges]
    edges_text = "\n".join(f"  - {item}" for item in edges)
    return (
        "StateGraph (AgentState <- MessagesState)\n"
        f"Nodos: {nodes}\n"
        f"Aristas:\n{edges_text}\n"
        "Condicional: model --tools_condition--> tools | END"
    )
