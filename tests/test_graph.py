"""Pruebas del StateGraph (topología y tools_condition) sin API keys."""

from __future__ import annotations

from agent.graph import AgentState, build_graph_topology, describe_graph
from langgraph.graph import MessagesState


def test_agent_state_es_messages_state() -> None:
    """AgentState debe basarse en MessagesState (contrato de la consigna)."""
    assert issubclass(AgentState, dict)
    assert "messages" in AgentState.__annotations__
    assert set(MessagesState.__annotations__).issubset(
        set(AgentState.__annotations__)
    )


def test_build_graph_topology_nodos_esperados() -> None:
    """La topología debe incluir nodos model y tools."""
    app = build_graph_topology()
    nodos = set(app.get_graph().nodes.keys())
    assert "model" in nodos
    assert "tools" in nodos
    assert "__start__" in nodos


def test_describe_graph_menciona_tools_condition() -> None:
    """La descripción textual debe documentar tools_condition y persistencia."""
    texto = describe_graph()
    assert "tools_condition" in texto
    assert "MessagesState" in texto
    assert "thread_id" in texto
