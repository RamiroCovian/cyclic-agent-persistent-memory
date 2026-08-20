"""Punto de entrada del agente.

En esta fase solo valida que las herramientas personalizadas carguen
correctamente. El StateGraph y el LLM se agregan en ramas siguientes.
"""

from __future__ import annotations

from agent.tools import obtener_herramientas


def mostrar_herramientas() -> None:
    """Imprime nombre y descripción de cada herramienta registrada.

    Útil para verificar que los docstrings del contrato ``@tool``
    quedaron correctamente expuestos al LLM.
    """
    herramientas = obtener_herramientas()
    print(f"Herramientas registradas: {len(herramientas)}")
    for herramienta in herramientas:
        print(f"- {herramienta.name}: {herramienta.description}")


def main() -> None:
    """Ejecuta la verificación local de herramientas del agente."""
    mostrar_herramientas()


if __name__ == "__main__":
    main()
