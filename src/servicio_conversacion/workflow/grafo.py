"""
grafo.py
========
Grafo LangGraph — EntrevistasCEV.

Soporta cuatro condiciones experimentales:
  baseline  — Sin BDI, sin RAG
  rag_only  — Sin BDI, con RAG
  bdi_only  — Con BDI, sin RAG
  bdi_rag   — Sistema completo (BDI + RAG)  ← producción por defecto

El flujo de cada condición sigue la misma lógica que generador_trazas.py,
reutilizando los mismos nodos del sistema de producción.
"""

from functools import lru_cache

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from workflow.aristas import debe_resumir_conversacion
from nodos import (
    nodo_macro_bdi,
    nodo_micro_bdi,
    nodo_conversacion,
    nodo_resumir_conversacion,
    nodo_recuperador,
    nodo_resumir_contexto,
    nodo_conector,
)
from estado import EstadoActor


# ── Nodo conversación sin herramientas RAG ──────────────────────────────────
# Usado en las condiciones baseline y bdi_only donde el agente responde
# directamente sin poder invocar la herramienta de recuperación.

async def _nodo_conv_sin_rag(estado: EstadoActor, config: RunnableConfig):
    from chains import obtener_cadena_actor

    mensajes = estado.get("messages", [])
    mensajes_humanos = [
        m for m in mensajes
        if (hasattr(m, "type") and m.type == "human")
        or (hasattr(m, "role") and m.role == "user")
    ]
    numero_turno = max(len(mensajes_humanos), 1)

    chain = obtener_cadena_actor(
        actor_nombre              = estado["actor_nombre"],
        actor_perspectiva         = estado["actor_perspectiva"],
        actor_estilo              = estado["actor_estilo"],
        actor_contexto            = estado.get("actor_contexto", ""),
        id_actor                  = estado["id_actor"],
        macro_deseo               = estado.get("macro_deseo", ""),
        parametros_comportamiento = estado.get("parametros_comportamiento", ""),
        micro_intencion           = estado.get("micro_intencion", ""),
        resumen_conversacion      = estado.get("resumen_conversacion", ""),
        numero_turno              = numero_turno,
        usar_herramientas         = False,
    )
    response = await chain.ainvoke({"messages": estado["messages"]}, config)
    print("💬 RESPUESTA DIRECTA (sin RAG)")
    return {"messages": response}


# ── Builders de grafo por condición ─────────────────────────────────────────

def _crear_grafo_baseline() -> StateGraph:
    """Sin BDI, sin RAG."""
    g = StateGraph(EstadoActor)
    g.add_node("nodo_conversacion",         _nodo_conv_sin_rag)
    g.add_node("nodo_conector",             nodo_conector)
    g.add_node("nodo_resumir_conversacion", nodo_resumir_conversacion)

    g.add_edge(START, "nodo_conversacion")
    g.add_edge("nodo_conversacion", "nodo_conector")
    g.add_conditional_edges("nodo_conector", debe_resumir_conversacion)
    g.add_edge("nodo_resumir_conversacion", END)
    return g


def _crear_grafo_rag_only() -> StateGraph:
    """Sin BDI, con RAG."""
    g = StateGraph(EstadoActor)
    g.add_node("nodo_conversacion",         nodo_conversacion)
    g.add_node("nodo_rag",                  nodo_recuperador)
    g.add_node("nodo_resumir_contexto",     nodo_resumir_contexto)
    g.add_node("nodo_conector",             nodo_conector)
    g.add_node("nodo_resumir_conversacion", nodo_resumir_conversacion)

    g.add_edge(START, "nodo_conversacion")
    g.add_conditional_edges(
        "nodo_conversacion",
        tools_condition,
        {"tools": "nodo_rag", END: "nodo_conector"},
    )
    g.add_edge("nodo_rag",              "nodo_resumir_contexto")
    g.add_edge("nodo_resumir_contexto", "nodo_conversacion")
    g.add_conditional_edges("nodo_conector", debe_resumir_conversacion)
    g.add_edge("nodo_resumir_conversacion", END)
    return g


def _crear_grafo_bdi_only() -> StateGraph:
    """Con BDI, sin RAG."""
    g = StateGraph(EstadoActor)
    g.add_node("nodo_macro_bdi",            nodo_macro_bdi)
    g.add_node("nodo_micro_bdi",            nodo_micro_bdi)
    g.add_node("nodo_conversacion",         _nodo_conv_sin_rag)
    g.add_node("nodo_conector",             nodo_conector)
    g.add_node("nodo_resumir_conversacion", nodo_resumir_conversacion)

    g.add_edge(START,              "nodo_macro_bdi")
    g.add_edge("nodo_macro_bdi",   "nodo_micro_bdi")
    g.add_edge("nodo_micro_bdi",   "nodo_conversacion")
    g.add_edge("nodo_conversacion", "nodo_conector")
    g.add_conditional_edges("nodo_conector", debe_resumir_conversacion)
    g.add_edge("nodo_resumir_conversacion", END)
    return g


def _crear_grafo_bdi_rag() -> StateGraph:
    """Sistema completo: BDI + RAG."""
    g = StateGraph(EstadoActor)
    g.add_node("nodo_macro_bdi",             nodo_macro_bdi)
    g.add_node("nodo_micro_bdi",             nodo_micro_bdi)
    g.add_node("nodo_conversacion",          nodo_conversacion)
    g.add_node("nodo_rag",                   nodo_recuperador)
    g.add_node("nodo_resumir_conversacion",  nodo_resumir_conversacion)
    g.add_node("nodo_resumir_contexto",      nodo_resumir_contexto)
    g.add_node("nodo_conector",              nodo_conector)

    g.add_edge(START,             "nodo_macro_bdi")
    g.add_edge("nodo_macro_bdi",  "nodo_micro_bdi")
    g.add_edge("nodo_micro_bdi",  "nodo_conversacion")
    g.add_conditional_edges(
        "nodo_conversacion",
        tools_condition,
        {"tools": "nodo_rag", END: "nodo_conector"},
    )
    g.add_edge("nodo_rag",              "nodo_resumir_contexto")
    g.add_edge("nodo_resumir_contexto", "nodo_conversacion")
    g.add_conditional_edges("nodo_conector", debe_resumir_conversacion)
    g.add_edge("nodo_resumir_conversacion", END)
    return g


_BUILDERS = {
    "baseline": _crear_grafo_baseline,
    "rag_only": _crear_grafo_rag_only,
    "bdi_only": _crear_grafo_bdi_only,
    "bdi_rag" : _crear_grafo_bdi_rag,
}


@lru_cache(maxsize=4)
def crear_grafo_flujo(condicion: str = "bdi_rag") -> StateGraph:
    builder_fn = _BUILDERS.get(condicion, _crear_grafo_bdi_rag)
    return builder_fn()


# Compilado sin checkpointer — para pruebas locales
grafo = crear_grafo_flujo().compile()
