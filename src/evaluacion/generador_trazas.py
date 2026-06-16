"""
generador_trazas.py
===================
Genera trazas de conversación para las cuatro condiciones experimentales
del sistema EntrevistasCEV y las guarda en MongoDB.

Condiciones:
  baseline  — Solo prompt base, sin BDI, sin RAG
  rag_only  — Prompt + RAG, sin BDI
  bdi_only  — BDI activo, sin RAG
  bdi_rag   — Sistema completo (BDI + RAG)

Uso:
    python generador_trazas.py --condicion baseline
    python generador_trazas.py --condicion rag_only
    python generador_trazas.py --condicion bdi_only
    python generador_trazas.py --condicion bdi_rag
    python generador_trazas.py --condicion all

Las trazas se guardan en:
  DB: EntrevistasCEV / Colección: evaluacion_trazas
"""

import asyncio
import argparse
import random
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional

# ─── Seed fijo para reproducibilidad entre ciclos de evaluación ───────────────
SEED = 42
random.seed(SEED)
try:
    import numpy as np
    np.random.seed(SEED)
except ImportError:
    pass

# ─── Ajuste de rutas ─────────────────────────────────────────────────────────
_ROOT     = Path(__file__).resolve().parent.parent.parent
_SRC      = _ROOT / "src"
_WORKFLOW = _SRC / "servicio_conversacion" / "workflow"
_PERS     = _SRC / "personalidades"
_SERVICIO = _SRC / "servicio_conversacion"

for _p in [str(_ROOT), str(_SRC), str(_WORKFLOW), str(_PERS), str(_SERVICIO)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─── Imports del sistema ──────────────────────────────────────────────────────
from pymongo import MongoClient
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.checkpoint.mongodb import MongoDBSaver
from jinja2 import Template

from config import OPENAI_API_KEY, OPENAI_LLM_MODEL, MONGO_URI, DB_NAME as MONGO_DB
from estado import EstadoActor
from prompts import TARJETA_PERSONAJE_ACTOR
from tools import herramientas, recuperador as _verificador
from crearActores import ActoresInvolucrados
from nodos import (
    nodo_conversacion,
    nodo_recuperador,
    nodo_resumir_contexto,
    nodo_conector,
    nodo_resumir_conversacion,
    nodo_macro_bdi,
    nodo_micro_bdi,
)
from aristas import debe_resumir_conversacion

# Importar banco de preguntas del mismo directorio
sys.path.insert(0, str(Path(__file__).resolve().parent))
from banco_pruebas import BANCO_PREGUNTAS, Pregunta

# ─── Configuración MongoDB ────────────────────────────────────────────────────
COL_TRAZAS       = "evaluacion_trazas"
COL_CHECKPOINTS  = "langgraph_checkpoints_eval"   # separado de producción
COL_WRITES       = "langgraph_writes_eval"


# ─── Modelo de traza por turno ────────────────────────────────────────────────

@dataclass
class TurnoTraza:
    condicion                : str
    actor_id                 : str
    turno                    : int
    tipologia                : str
    pregunta                 : str
    respuesta                : str
    macro_deseo              : str
    micro_deseo              : str
    micro_intencion          : str
    actor_contexto           : str
    parametros_comportamiento: str
    nivel_tension            : str
    conteo_negaciones        : str
    reconocimiento_dado      : str
    retrieval_metadata       : dict  # role_purity, n_docs_total, n_docs_rol, etc.
    verificacion_contexto    : str   # chunks recuperados con la RESPUESTA como query (factual_grounding)
    respuesta_invalida       : bool  # True si el agente generó un rechazo en lugar de responder en rol


# ─── Nodo conversación sin RAG (baseline) ────────────────────────────────────

async def _nodo_conv_sin_rag(estado: EstadoActor, config: RunnableConfig):
    """Baseline: sin herramientas RAG ni BDI."""
    model = ChatOpenAI(
        api_key           = OPENAI_API_KEY,
        model             = OPENAI_LLM_MODEL,
        temperature       = 0.7,
        top_p             = 0.9,
        frequency_penalty = 0.2,
        presence_penalty  = 0.1,
    )
    system_rendered = Template(TARJETA_PERSONAJE_ACTOR).render(
        actor_nombre              = estado["actor_nombre"],
        actor_perspectiva         = estado["actor_perspectiva"],
        actor_estilo              = estado["actor_estilo"],
        actor_contexto            = "",
        id_actor                  = estado["id_actor"],
        macro_deseo               = "",
        parametros_comportamiento = "",
        micro_intencion           = "",
        resumen_conversacion      = estado.get("resumen_conversacion", ""),
    )
    prompt   = ChatPromptTemplate.from_messages([
        ("system", system_rendered),
        MessagesPlaceholder(variable_name="messages"),
    ])
    response = await (prompt | model).ainvoke({"messages": estado["messages"]}, config)
    return {"messages": response}


# ─── Nodo conversación con BDI, sin RAG ──────────────────────────────────────

async def _nodo_conv_bdi_sin_rag(estado: EstadoActor, config: RunnableConfig):
    """
    bdi_only: usa el estado BDI (macro_deseo, parametros_comportamiento,
    micro_intencion) pero no llama ninguna herramienta RAG.
    """
    model = ChatOpenAI(
        api_key           = OPENAI_API_KEY,
        model             = OPENAI_LLM_MODEL,
        temperature       = 0.7,
        top_p             = 0.9,
        frequency_penalty = 0.2,
        presence_penalty  = 0.1,
    )
    system_rendered = Template(TARJETA_PERSONAJE_ACTOR).render(
        actor_nombre              = estado["actor_nombre"],
        actor_perspectiva         = estado["actor_perspectiva"],
        actor_estilo              = estado["actor_estilo"],
        actor_contexto            = "",
        id_actor                  = estado["id_actor"],
        macro_deseo               = estado.get("macro_deseo", ""),
        parametros_comportamiento = estado.get("parametros_comportamiento", ""),
        micro_intencion           = estado.get("micro_intencion", ""),
        resumen_conversacion      = estado.get("resumen_conversacion", ""),
    )
    prompt   = ChatPromptTemplate.from_messages([
        ("system", system_rendered),
        MessagesPlaceholder(variable_name="messages"),
    ])
    response = await (prompt | model).ainvoke({"messages": estado["messages"]}, config)
    return {"messages": response}


# ─── Nodo conversación con RAG FORZADO (evaluación) ──────────────────────────

async def _nodo_conv_eval_rag(estado: EstadoActor, config: RunnableConfig):
    """
    Variante de nodo_conversacion para evaluación que FUERZA la llamada RAG
    en la primera invocación de cada turno (tool_choice='required').
    En la segunda invocación (ya_uso_rag=True) genera respuesta directa.
    Esto garantiza que actor_contexto siempre se pueble para medir faithfulness.
    """
    from chains import obtener_cadena_actor

    mensajes   = estado.get("messages", [])
    ya_uso_rag = (
        bool(mensajes)
        and hasattr(mensajes[-1], "type")
        and mensajes[-1].type == "tool"
    )

    # Calcular número de turno igual que en nodo_conversacion de producción
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
        usar_herramientas         = True,
        tool_choice               = "none" if ya_uso_rag else "required",
    )

    ctx_len = len(estado.get("actor_contexto", ""))
    if ya_uso_rag:
        print(f"💬 RESPUESTA DIRECTA (post-RAG) — actor_contexto: {ctx_len} chars")
    else:
        print(f"🔍 Llamando RAG (tool_choice=required)...")

    response = await chain.ainvoke({"messages": estado["messages"]}, config)

    if getattr(response, "tool_calls", None):
        print(f"   → tool_call: {response.tool_calls[0]['args']}")
    elif not ya_uso_rag:
        print(f"⚠  RAG no fue llamado aunque tool_choice=required")

    return {"messages": response}


# ─── Constructores de grafos por condición ────────────────────────────────────

def _crear_grafo_baseline() -> StateGraph:
    """Grafo sin BDI y sin RAG."""
    g = StateGraph(EstadoActor)
    g.add_node("nodo_conversacion",         _nodo_conv_sin_rag)
    g.add_node("nodo_conector",             nodo_conector)
    g.add_node("nodo_resumir_conversacion", nodo_resumir_conversacion)

    g.add_edge(START,               "nodo_conversacion")
    g.add_edge("nodo_conversacion", "nodo_conector")
    g.add_conditional_edges("nodo_conector", debe_resumir_conversacion)
    g.add_edge("nodo_resumir_conversacion", END)
    return g


def _crear_grafo_rag_only() -> StateGraph:
    """
    Grafo con RAG forzado, sin BDI.
    Usa _nodo_conv_eval_rag que garantiza tool_choice='required' en cada turno.
    """
    g = StateGraph(EstadoActor)
    g.add_node("nodo_conversacion",         _nodo_conv_eval_rag)
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
    """
    Grafo BDI activo sin RAG.
    Incluye macro_bdi y micro_bdi pero la conversación se genera
    directamente sin llamar herramientas de recuperación.
    """
    from nodos import nodo_macro_bdi, nodo_micro_bdi
    from aristas import debe_resumir_conversacion as _debe_resumir

    g = StateGraph(EstadoActor)
    g.add_node("nodo_macro_bdi",             nodo_macro_bdi)
    g.add_node("nodo_micro_bdi",             nodo_micro_bdi)
    g.add_node("nodo_conversacion",          _nodo_conv_bdi_sin_rag)
    g.add_node("nodo_conector",              nodo_conector)
    g.add_node("nodo_resumir_conversacion",  nodo_resumir_conversacion)

    g.add_edge(START,             "nodo_macro_bdi")
    g.add_edge("nodo_macro_bdi",  "nodo_micro_bdi")
    g.add_edge("nodo_micro_bdi",  "nodo_conversacion")
    g.add_edge("nodo_conversacion", "nodo_conector")
    g.add_conditional_edges("nodo_conector", _debe_resumir)
    g.add_edge("nodo_resumir_conversacion", END)
    return g


def _crear_grafo_bdi_rag() -> StateGraph:
    """
    Grafo completo BDI+RAG para evaluación.
    Usa _nodo_conv_eval_rag en lugar de nodo_conversacion para forzar RAG.
    """
    from nodos import nodo_macro_bdi, nodo_micro_bdi
    from aristas import debe_resumir_conversacion as _debe_resumir

    g = StateGraph(EstadoActor)
    g.add_node("nodo_macro_bdi",             nodo_macro_bdi)
    g.add_node("nodo_micro_bdi",             nodo_micro_bdi)
    g.add_node("nodo_conversacion",          _nodo_conv_eval_rag)
    g.add_node("nodo_rag",                   nodo_recuperador)
    g.add_node("nodo_resumir_contexto",      nodo_resumir_contexto)
    g.add_node("nodo_conector",              nodo_conector)
    g.add_node("nodo_resumir_conversacion",  nodo_resumir_conversacion)

    g.add_edge(START,              "nodo_macro_bdi")
    g.add_edge("nodo_macro_bdi",   "nodo_micro_bdi")
    g.add_edge("nodo_micro_bdi",   "nodo_conversacion")
    g.add_conditional_edges(
        "nodo_conversacion",
        tools_condition,
        {"tools": "nodo_rag", END: "nodo_conector"},
    )
    g.add_edge("nodo_rag",              "nodo_resumir_contexto")
    g.add_edge("nodo_resumir_contexto", "nodo_conversacion")
    g.add_conditional_edges("nodo_conector", _debe_resumir)
    g.add_edge("nodo_resumir_conversacion", END)
    return g


# ─── Extracción de contexto RAG del historial de mensajes ────────────────────

def _extraer_contexto_rag(messages: list) -> str:
    """
    Devuelve el contenido del ToolMessage más reciente que aparece
    DESPUÉS del último HumanMessage en el historial.
    Retorna "" si no hubo RAG en este turno.
    """
    last_human_idx = -1
    for i, msg in enumerate(messages):
        tipo = getattr(msg, "type", None)
        if isinstance(msg, HumanMessage) or tipo == "human":
            last_human_idx = i

    if last_human_idx < 0:
        return ""

    for msg in messages[last_human_idx:]:
        tipo = getattr(msg, "type", None)
        if isinstance(msg, ToolMessage) or tipo == "tool":
            return msg.content or ""

    return ""


# ─── Verificación factual: retrieval con la respuesta como query ──────────────

_ROL_NORM = {
    "víctima"     : "victima",
    "victima"     : "victima",
    "victimario"  : "victimario",
    "actor_armado": "victimario",
    "tercero"     : "tercero",
}


def _es_respuesta_invalida(respuesta: str) -> bool:
    """
    Detecta si la respuesta del agente es un rechazo del modelo en lugar de
    una respuesta en rol. Esto ocurre cuando el LLM bloquea preguntas confrontativas
    o temas del conflicto armado por sus propios filtros de seguridad.
    """
    if not respuesta or len(respuesta.strip()) < 10:
        return True

    text_lower = respuesta.lower().strip()

    patrones_rechazo = [
        "lo siento",
        "no puedo",
        "no puedo ayudar",
        "i'm sorry",
        "i cannot",
        "i'm not able",
        "disculpa",
        "disculpe",
        "como asistente",
        "como modelo de lenguaje",
        "as an ai",
        "as a language model",
        "no me es posible",
        "está fuera de mis capacidades",
        "no tengo la capacidad",
        "no es apropiado",
        "no es adecuado para mí",
    ]

    return any(text_lower.startswith(p) or f". {p}" in text_lower for p in patrones_rechazo)


async def _recuperar_verificacion(respuesta: str, actor_id: str) -> str:
    """
    Recupera chunks del corpus CEV usando la RESPUESTA del agente como query,
    no la pregunta del entrevistador.

    Propósito: comprobar si las afirmaciones factuales de la respuesta tienen
    respaldo en el corpus, independientemente de qué se recuperó como contexto
    RAG en este turno (actor_contexto). Esto alimenta la métrica factual_grounding.

    Se ejecuta para TODAS las condiciones (baseline, rag_only, bdi_only, bdi_rag)
    porque mide conocimiento general del corpus, no uso del RAG.
    """
    if not respuesta.strip():
        return ""

    # Usar los primeros 400 caracteres de la respuesta como query de verificación
    query = f"testimonio {actor_id} conflicto armado colombiano: {respuesta[:400]}"
    try:
        docs = await asyncio.to_thread(_verificador.invoke, query)
    except Exception as exc:
        print(f"⚠  verificacion_contexto no disponible: {exc}")
        return ""

    if not docs:
        return ""

    # Filtrar por rol (misma lógica que tools.py)
    rol_norm = _ROL_NORM.get(actor_id, actor_id)
    docs_rol = [d for d in docs if _ROL_NORM.get(
        d.metadata.get("clasificacion_final", "").strip().lower(), ""
    ) == rol_norm]
    docs_final = docs_rol if docs_rol else docs

    fragmentos = []
    for i, d in enumerate(docs_final, 1):
        rol_source = d.metadata.get("clasificacion_final", "desconocido").strip()
        fragmentos.append(
            f"--- FRAGMENTO {i} [ROL: {rol_source.upper()}] ---\n{d.page_content}"
        )
    return "\n\n".join(fragmentos)


# ─── Runner por condición y actor ─────────────────────────────────────────────

async def _run_condicion(
    condicion        : str,
    actor_id         : str,
    actor_nombre     : str,
    actor_perspectiva: str,
    actor_estilo     : str,
    preguntas        : List[Pregunta],
    db,
) -> None:
    """
    Ejecuta todas las preguntas de un actor bajo una condición experimental
    y guarda las trazas turno a turno en MongoDB.

    Idempotente: si ya existen registros para (condicion, actor_id),
    se omite el procesamiento.
    """
    thread_id = f"eval_{condicion}_{actor_id}"

    # ── Idempotencia ────────────────────────────────────────────────
    existentes = db[COL_TRAZAS].count_documents({
        "condicion": condicion,
        "actor_id" : actor_id,
    })
    if existentes >= len(preguntas):
        print(f"  ⏭  Saltando {condicion}/{actor_id} "
              f"({existentes} trazas ya guardadas)")
        return

    print(f"\n{'='*62}")
    print(f"  Condición : {condicion.upper():<12}  Actor : {actor_id}")
    print(f"  Thread ID : {thread_id}")
    print(f"{'='*62}")

    # ── Seleccionar constructor de grafo ─────────────────────────────
    if condicion == "baseline":
        graph_builder = _crear_grafo_baseline()
    elif condicion == "rag_only":
        graph_builder = _crear_grafo_rag_only()
    elif condicion == "bdi_only":
        graph_builder = _crear_grafo_bdi_only()
    else:
        graph_builder = _crear_grafo_bdi_rag()

    with MongoDBSaver.from_conn_string(
        conn_string                 = MONGO_URI,
        db_name                     = MONGO_DB,
        checkpoint_collection_name  = COL_CHECKPOINTS,
        writes_collection_name      = COL_WRITES,
    ) as checkpointer:

        grafo  = graph_builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        for i, pregunta in enumerate(preguntas, start=1):
            label = f"[{i:02d}/{len(preguntas)}]"
            print(f"  {label} {pregunta.tipologia:14s} "
                  f"→ {pregunta.texto[:55]}...")

            datos_entrada = {
                "messages"         : [HumanMessage(content=pregunta.texto)],
                "id_actor"         : actor_id,
                "actor_nombre"     : actor_nombre,
                "actor_perspectiva": actor_perspectiva,
                "actor_estilo"     : actor_estilo,
                "actor_contexto"   : "",
            }

            try:
                estado_salida = await grafo.ainvoke(
                    input  = datos_entrada,
                    config = config,
                )

                # Respuesta del agente (último AIMessage)
                ultimo = estado_salida["messages"][-1]
                respuesta = getattr(ultimo, "content", "") or ""

                # Validar que el agente respondió en rol y no generó un rechazo
                invalida = _es_respuesta_invalida(respuesta)
                if invalida:
                    print(f"       ⚠  Respuesta inválida (rechazo del modelo): "
                          f"{respuesta[:80]!r}")

                # Contexto RAG recuperado en este turno
                # Primero intentamos el campo de estado (fuente directa de nodo_resumir_contexto),
                # luego fallback a extracción de mensajes por si el estado no se propagó.
                actor_contexto = ""
                if condicion in ("rag_only", "bdi_rag"):  # bdi_only no usa RAG
                    actor_contexto = (
                        estado_salida.get("actor_contexto", "")
                        or _extraer_contexto_rag(estado_salida["messages"])
                    )

                # Chunks de verificación factual: solo si la respuesta es válida
                verificacion_contexto = ""
                if not invalida:
                    verificacion_contexto = await _recuperar_verificacion(
                        respuesta, actor_id
                    )

                traza = TurnoTraza(
                    condicion                = condicion,
                    actor_id                 = actor_id,
                    turno                    = i,
                    tipologia                = pregunta.tipologia,
                    pregunta                 = pregunta.texto,
                    respuesta                = respuesta,
                    macro_deseo              = estado_salida.get("macro_deseo", ""),
                    micro_deseo              = estado_salida.get("micro_deseo", ""),
                    micro_intencion          = estado_salida.get("micro_intencion", ""),
                    actor_contexto           = actor_contexto,
                    parametros_comportamiento= estado_salida.get("parametros_comportamiento", ""),
                    nivel_tension            = estado_salida.get("nivel_tension", "0.0"),
                    conteo_negaciones        = estado_salida.get("conteo_negaciones", "0"),
                    reconocimiento_dado      = estado_salida.get("reconocimiento_dado", "0.0"),
                    retrieval_metadata       = estado_salida.get("retrieval_metadata", {}),
                    verificacion_contexto    = verificacion_contexto,
                    respuesta_invalida       = invalida,
                )

                db[COL_TRAZAS].insert_one(asdict(traza))
                tiene_rag = "con RAG" if actor_contexto else "sin RAG"
                estado_str = "⚠  INVÁLIDA" if invalida else "✓  Guardado"
                print(f"       {estado_str} ({tiene_rag})")

            except Exception as exc:
                print(f"       ✗  Error en turno {i}: {exc}")

            time.sleep(1.5)   # respeta rate limit de Groq


# ─── Test rápido: imprime actor_contexto sin guardar en MongoDB ───────────────

async def test_actor_contexto(actor_id: str, pregunta: str) -> None:
    """
    Ejecuta un único turno RAG forzado y muestra el actor_contexto resultante
    en terminal. No guarda nada en MongoDB ni usa checkpointer.

    Uso:
        python generador_trazas.py --test-contexto \
            --actor-id victima \
            --pregunta "¿Qué recuerda del día que llegaron?"
    """
    factory = ActoresInvolucrados()
    actor   = factory.obtener_actor(actor_id)

    graph_builder = _crear_grafo_rag_only()
    grafo  = graph_builder.compile()                          # sin checkpointer
    config = {"configurable": {"thread_id": f"test_{actor_id}"}}

    datos = {
        "messages"         : [HumanMessage(content=pregunta)],
        "id_actor"         : actor_id,
        "actor_nombre"     : actor.nombre,
        "actor_perspectiva": actor.perspectiva,
        "actor_estilo"     : actor.estilo,
        "actor_contexto"   : "",
    }

    print(f"\n{'='*62}")
    print(f"  TEST ACTOR_CONTEXTO")
    print(f"  Actor   : {actor_id}")
    print(f"  Pregunta: {pregunta}")
    print(f"{'='*62}\n")

    estado = await grafo.ainvoke(input=datos, config=config)

    actor_contexto = (
        estado.get("actor_contexto", "")
        or _extraer_contexto_rag(estado["messages"])
    )

    print(f"\n{'='*62}")
    print("  ACTOR_CONTEXTO RESULTANTE")
    print(f"{'='*62}")
    print(actor_contexto or "(vacío — RAG no recuperó fragmentos)")
    print(f"{'='*62}\n")


# ─── Punto de entrada ─────────────────────────────────────────────────────────

async def generar_trazas(condicion: str = "all") -> None:
    """
    Genera trazas de conversación para las condiciones indicadas.

    Args:
        condicion: "baseline" | "rag_only" | "bdi_rag" | "all"
    """
    client = MongoClient(MONGO_URI)
    db     = client[MONGO_DB]

    # Crear índice único para idempotencia
    db[COL_TRAZAS].create_index(
        [("condicion", 1), ("actor_id", 1), ("turno", 1)],
        unique=True,
    )

    condiciones = (
        ["baseline", "rag_only", "bdi_only", "bdi_rag"]
        if condicion == "all"
        else [condicion]
    )

    factory = ActoresInvolucrados()

    for cond in condiciones:
        for actor_id in ["victima", "victimario", "tercero"]:
            actor     = factory.obtener_actor(actor_id)
            preguntas = BANCO_PREGUNTAS[actor_id]
            await _run_condicion(
                condicion         = cond,
                actor_id          = actor_id,
                actor_nombre      = actor.nombre,
                actor_perspectiva = actor.perspectiva,
                actor_estilo      = actor.estilo,
                preguntas         = preguntas,
                db                = db,
            )

    client.close()
    print("\n✅  Generación de trazas completada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generador de trazas — EntrevistasCEV"
    )
    parser.add_argument(
        "--condicion",
        choices=["baseline", "rag_only", "bdi_only", "bdi_rag", "all"],
        default="all",
        help="Condición experimental a generar (default: all)",
    )
    parser.add_argument(
        "--test-contexto",
        action="store_true",
        help="Modo test: ejecuta un turno RAG y muestra actor_contexto sin guardar en MongoDB",
    )
    parser.add_argument(
        "--actor-id",
        choices=["victima", "victimario", "tercero"],
        default="victima",
        help="Actor a usar en --test-contexto (default: victima)",
    )
    parser.add_argument(
        "--pregunta",
        type=str,
        default="¿Qué recuerda del día que llegaron?",
        help="Pregunta a usar en --test-contexto",
    )
    args = parser.parse_args()

    if args.test_contexto:
        asyncio.run(test_actor_contexto(args.actor_id, args.pregunta))
    else:
        asyncio.run(generar_trazas(args.condicion))
