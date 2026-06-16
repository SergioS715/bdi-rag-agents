"""
nodos.py
========
Nodos del grafo LangGraph — EntrevistasCEV.

Cambios respecto a la versión anterior:
  - Se agrega nodo_macro_bdi: inicializa el Macro-BDI del agente
    antes de que comience la conversación.
  - nodo_conversacion ahora inyecta macro_deseo y parametros_comportamiento
    al prompt si están disponibles.
  - Todo lo demás permanece igual.
"""

from langchain_core.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import ToolNode

from chains import (
    obtener_cadena_resumen_contexto,
    obtener_cadena_resumen_conversacion,
    obtener_cadena_actor,
)
from estado import EstadoActor
from tools import herramientas
from config import TOTAL_MESSAGES_AFTER_SUMMARY

# Importaciones BDI
from personalidad import construir_personalidad
from macro_bdi import generar_macro_bdi
from micro_bdi import procesar_turno_micro_bdi, micro_creencia_desde_estado

nodo_recuperador = ToolNode(herramientas)


# ══════════════════════════════════════════════════════════════════════════════
# NODO NUEVO — Macro-BDI
# Se ejecuta UNA SOLA VEZ al inicio de cada sesión de simulación.
# En el modo chatbot estándar este nodo no se agrega al grafo.
# ══════════════════════════════════════════════════════════════════════════════

async def nodo_macro_bdi(estado: EstadoActor):
    """
    Inicializa el Macro-BDI del agente.

    Implementa §3.4.7 del paper: recibe el perfil textual del actor,
    calcula PersonalityFeatures y genera el macro_deseo via LLM.

    Posición en el grafo (modo simulación):
        START → nodo_macro_bdi → nodo_conversacion → ...

    Escribe en el estado:
        macro_deseo              : objetivo narrativo del agente (oración única)
        macro_creencia           : perfil de personalidad formateado para prompts
        parametros_comportamiento: parámetros de comportamiento clave

    No modifica messages ni ningún otro campo existente.
    """
    rol = estado["id_actor"]

    # Construir perfil textual desde los campos que ya están en el estado
    perfil_texto = (
        estado["actor_perspectiva"] +
        "\n" +
        estado["actor_estilo"]
    )

    # Verificar si ya tiene macro_deseo (evitar recalcular en cada turno)
    if estado.get("macro_deseo", ""):
        return {}

    print(f"🧠 MACRO-BDI INIT → calculando personalidad para '{rol}'...")

    # Paso 1: estimar personalidad (MBTI + Enneagram + 24 features)
    personalidad = construir_personalidad(perfil_texto, rol)

    # Paso 2: generar macro_deseo
    macro = generar_macro_bdi(rol, personalidad)

    print(f"🎯 MACRO DESIRE → {macro.macro_deseo}")
    print("NODO_MACRO_BDI EJECUTANDOSE")
    print("macro_deseo actual:", estado.get("macro_deseo"))
    return {
        "macro_deseo"              : macro.macro_deseo,
        "macro_creencia"           : macro.macro_creencia,
        "parametros_comportamiento": personalidad.resumen_para_prompt(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# NODOS ORIGINALES — con extensión mínima en nodo_conversacion
# ══════════════════════════════════════════════════════════════════════════════

async def nodo_conversacion(estado: EstadoActor, config: RunnableConfig):
    resumen_conversacion      = estado.get("resumen_conversacion", "")
    macro_deseo               = estado.get("macro_deseo", "")
    parametros_comportamiento = estado.get("parametros_comportamiento", "")
    micro_intencion           = estado.get("micro_intencion", "")

    # Si el último mensaje es un ToolMessage, ya recibió el contexto RAG.
    # No enlazar herramientas para forzar respuesta directa y cortar el bucle.
    mensajes = estado.get("messages", [])
    ya_uso_rag = (
        bool(mensajes)
        and hasattr(mensajes[-1], "type")
        and mensajes[-1].type == "tool"
    )

    # Calcular número de turno contando mensajes humanos ya procesados
    mensajes_humanos = [
        m for m in mensajes
        if (hasattr(m, "type") and m.type == "human")
        or (hasattr(m, "role") and m.role == "user")
    ]
    numero_turno = max(len(mensajes_humanos), 1)

    # Pasar las variables directamente al construir la cadena
    conversation_chain = obtener_cadena_actor(
        actor_nombre              = estado["actor_nombre"],
        actor_perspectiva         = estado["actor_perspectiva"],
        actor_estilo              = estado["actor_estilo"],
        actor_contexto            = estado["actor_contexto"],
        id_actor                  = estado["id_actor"],
        macro_deseo               = macro_deseo,
        parametros_comportamiento = parametros_comportamiento,
        micro_intencion           = micro_intencion,
        resumen_conversacion      = resumen_conversacion,
        numero_turno              = numero_turno,
        usar_herramientas         = not ya_uso_rag,
    )

    # Ahora ainvoke solo recibe messages — el system ya está renderizado
    response = await conversation_chain.ainvoke(
        {"messages": estado["messages"]},
        config,
    )

    if response.tool_calls:
        print(f"🔍 RAG ACTIVADO → query: {response.tool_calls[0]['args']}")
    else:
        print(f"💬 RESPUESTA DIRECTA (sin RAG)")

    return {"messages": response}


async def nodo_resumir_conversacion(estado: EstadoActor):
    resumen_conversacion = estado.get("resumen_conversacion", "")
    summary_chain = obtener_cadena_resumen_conversacion(
        resumen_conversacion = resumen_conversacion,
        actor_nombre         = estado["actor_nombre"],
    )
    response = await summary_chain.ainvoke(
        {"messages": estado["messages"]}
    )
    delete_messages = [
        RemoveMessage(id=m.id)
        for m in estado["messages"][: -TOTAL_MESSAGES_AFTER_SUMMARY]
    ]
    return {"resumen_conversacion": response.content, "messages": delete_messages}

async def nodo_resumir_contexto(estado: EstadoActor):
    from langchain_core.messages import ToolMessage
    from tools import _limpiar_tokens_anonimizacion

    mensajes = estado["messages"]
    # Buscar el último ToolMessage — no asumir que es el último mensaje
    ultimo = None
    for msg in reversed(mensajes):
        if isinstance(msg, ToolMessage) or (hasattr(msg, "type") and msg.type == "tool"):
            ultimo = msg
            break

    if ultimo is None:
        print("⚠  nodo_resumir_contexto: no se encontró ToolMessage en el historial")
        return {}

    context_summary_chain = obtener_cadena_resumen_contexto(
        id_actor=estado.get("id_actor", "")
    )
    response = await context_summary_chain.ainvoke(
        {"context": ultimo.content}
    )

    # Limpiar tokens de anonimización del resumen antes de inyectar al actor
    contexto_limpio = _limpiar_tokens_anonimizacion(response.content)

    nuevo_tool_msg = ToolMessage(
        content      = contexto_limpio,
        tool_call_id = ultimo.tool_call_id,
    )

    print(f"📄 CONTEXTO RAG → {len(contexto_limpio)} caracteres resumidos (limpio)")

    # Capturar metadata de calidad del retrieval (para métricas de evaluación)
    try:
        from tools import get_ultimo_retrieval_metadata
        retrieval_metadata = get_ultimo_retrieval_metadata()
    except Exception:
        retrieval_metadata = {}

    return {
        "messages"          : [RemoveMessage(id=ultimo.id), nuevo_tool_msg],
        "actor_contexto"    : contexto_limpio,
        "retrieval_metadata": retrieval_metadata,
    }


async def nodo_micro_bdi(estado: EstadoActor):
    """
    Ejecuta el ciclo Micro-BDI completo para el turno actual (§3.5).
    Corre ANTES de nodo_conversacion en cada turno de la simulación.

    Lee el último mensaje del historial, pasa por los 4 pasos del
    Micro-BDI y escribe micro_intencion + estado relacional en el State.

    En modo chatbot (macro_deseo vacío) retorna {} sin hacer nada.
    """
    # Modo chatbot: si no hay macro_deseo, el Micro-BDI no opera
    if not estado.get("macro_deseo", ""):
        return {}

    # Obtener el último mensaje humano
    mensajes = estado.get("messages", [])
    if not mensajes:
        return {}

    ultimo_humano = ""
    for msg in reversed(mensajes):
        # LangGraph usa role "human" para mensajes del usuario
        if hasattr(msg, "type") and msg.type == "human":
            ultimo_humano = msg.content
            break
        elif hasattr(msg, "role") and msg.role == "user":
            ultimo_humano = msg.content
            break

    if not ultimo_humano:
        return {}

    print(f"🔄 MICRO-BDI → procesando turno para '{estado['id_actor']}'...")

    # Reconstruir micro_creencia del turno anterior desde el estado
    creencia_anterior = micro_creencia_desde_estado(estado)

    # Ejecutar ciclo completo §3.5.1 → §3.5.4
    resultado = procesar_turno_micro_bdi(
        rol             = estado["id_actor"],
        ultimo_mensaje  = ultimo_humano,
        macro_deseo     = estado.get("macro_deseo", ""),
        behavior_params = estado.get("parametros_comportamiento", ""),
        creencia_actual = creencia_anterior,
        actor_contexto  = estado.get("actor_contexto", ""),
    )

    print(
        f"  Etapa: {resultado.micro_deseo.etapa_discusion} | "
        f"Tension: {resultado.micro_creencia.nivel_tension:.2f} | "
        f"Tipo: {resultado.turno_analisis.tipo}"
    )

    return {
        "micro_intencion"          : resultado.micro_intencion_str(),
        "micro_deseo"              : resultado.micro_deseo_str(),
        "nivel_tension"            : resultado.nivel_tension_str(),
        "conteo_negaciones"        : resultado.conteo_negaciones_str(),
        "reconocimiento_dado"      : resultado.reconocimiento_dado_str(),
        "credibilidad_interlocutor": resultado.credibilidad_interlocutor_str(),
        "afinidad_interlocutor"    : resultado.afinidad_interlocutor_str(),
        "conteo_acusaciones"       : resultado.conteo_acusaciones_str(),
    }


async def nodo_conector(estado: EstadoActor):
    """Sin cambios respecto a la versión original."""
    return {}
