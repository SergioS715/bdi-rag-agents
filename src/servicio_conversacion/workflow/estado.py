"""
estado.py
=========
Estado del agente conversacional — EntrevistasCEV.

Campos Macro-BDI (agregados en iteración anterior):
    macro_deseo              : objetivo narrativo del agente para toda la simulación
    macro_creencia           : resumen de personalidad formateado para prompts
    parametros_comportamiento: parámetros de comportamiento clave (empathy, etc.)

Campos Micro-BDI (agregados ahora — §3.5 del paper):
    micro_intencion           : plan + intención del turno generados por micro_bdi.py
    micro_deseo               : discussion_stage CEV del turno actual
    nivel_tension             : tensión acumulada en la conversación (float como str)
    conteo_negaciones         : veces que el interlocutor negó responsabilidad (int como str)
    reconocimiento_dado       : si hubo reconocimiento del daño (float como str)
    credibilidad_interlocutor : credibilidad percibida del interlocutor (float como str)
    afinidad_interlocutor     : afinidad emocional hacia el interlocutor (float como str)
    conteo_acusaciones        : veces que el interlocutor acusó directamente (int como str)

Todos los campos nuevos tienen default "" para no romper
el modo chatbot actual ni los checkpoints existentes en MongoDB.
"""

from langgraph.graph import MessagesState


class EstadoActor(MessagesState):
    """Estado del agente conversacional — EntrevistasCEV."""

    # ── Campos originales ──────────────────────────────────────────────────
    id_actor          : str
    actor_nombre      : str
    actor_perspectiva : str
    actor_estilo      : str
    actor_contexto    : str
    resumen_conversacion: str

    # ── Campos Macro-BDI ───────────────────────────────────────────────────
    macro_deseo              : str   # "Quiero que reconozcan el daño sin pedirme que olvide..."
    macro_creencia           : str   # resumen Enneagram + parámetros para prompts
    parametros_comportamiento: str   # "empathy=0.78, aggressiveness=0.22, ..."

    # ── Campos Micro-BDI ───────────────────────────────────────────────────
    micro_intencion          : str   # "Plan del turno: ...\nIntención concreta: ..."
    micro_deseo              : str   # "defensa_posicion" | "confrontacion" | etc.
    nivel_tension            : str   # float serializado: "0.35"
    conteo_negaciones        : str   # int serializado:   "2"
    reconocimiento_dado      : str   # float serializado: "0.25"
    credibilidad_interlocutor: str   # float serializado: "0.5"
    afinidad_interlocutor    : str   # float serializado: "0.5"
    conteo_acusaciones       : str   # int serializado:   "0"

    # ── Métricas de calidad del retrieval (Parte A) ────────────────────────
    # Poblado por nodo_resumir_contexto después de cada llamada RAG.
    # Campos: n_docs_total, n_docs_rol, role_purity, usado_fallback,
    #         roles_encontrados
    retrieval_metadata       : dict  # {} cuando no hubo RAG en el turno


def estado_a_cadena(state: EstadoActor) -> str:
    """Útil para debug — imprime el estado actual del agente."""
    conversacion = (
        state.get("resumen_conversacion")  or
        state.get("messages") or
        ""
    )
    return (
        f"id={state['id_actor']} | "
        f"macro_deseo={state.get('macro_deseo','')[:50]}... | "
        f"micro_deseo={state.get('micro_deseo','')} | "
        f"tension={state.get('nivel_tension','0')} | "
        f"conversacion={str(conversacion)[:60]}..."
    )
