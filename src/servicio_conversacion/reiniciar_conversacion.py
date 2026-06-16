import logging
from pymongo import MongoClient

# ─── Configuración MongoDB ────────────────────────────────────────
# Mismas constantes que en generate_response.py
MONGO_URI                      = "mongodb://localhost:27017"
MONGO_DB_NAME                  = "EntrevistasCEV"
MONGO_COLECCION_CHECKPOINTS    = "langgraph_checkpoints"
MONGO_COLECCION_ESCRITURAS     = "langgraph_writes"

log = logging.getLogger(__name__)


async def reiniciar_estado_conversacion(id_actor: str = None) -> dict:
    """Elimina el estado de conversación guardado en MongoDB.

    Si se pasa un id_actor, solo borra los checkpoints de ese actor.
    Si no se pasa nada, borra todas las conversaciones (reset total).

    Args:
        id_actor: "victima" | "victimario" | "tercero" | None (borra todo)

    Returns:
        dict con status y mensaje descriptivo.

    Raises:
        Exception: Si hay error conectando a MongoDB o borrando datos.
    """
    try:
        client = MongoClient(MONGO_URI)
        db     = client[MONGO_DB_NAME]

        colecciones_existentes = db.list_collection_names()
        eliminadas = []

        if id_actor:
            # ── Borrar solo los checkpoints de un actor específico ──
            # LangGraph guarda el thread_id dentro del documento
            for nombre_col in [MONGO_COLECCION_CHECKPOINTS,
                                MONGO_COLECCION_ESCRITURAS]:
                if nombre_col in colecciones_existentes:
                    resultado = db[nombre_col].delete_many(
                        {"thread_id": {"$regex": f"^{id_actor}"}}
                    )
                    log.info(
                        f"Borrados {resultado.deleted_count} documentos "
                        f"de '{nombre_col}' para actor '{id_actor}'"
                    )
                    if resultado.deleted_count > 0:
                        eliminadas.append(
                            f"{nombre_col} ({resultado.deleted_count} docs)"
                        )
        else:
            # ── Reset total — borrar colecciones completas ──────────
            for nombre_col in [MONGO_COLECCION_CHECKPOINTS,
                                MONGO_COLECCION_ESCRITURAS]:
                if nombre_col in colecciones_existentes:
                    db.drop_collection(nombre_col)
                    eliminadas.append(nombre_col)
                    log.info(f"Colección eliminada: {nombre_col}")

        client.close()

        if eliminadas:
            return {
                "status"  : "success",
                "message" : f"Conversación reiniciada. Eliminado: {', '.join(eliminadas)}",
            }
        else:
            return {
                "status"  : "success",
                "message" : "No había estado previo que eliminar.",
            }

    except Exception as e:
        log.error(f"Error al reiniciar conversación: {str(e)}")
        raise Exception(f"Error al reiniciar conversación: {str(e)}")
