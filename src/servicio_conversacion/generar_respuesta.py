import sys
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator, Union

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.mongodb import MongoDBSaver

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    MONGO_URI,
    DB_NAME             as MONGO_BD_NOMBRE,
    MONGO_STATE_CHECKPOINT_COLLECTION as MONGO_COLECCION_CHECKPOINTS,
    MONGO_STATE_WRITES_COLLECTION     as MONGO_COLECCION_ESCRITURAS,
)

from workflow.grafo import crear_grafo_flujo
from workflow.estado import EstadoActor


# ─────────────────────────────────────────────────────────────────
# RESPUESTA COMPLETA (sin streaming)
# ─────────────────────────────────────────────────────────────────
async def obtener_respuesta(
    mensajes              : str | list[str] | list[dict[str, Any]],
    id_actor              : str,
    actor_nombre          : str,
    actor_perspectiva     : str,
    actor_estilo          : str,
    actor_contexto        : str,
    condicion             : str = "bdi_rag",
    hilo_nuevo            : bool = False,
    macro_deseo           : str = "",
    macro_creencia        : str = "",
    parametros_comportamiento : str = "",
) -> tuple[str, EstadoActor]:

    constructor_grafo = crear_grafo_flujo(condicion)

    try:
        with MongoDBSaver.from_conn_string(
            conn_string=MONGO_URI,
            db_name=MONGO_BD_NOMBRE,
            checkpoint_collection_name=MONGO_COLECCION_CHECKPOINTS,
            writes_collection_name=MONGO_COLECCION_ESCRITURAS,
        ) as checkpointer:

            grafo = constructor_grafo.compile(checkpointer=checkpointer)

            base_thread = f"{id_actor}-{condicion}"
            thread_id = (
                base_thread if not hilo_nuevo
                else f"{base_thread}-{uuid.uuid4()}"
            )

            config = {"configurable": {"thread_id": thread_id}}

            datos_entrada = {
                "messages"         : _formatear_mensajes(mensajes),
                "id_actor"         : id_actor,
                "actor_nombre"     : actor_nombre,
                "actor_perspectiva": actor_perspectiva,
                "actor_estilo"     : actor_estilo,
                "actor_contexto"   : actor_contexto,
            }

            if macro_deseo:
                datos_entrada["macro_deseo"] = macro_deseo

            if macro_creencia:
                datos_entrada["macro_creencia"] = macro_creencia

            if parametros_comportamiento:
                datos_entrada["parametros_comportamiento"] = parametros_comportamiento

            estado_salida = await grafo.ainvoke(
                input=datos_entrada,
                config=config,
            )

        ultimo_mensaje = estado_salida["messages"][-1]
        return ultimo_mensaje.content, EstadoActor(**estado_salida)

    except Exception as e:
        raise RuntimeError(f"Error ejecutando el grafo: {str(e)}") from e


# ─────────────────────────────────────────────────────────────────
# RESPUESTA STREAMING
# ─────────────────────────────────────────────────────────────────
async def obtener_respuesta_streaming(
    mensajes              : str | list[str] | list[dict[str, Any]],
    id_actor              : str,
    actor_nombre          : str,
    actor_perspectiva     : str,
    actor_estilo          : str,
    actor_contexto        : str,
    condicion             : str = "bdi_rag",
    hilo_nuevo            : bool = False,
    macro_deseo           : str = "",
    macro_creencia        : str = "",
    parametros_comportamiento : str = "",
) -> AsyncGenerator[str, None]:

    constructor_grafo = crear_grafo_flujo(condicion)

    try:
        with MongoDBSaver.from_conn_string(
            conn_string=MONGO_URI,
            db_name=MONGO_BD_NOMBRE,
            checkpoint_collection_name=MONGO_COLECCION_CHECKPOINTS,
            writes_collection_name=MONGO_COLECCION_ESCRITURAS,
        ) as checkpointer:

            grafo = constructor_grafo.compile(checkpointer=checkpointer)

            base_thread = f"{id_actor}-{condicion}"
            thread_id = (
                base_thread if not hilo_nuevo
                else f"{base_thread}-{uuid.uuid4()}"
            )

            config = {"configurable": {"thread_id": thread_id}}

            datos_entrada = {
                "messages"         : _formatear_mensajes(mensajes),
                "id_actor"         : id_actor,
                "actor_nombre"     : actor_nombre,
                "actor_perspectiva": actor_perspectiva,
                "actor_estilo"     : actor_estilo,
                "actor_contexto"   : actor_contexto,
            }

            if macro_deseo:
                datos_entrada["macro_deseo"] = macro_deseo

            if macro_creencia:
                datos_entrada["macro_creencia"] = macro_creencia

            if parametros_comportamiento:
                datos_entrada["parametros_comportamiento"] = parametros_comportamiento

            async for chunk in grafo.astream(
                input=datos_entrada,
                config=config,
                stream_mode="messages",
            ):
                if (
                    chunk[1]["langgraph_node"] == "nodo_conversacion"
                    and isinstance(chunk[0], AIMessageChunk)
                    and chunk[0].content
                ):
                    yield chunk[0].content

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"Error en streaming del grafo: {str(e)}") from e


# ─────────────────────────────────────────────────────────────────
# UTILIDAD
# ─────────────────────────────────────────────────────────────────

def _formatear_mensajes(
    mensajes: Union[str, list[dict[str, Any]]],
) -> list[Union[HumanMessage, AIMessage]]:
    if isinstance(mensajes, str):
        return [HumanMessage(content=mensajes)]

    if isinstance(mensajes, list):
        if not mensajes:
            return []

        if (
            isinstance(mensajes[0], dict)
            and "role" in mensajes[0]
            and "content" in mensajes[0]
        ):
            resultado = []
            for msg in mensajes:
                if msg["role"] == "user":
                    resultado.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    resultado.append(AIMessage(content=msg["content"]))
            return resultado

        return [HumanMessage(content=m) for m in mensajes]

    return []
