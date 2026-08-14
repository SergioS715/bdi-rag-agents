import logging
import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AsyncOpenAI

# ── Rutas al resto del proyecto ───────────────────────────────────
APP_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.extend([
    ROOT_DIR,
    BASE_DIR,
    os.path.join(BASE_DIR, "personalidades"),
    os.path.join(BASE_DIR, "servicio_conversacion"),
    os.path.join(BASE_DIR, "servicio_conversacion", "workflow"),
])

from crearActores import ActoresInvolucrados
from actor import Actor
from generar_respuesta import obtener_respuesta, obtener_respuesta_streaming
from reiniciar_conversacion import reiniciar_estado_conversacion
import config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ─────────────────────────────────────────────
# GUARDRAIL — filtra mensajes fuera de tema
# ─────────────────────────────────────────────

_guardrail_client = AsyncOpenAI(
    api_key  = config.DEEPSEEK_API_KEY,
    base_url = config.DEEPSEEK_BASE_URL,
)

_GUARDRAIL_SYSTEM = """Eres un clasificador de mensajes para una simulación académica de entrevistas del conflicto armado colombiano (CEV).

RELEVANTE: el mensaje trata sobre el conflicto armado, violencia, desplazamiento, masacres, grupos armados, víctimas, familia, emociones, memoria, historia personal, testimonios, perdón, reconciliación, Colombia, paz, guerra, o cualquier cosa relacionada con la experiencia humana del conflicto.
BLOQUEADO: el mensaje pide ayuda con programación, código, matemáticas, recetas, idiomas, tecnología, entretenimiento, o cualquier tema sin relación con el conflicto armado colombiano.

Responde ÚNICAMENTE con una palabra: RELEVANTE o BLOQUEADO"""

_DEFLEXION_POR_ROL = {
    "victima"   : "Eso que me preguntás no tiene que ver con lo que yo viví. Si querés saber algo de mi historia, de lo que pasé, preguntame eso.",
    "victimario": "Eso no me compete. Estoy aquí para hablar de lo que pasó, de la guerra. No de otras cosas.",
    "tercero"   : "Eso está por fuera de lo que yo puedo contarles. Pregúntenme sobre lo que presencié, sobre lo que vivió la comunidad.",
}
_DEFLEXION_DEFAULT = "Eso no tiene que ver con lo que estamos hablando aquí. Pregúntame sobre lo que viví, sobre el conflicto."


async def _guardrail_es_permitido(mensaje: str) -> bool:
    """Retorna True si el mensaje es relevante para la simulación."""
    try:
        response = await _guardrail_client.chat.completions.create(
            model       = config.DEEPSEEK_LLM_MODEL,
            messages    = [
                {"role": "system", "content": _GUARDRAIL_SYSTEM},
                {"role": "user",   "content": mensaje},
            ],
            temperature = 0.0,
            max_tokens  = 5,
        )
        clasificacion = response.choices[0].message.content.strip().upper()
        permitido = "RELEVANTE" in clasificacion
        if not permitido:
            log.info(f"[guardrail] Mensaje bloqueado: {mensaje[:80]!r}")
        return permitido
    except Exception as e:
        log.warning(f"[guardrail] Error en clasificación, permitiendo por defecto: {e}")
        return True


# ─────────────────────────────────────────────
# CACHÉ DE ACTORES BDI
# ─────────────────────────────────────────────
# Guarda el actor con personalidad BDI por rol.
# Se llena cuando el usuario llama a POST /iniciar-actor.
# Se reutiliza en todos los mensajes posteriores de esa sesión.
# Evita recalcular MBTI + macro_deseo en cada request.
#
# Estructura: { "victima": Actor, "victimario": Actor, "tercero": Actor }
_cache_actores_bdi: dict[str, Actor] = {}


def _obtener_actor(id_actor: str, modo_bdi: bool = False) -> Actor:
    """
    Retorna el actor para el rol dado.

    Si modo_bdi=True y ya está en caché, retorna el cacheado.
    Si modo_bdi=True y no está en caché, lo construye y cachea.
    Si modo_bdi=False, retorna actor sin personalidad BDI (comportamiento original).
    """
    if not modo_bdi:
        return ActoresInvolucrados.obtener_actor(id_actor)

    if id_actor not in _cache_actores_bdi:
        log.info(f"[BDI] Construyendo personalidad para '{id_actor}'...")
        _cache_actores_bdi[id_actor] = ActoresInvolucrados.obtener_actor_bdi(id_actor)
        log.info(f"[BDI] Actor '{id_actor}' listo con personalidad BDI.")

    return _cache_actores_bdi[id_actor]


# ─────────────────────────────────────────────
# ARRANQUE / APAGADO
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 API EntrevistasCEV iniciada")
    yield
    log.info("🛑 API EntrevistasCEV detenida")


app = FastAPI(
    title="EntrevistasCEV — Agente Conversacional",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="."), name="static")


# ─────────────────────────────────────────────
# MODELOS DE DATOS
# ─────────────────────────────────────────────

CONDICIONES_CON_BDI = {"bdi_only", "bdi_rag"}


class MensajeConversacion(BaseModel):
    mensaje   : str
    id_actor  : str   # "victima" | "victimario" | "tercero"
    condicion : str = "bdi_rag"  # "baseline" | "rag_only" | "bdi_only" | "bdi_rag"


class SolicitudIniciarActor(BaseModel):
    id_actor  : str   # "victima" | "victimario" | "tercero"
    condicion : str = "bdi_rag"


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/")
async def raiz():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    return FileResponse(ruta)


@app.get("/actores")
async def listar_actores():
    """Retorna los roles disponibles para seleccionar en la UI."""
    return {
        "actores": ActoresInvolucrados.obtener_actores_disponibles()
    }


@app.post("/iniciar-actor")
async def iniciar_actor(req: SolicitudIniciarActor):
    """
    Inicializa el actor con personalidad BDI completa.

    Llama a este endpoint UNA SOLA VEZ cuando el usuario selecciona
    un rol en la UI, antes de empezar a chatear.

    El proceso tarda 3-8 segundos porque hace dos llamadas al LLM:
      1. Estimar MBTI desde el perfil del actor
      2. Generar macro_deseo desde MBTI + banco de sabiduría

    Una vez completado, el actor queda en caché y los mensajes
    posteriores no tienen overhead adicional.
    """
    try:
        usa_bdi = req.condicion in CONDICIONES_CON_BDI
        actor = _obtener_actor(req.id_actor, modo_bdi=usa_bdi)

        enneagram = ""
        if actor.personalidad:
            enneagram = actor.personalidad.enneagram.tipo_dominante()

        return {
            "id_actor"    : actor.id,
            "actor_nombre": actor.nombre,
            "enneagram"   : enneagram,
            "condicion"   : req.condicion,
            "listo"       : True,
            "mensaje"     : f"Actor '{actor.id}' inicializado (condición: {req.condicion}).",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.error(f"[iniciar-actor] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(msg: MensajeConversacion):
    """
    Respuesta completa (sin streaming).

    Si el actor ya fue inicializado con BDI (via /iniciar-actor),
    usa el actor cacheado con personalidad. Si no, usa el actor
    sin personalidad (comportamiento original).
    """
    try:
        if not await _guardrail_es_permitido(msg.mensaje):
            deflexion = _DEFLEXION_POR_ROL.get(msg.id_actor, _DEFLEXION_DEFAULT)
            return {"respuesta": deflexion}

        condicion = msg.condicion
        usa_bdi   = condicion in CONDICIONES_CON_BDI
        modo_bdi  = usa_bdi and msg.id_actor in _cache_actores_bdi
        actor     = _obtener_actor(msg.id_actor, modo_bdi=modo_bdi)

        macro_deseo               = ""
        macro_creencia            = ""
        parametros_comportamiento = ""

        if modo_bdi and actor.personalidad:
            parametros_comportamiento = actor.personalidad.resumen_para_prompt()

        respuesta, _ = await obtener_respuesta(
            mensajes                  = msg.mensaje,
            id_actor                  = actor.id,
            actor_nombre              = actor.nombre,
            actor_perspectiva         = actor.perspectiva,
            actor_estilo              = actor.estilo,
            actor_contexto            = "",
            condicion                 = condicion,
            macro_deseo               = macro_deseo,
            macro_creencia            = macro_creencia,
            parametros_comportamiento = parametros_comportamiento,
        )
        return {"respuesta": respuesta}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket para respuesta token a token (streaming).

    Protocolo:
        Cliente envía : { "mensaje": "...", "id_actor": "victima" }
        Servidor envía: { "streaming": true }
                        { "chunk": "token..." }  ← repetido
                        { "respuesta": "...", "streaming": false }
    """
    await websocket.accept()
    log.info("WebSocket conectado")

    try:
        while True:
            datos = await websocket.receive_json()

            if "mensaje" not in datos or "id_actor" not in datos:
                await websocket.send_json({
                    "error": "Campos requeridos: 'mensaje' e 'id_actor'"
                })
                continue

            try:
                id_actor  = datos["id_actor"]
                condicion = datos.get("condicion", "bdi_rag")
                usa_bdi   = condicion in CONDICIONES_CON_BDI
                modo_bdi  = usa_bdi and id_actor in _cache_actores_bdi
                actor     = _obtener_actor(id_actor, modo_bdi=modo_bdi)
            except ValueError as e:
                await websocket.send_json({"error": str(e)})
                continue

            parametros_comportamiento = ""
            macro_deseo               = ""
            macro_creencia            = ""

            if modo_bdi and actor.personalidad:
                parametros_comportamiento = actor.personalidad.resumen_para_prompt()

            try:
                if not await _guardrail_es_permitido(datos["mensaje"]):
                    deflexion = _DEFLEXION_POR_ROL.get(id_actor, _DEFLEXION_DEFAULT)
                    await websocket.send_json({"streaming": True})
                    await websocket.send_json({"chunk": deflexion})
                    await websocket.send_json({"respuesta": deflexion, "streaming": False})
                    continue

                await websocket.send_json({"streaming": True})

                respuesta_completa = ""
                async for chunk in obtener_respuesta_streaming(
                    mensajes                  = datos["mensaje"],
                    id_actor                  = actor.id,
                    actor_nombre              = actor.nombre,
                    actor_perspectiva         = actor.perspectiva,
                    actor_estilo              = actor.estilo,
                    actor_contexto            = "",
                    condicion                 = condicion,
                    macro_deseo               = macro_deseo,
                    macro_creencia            = macro_creencia,
                    parametros_comportamiento = parametros_comportamiento,
                ):
                    respuesta_completa += chunk
                    await websocket.send_json({"chunk": chunk})

                await websocket.send_json({
                    "respuesta" : respuesta_completa,
                    "streaming" : False,
                })

            except Exception as e:
                log.error(f"Error generando respuesta: {e}")
                await websocket.send_json({"error": str(e)})

    except WebSocketDisconnect:
        log.info("WebSocket desconectado")


@app.post("/reset-memoria")
async def reset_total():
    """Borra toda la memoria de conversaciones."""
    try:
        _cache_actores_bdi.clear()
        return await reiniciar_estado_conversacion()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset-memoria/{id_actor}")
async def reset_actor(id_actor: str):
    """Borra la memoria de un actor específico."""
    try:
        _cache_actores_bdi.pop(id_actor, None)
        return await reiniciar_estado_conversacion(id_actor=id_actor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# ARRANQUE DIRECTO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
