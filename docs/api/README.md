# Documentación de la API

La API del sistema está implementada en [src/app/api.py](../../src/app/api.py) con **FastAPI**. Expone endpoints REST y WebSocket para interactuar con los agentes conversacionales.

---

## Base URL

```
http://localhost:8000
```

---

## Endpoints

### `GET /`

Sirve la interfaz web (`src/app/index.html`).

**Respuesta:** HTML de la interfaz.

---

### `GET /actores`

Lista los roles de actores disponibles.

**Respuesta:**
```json
["victima", "victimario", "tercero"]
```

---

### `POST /iniciar-actor`

Inicializa la personalidad BDI de un actor. Ejecuta el pipeline Macro-BDI: MBTI → Eneagrama → 24 parámetros → `macro_deseo`.

**Body:**
```json
{
  "id_actor": "victima",
  "nombre": "María",
  "perspectiva": "Mujer campesina del Cauca desplazada en 2002.",
  "estilo": "Habla con oralidad, evita términos técnicos."
}
```

**Respuesta:**
```json
{
  "status": "ok",
  "actor_id": "victima",
  "macro_deseo": "Quiero que reconozcan el daño que nos hicieron..."
}
```

---

### `POST /chat`

Genera una respuesta completa (sin streaming) a un mensaje del entrevistador.

**Body:**
```json
{
  "id_actor": "victima",
  "mensaje": "¿Cómo fue el momento en que tuvieron que salir de su tierra?"
}
```

**Respuesta:**
```json
{
  "respuesta": "Eso fue muy duro... nos tocó salir de noche, sin poder llevarnos nada.",
  "micro_deseo": "apertura_narrativa",
  "nivel_tension": 0.4
}
```

---

### `WS /ws/chat`

Respuesta en streaming token a token via WebSocket.

**Mensaje enviado (JSON):**
```json
{
  "id_actor": "victima",
  "mensaje": "¿Cómo fue el momento en que tuvieron que salir de su tierra?"
}
```

**Mensajes recibidos:** tokens de texto a medida que se generan, finalizando con `{"done": true}`.

---

### `POST /reset-memoria`

Limpia el historial completo de todos los actores.

**Respuesta:**
```json
{"status": "ok", "mensaje": "Memoria de todos los actores reiniciada."}
```

---

### `POST /reset-memoria/{id_actor}`

Limpia el historial de un actor específico.

**Parámetro de ruta:** `id_actor` — `victima`, `victimario` o `tercero`.

**Respuesta:**
```json
{"status": "ok", "mensaje": "Memoria del actor 'victima' reiniciada."}
```

---

## Códigos de error comunes

| Código | Causa |
|--------|-------|
| `422` | Body JSON inválido o campos requeridos faltantes |
| `500` | Error interno (falla de LLM, MongoDB inaccesible) |

---

## Variables de entorno requeridas

Ver [`.env.example`](../../.env.example):

| Variable | Descripción |
|----------|-------------|
| `OPENAI_API_KEY` | Clave de API de OpenAI (GPT-4o-mini) |
| `ATLAS_URI` | URI de conexión a MongoDB Atlas (vector store RAG) |
| `MONGO_URI` | URI de MongoDB local (checkpoints LangGraph) |
