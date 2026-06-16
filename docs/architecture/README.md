# Arquitectura del sistema

Sistema de agentes conversacionales que simulan actores del conflicto armado colombiano usando el modelo BDI y RAG sobre el corpus de la CEV.

---

## Visión general

```
Frontend (index.html)
    ↓ WebSocket / REST
FastAPI (api.py)
    ↓ inicializa actor
crearActores.py → personalidad.py → macro_bdi.py
    ↓ estado inicial
LangGraph StateGraph (grafo.py)
    ├── nodo_macro_bdi   (1 vez por sesión)
    ├── nodo_micro_bdi   (cada turno)
    ├── nodo_conversacion → LLM (OpenAI gpt-4o-mini)
    ├── nodo_recuperador → MongoDB Atlas (búsqueda híbrida)
    └── nodo_resumir_contexto / nodo_resumir_conversacion
    ↓ checkpoint
MongoDB local (langgraph_checkpoints_eval)
```

---

## Componentes

### 1. Capa de personalidad BDI

**Macro-BDI** (`src/personalidades/macro_bdi.py`) — se ejecuta **una vez** al inicio de sesión:

```
Perfil del actor
    → LLM estima 8 dimensiones MBTI (0–1)
    → Fórmulas lineales derivan tipo Eneagrama (9 tipos)
    → Calcular 24 parámetros de comportamiento
    → Seleccionar patrones de sabiduría convencional por rol
    → LLM sintetiza macro_deseo (objetivo narrativo de la sesión)
```

**Micro-BDI** (`src/personalidades/micro_bdi.py`) — se ejecuta **cada turno**:

```
Mensaje del entrevistador
    → §1 Clasificar mensaje (acusación / pregunta / evasión / reconocimiento)
    → §2 Actualizar creencias (credibilidad, nivel_tension, conteo_negaciones)
    → §3 Elegir etapa discursiva:
          apertura_narrativa | defensa_posicion | negociacion_verdad
          | confrontacion | cierre_simbolico
    → §4 Generar micro_intencion (instrucciones específicas para el LLM)
```

**Los 24 parámetros de comportamiento** se distribuyen en 5 dimensiones:

| Dimensión | Parámetros |
|-----------|-----------|
| Sesgo declarativo | consistencia_logica, especificidad_detalle, profundidad_intuitiva, claridad_concision |
| Confianza | prueba_social, honestidad, consistencia |
| Afinidad | amabilidad, resonancia_emocional, expresion_atractiva |
| Deseo | autorrealizacion, aprobacion_social, estabilidad, amor_intimidad, libertad, aventura, relaciones |
| Comportamiento | conducta_evasiva, conducta_agresiva, adaptabilidad, introversion, extroversion, empatia, asertividad |

---

### 2. Grafo LangGraph

**Archivo:** `src/servicio_conversacion/workflow/grafo.py`

```
START
  → nodo_macro_bdi      (genera macro_deseo — solo si está vacío)
  → nodo_micro_bdi      (actualiza creencias y elige etapa discursiva)
  → nodo_conversacion   (LLM genera respuesta con tool_choice)
        ↓ si llamó RAG
        → nodo_recuperador
        → nodo_resumir_contexto
        → nodo_conversacion   (segunda llamada — genera respuesta final)
        ↓ si no llamó RAG
  → nodo_conector
        ↓ si > 30 mensajes
        → nodo_resumir_conversacion
  → END
```

**Estado compartido** (`EstadoActor` en `src/servicio_conversacion/workflow/estado.py`):

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `messages` | list | Historial de mensajes LangChain |
| `id_actor` | str | Rol del actor (`victima` / `victimario` / `tercero`) |
| `macro_deseo` | str | Objetivo narrativo de la sesión |
| `micro_intencion` | str | Instrucciones tácticas del turno actual |
| `nivel_tension` | str | Tensión acumulada (0.0–1.0) |
| `actor_contexto` | str | Chunks RAG recuperados |
| `parametros_comportamiento` | str | Los 24 parámetros en texto |

**Persistencia:** `MongoDBSaver` con colecciones `langgraph_checkpoints` y `langgraph_writes` en MongoDB local.

---

### 3. RAG híbrido

**Archivos:** `src/rag/retrievers.py`, `src/rag/embeddings.py`

- **Vector store:** MongoDB Atlas (`ChunksProyecto.Chunks`)
- **Modelo de embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace, local)
- **Búsqueda híbrida:** vector search + full-text search con `hybrid_search_index`
- **Filtro por rol:** los chunks se filtran para que coincidan con el rol del actor activo
- **Compresión:** `nodo_resumir_contexto` comprime los chunks si superan el límite de contexto

---

### 4. Modelos utilizados

| Modelo | Proveedor | Uso |
|--------|-----------|-----|
| `gpt-4o-mini` | OpenAI | Conversación, BDI, resúmenes |
| `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace | Embeddings RAG y grounding_score |

---

### 5. Módulo de evaluación

**Directorio:** `src/evaluacion/`

Evalúa 4 condiciones experimentales (ablación) con 276 trazas totales:

| Condición | BDI | RAG |
|-----------|-----|-----|
| `baseline` | ✗ | ✗ |
| `rag_only` | ✗ | ✓ |
| `bdi_only` | ✓ | ✗ |
| `bdi_rag` | ✓ | ✓ |

Ver [`docs/evaluacion_tecnica.md`](../evaluacion_tecnica.md) para el detalle completo del pipeline de evaluación.

---

## Decisiones de diseño

- **uv** como gestor de paquetes y entornos virtuales (más rápido que pip/poetry).
- **LangGraph** para el grafo de agentes porque permite persistencia de estado y streaming nativo.
- **MongoDB** tanto para checkpoints (local) como para el vector store (Atlas), centralizando la infraestructura de datos.
- **Evaluación ciega**: los jueces LLM (GPT-5, Gemini, Qwen) reciben las respuestas sin saber la condición experimental para evitar sesgo.
