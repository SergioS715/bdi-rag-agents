# EntrevistasCEV — Agentes Conversacionales con Personalidad BDI

### Trabajo de Grado · Pontificia Universidad Javeriana · Sergio Sosa

Sistema de agentes conversacionales que simulan actores del conflicto armado colombiano usando el modelo BDI (Beliefs-Desires-Intentions) y recuperación de testimonios reales del corpus de la Comisión para el Esclarecimiento de la Verdad (CEV).

---

## Tabla de Contenidos

1. [¿Qué hace el sistema?](#1-qué-hace-el-sistema)
2. [Arquitectura general](#2-arquitectura-general)
3. [Estructura del proyecto](#3-estructura-del-proyecto)
4. [Personalidad BDI](#4-personalidad-bdi)
5. [Flujo del grafo LangGraph](#5-flujo-del-grafo-langgraph)
6. [RAG híbrido](#6-rag-híbrido)
7. [Módulo de evaluación (detalle completo)](#7-módulo-de-evaluación)
8. [Instalación y ejecución](#8-instalación-y-ejecución)
9. [Configuración (.env)](#9-configuración-env)
10. [API endpoints](#10-api-endpoints)
11. Enlace a base de datos vectorial.  

---

## 1. ¿Qué hace el sistema?

Permite conversar con tres actores del conflicto colombiano, cada uno con personalidad única generada automáticamente:

| Actor                | Rol                                 |
| -------------------- | ----------------------------------- |
| **Víctima**   | Persona que sufrió daños directos |
| **Victimario** | Persona responsable de los hechos   |
| **Tercero**    | Observador o mediador del conflicto |

Cada actor tiene: rasgos MBTI estimados por LLM → tipo Eneagrama derivado → 24 parámetros de comportamiento → objetivo narrativo persistente (Macro-BDI) → decisiones tácticas por turno (Micro-BDI). Las respuestas se enriquecen con testimonios reales del corpus CEV vía RAG híbrido.

---

## 2. Arquitectura general

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
    ├── nodo_rag → MongoDB Atlas (búsqueda híbrida)
    └── nodo_resumir_contexto / nodo_resumir_conversacion
    ↓ checkpoint
MongoDB local (langgraph_checkpoints_eval)
```

**Modelos usados:**

| Modelo                                    | Proveedor   | Uso                              |
| ----------------------------------------- | ----------- | -------------------------------- |
| `gpt-4o-mini`                           | OpenAI      | Conversación, BDI, resúmenes   |
| `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace | Embeddings RAG y grounding_score |

---

## 3. Estructura del proyecto

```
trabajoGradoSergioSosa/
├── config.py                          # Lee variables del .env
├── pyproject.toml / uv.lock           # Dependencias (uv)
├── Makefile                           # Comandos: make run | make eval-full | make install
├── .env                               # Credenciales (NO subir al repositorio)
├── .env.example                       # Plantilla con variables requeridas
├── plantilla_anotacion.xlsx           # Scores de jueces ciegos (GPT-5, Gemini, Qwen)
│
├── conf/
│   └── mapeo_condiciones.json         # Mapeo prompt→condición para evaluación ciega
│
├── docs/
│   ├── Entrega Trabajo de grado *.pdf # Documento de tesis completo
│   └── Graficas Iniciales.pbix        # Dashboard Power BI
│
├── jupyter/
│   ├── notebooks/
│   │   ├── analisisdescriptivo.ipynb  # Análisis de n-gramas del corpus (UNA VEZ)
│   │   ├── analisis_estadistico.ipynb # Tests estadísticos sobre detalle_completo.csv
│   │   ├── analisis_jueces_ciegos.ipynb # Friedman + Wilcoxon (GPT-5, Gemini, Qwen)
│   │   └── registrar_evaluacion.ipynb # Registro manual de scores en Excel
│   └── datasets/
│       ├── datos_cev/                 # Corpus CEV: 200+ PDFs de testimonios
│       └── lexicons/                  # Léxicos temáticos (conflicto, actores, afectivo)
│
├── scripts/
│   └── registrar_evaluacion.py        # Registro de scores por consola
│
└── src/                               # Código fuente principal
    ├── app/
    │   ├── api.py                     # FastAPI (REST + WebSocket streaming)
    │   └── index.html                 # Interfaz web
    │
    ├── personalidades/
    │   ├── personalidad.py            # MBTI → Eneagrama → 24 parámetros
    │   ├── macro_bdi.py               # Objetivo narrativo (una vez por sesión)
    │   ├── micro_bdi.py               # Decisiones tácticas por turno
    │   ├── crearActores.py            # Fábrica de actores
    │   ├── prompts.py                 # Plantillas de prompts
    │   └── sabiduria_convencional.py  # Patrones discursivos por rol
    │
    ├── servicio_conversacion/
    │   ├── generar_respuesta.py       # Invoca el grafo (streaming y completo)
    │   ├── reiniciar_conversacion.py  # Limpia memoria del grafo
    │   └── workflow/
    │       ├── estado.py              # EstadoActor (TypedDict LangGraph)
    │       ├── grafo.py               # Construcción del StateGraph
    │       ├── nodos.py               # Lógica de cada nodo
    │       ├── chains.py              # Cadenas LangChain
    │       ├── aristas.py             # Condiciones de transición
    │       └── tools.py               # Herramienta RAG (recuperación)
    │
    ├── rag/
    │   ├── embeddings.py              # Modelo de embeddings HuggingFace
    │   └── retrievers.py              # Recuperador híbrido (vector + full-text)
    │
    └── evaluacion/                    # ← VER SECCIÓN 7 (detalle completo)
        │
        │  ── Pipeline principal (GPT-4o-mini) ──────────────────────────────────
        ├── correr_evaluacion_completa.py   # Orquestador del pipeline completo
        ├── generador_trazas.py             # Paso 1: genera trazas en MongoDB
        ├── evaluador.py                    # Paso 2: LLM-as-judge (GPT-4o-mini)
        ├── metricas_lexicas.py             # Paso 3: métricas léxicas sin LLM
        ├── reporte_evaluacion.py           # Paso 4: tablas CSV + Excel
        ├── ablacion_contrafactual.py       # Paso 5: deltas vs baseline
        ├── banco_pruebas.py                # 69 preguntas por actor (3 fases CEV)
        │
        │  ── Evaluación con modelos locales (para promedios) ───────────────────
        ├── correr_evaluacion_local.py      # Orquesta evaluación con phi4-mini/mistral
        ├── evaluador_local.py              # Evaluador usando Ollama (modelos locales)
        ├── generar_promedios.py            # Promedio global entre modelos locales
        ├── generar_promedios_condicion_rol.py  # Promedios por condición × rol
        ├── exportar_resultados_jueces.py   # Exporta plantilla_anotacion → CSVs
        ├── reporte_evaluacion_local.py     # Reporte para resultados de modelos locales
        ├── comparar_vocabulario.py         # Diff vocabulario baseline vs RAG
        ├── analizar_factual_grounding.py   # Diagnóstico de factual_grounding en Mongo
        │
        │  ── Resultados ─────────────────────────────────────────────────────────
        ├── resultados/                     # Pipeline GPT-4o-mini + corpus_ngrams_*.csv
        ├── resultados_gemma4/              # Evaluación con Gemma4 (para promedios)
        ├── resultados_llama3_2/            # Evaluación con Llama3.2 (para promedios)
        ├── resultados_mistral_7b/          # Evaluación con Mistral 7B (para promedios)
        ├── resultados_phi4_mini/           # Evaluación con Phi4-mini (para promedios)
        └── promedios/                      # Promedios y análisis descriptivos
            ├── analisis_descriptivo/       # Figuras y tablas de jueces ciegos
            └── resultados_jueces_llm/      # CSVs exportados desde plantilla_anotacion
```

---

## 4. Personalidad BDI

### Macro-BDI (se ejecuta UNA VEZ al inicio de sesión)

```
Perfil del actor
    → LLM estima 8 dimensiones MBTI (0–1)
    → Fórmulas lineales derivan tipo Eneagrama (9 tipos)
    → Calcular 24 parámetros de comportamiento
    → Seleccionar patrones de sabiduría convencional por rol
    → LLM sintetiza macro_deseo (objetivo narrativo de la sesión)
```

Ejemplo de `macro_deseo`:

> *"Quiero que reconozcan el daño que nos hicieron sin pedirme que olvide o que perdone antes de tiempo."*

### Micro-BDI (se ejecuta cada turno)

```
Mensaje del entrevistador
    → §1 Clasificar mensaje (acusación / pregunta / evasión / reconocimiento)
    → §2 Actualizar creencias (credibilidad, nivel_tension, conteo_negaciones)
    → §3 Elegir etapa discursiva:
          apertura_narrativa | defensa_posicion | negociacion_verdad
          | confrontacion | cierre_simbolico
    → §4 Generar micro_intencion (instrucciones específicas para el LLM)
```

### Los 24 parámetros de comportamiento

| Dimensión        | Parámetros                                                                                          |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| Sesgo declarativo | consistencia_logica, especificidad_detalle, profundidad_intuitiva, claridad_concision                |
| Confianza         | prueba_social, honestidad, consistencia                                                              |
| Afinidad          | amabilidad, resonancia_emocional, expresion_atractiva                                                |
| Deseo             | autorrealizacion, aprobacion_social, estabilidad, amor_intimidad, libertad, aventura, relaciones     |
| Comportamiento    | conducta_evasiva, conducta_agresiva, adaptabilidad, introversion, extroversion, empatia, asertividad |

---

## 5. Flujo del grafo LangGraph

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

Estado compartido (`EstadoActor`): `messages`, `id_actor`, `actor_nombre`, `actor_perspectiva`, `actor_estilo`, `resumen_conversacion`, `macro_deseo`, `macro_creencia`, `parametros_comportamiento`, `micro_intencion`, `micro_deseo`, `nivel_tension`, `conteo_negaciones`, `reconocimiento_dado`, `actor_contexto` (chunks RAG).

Persistencia: `MongoDBSaver` con colecciones `langgraph_checkpoints` y `langgraph_writes` en MongoDB local.

---

## 6. RAG híbrido

- **Vector store**: MongoDB Atlas (`ChunksProyecto.Chunks`)
- **Modelo de embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace)
- **Búsqueda híbrida**: vector search + full-text search con `hybrid_search_index`
- **Filtro por rol**: los chunks recuperados se filtran para el rol del actor activo
- **Compresión**: `nodo_resumir_contexto` comprime los chunks si son muy largos

---

## 7. Módulo de evaluación (detalle completo)

> Esta sección documenta en detalle el módulo `ArchivosCode/Evaluacion/` para facilitar el trabajo futuro.

### 7.1 Diseño experimental

Se evalúan **4 condiciones experimentales** para medir el efecto causal de RAG y BDI:

| Código | Condición   | BDI | RAG |
| ------- | ------------ | --- | --- |
| A       | `baseline` | ✗  | ✗  |
| B       | `rag_only` | ✗  | ✓  |
| C       | `bdi_only` | ✓  | ✗  |
| D       | `bdi_rag`  | ✓  | ✓  |

Cada condición usa el mismo banco de **69 preguntas** por actor (`banco_pruebas.py`), distribuidas en:

- 7 cerradas/fácticas · 7 abiertas/relatos · 9 confrontativas (arco narrativo CEV en 3 fases)
- 3 actores × 23 preguntas × 4 condiciones = **276 trazas totales**

### 7.2 Pipeline completo

```bash
# Correr todo desde cero (borra datos previos)
python ArchivosCode/Evaluacion/correr_evaluacion_completa.py

# Opciones:
# --skip-borrar     conserva trazas y resultados existentes
# --desde N         retoma desde el paso N (1–5) sin borrar nada
# --sin-grafico     omite la generación de .png
```

**Pasos del pipeline:**

| Paso | Script                        | Duración est. | Descripción                                                                  |
| ---- | ----------------------------- | -------------- | ----------------------------------------------------------------------------- |
| 0    | (interno)                     | < 1s           | Limpia MongoDB (evaluacion_trazas, evaluacion_resultados, checkpoints)        |
| 1    | `generador_trazas.py`       | ~20 min        | Corre el grafo por las 4 condiciones y guarda trazas en MongoDB               |
| 2    | `evaluador.py`              | ~30 min        | LLM-as-a-judge (OpenAI GPT-4o-mini) sobre trazas + grounding_score embeddings |
| 3    | `metricas_lexicas.py`       | ~2 min         | Métricas léxicas desde corpus_ngrams (NO requiere re-correr el notebook)    |
| 4    | `reporte_evaluacion.py`     | ~1 min         | Tablas comparativas CSV + Excel + estadística                                |
| 5    | `ablacion_contrafactual.py` | ~1 min         | Deltas vs baseline + comparaciones directas entre condiciones                 |

Todo el output se guarda en `resultados/pipeline_YYYY-MM-DD_HH-MM.log`.

### 7.3 Colecciones MongoDB del módulo

| Colección                     | Llenada por             | Leída por                                               |
| ------------------------------ | ----------------------- | -------------------------------------------------------- |
| `evaluacion_trazas`          | `generador_trazas.py` | `evaluador.py`, `metricas_lexicas.py`                |
| `evaluacion_resultados`      | `evaluador.py`        | `reporte_evaluacion.py`, `ablacion_contrafactual.py` |
| `langgraph_checkpoints_eval` | `generador_trazas.py` | — (estado interno del grafo)                            |
| `langgraph_writes_eval`      | `generador_trazas.py` | —                                                       |

### 7.4 Estructura de una traza (`evaluacion_trazas`)

Cada documento representa un turno de conversación:

```python
{
  "condicion"               : "bdi_rag",      # baseline | rag_only | bdi_only | bdi_rag
  "actor_id"                : "victima",
  "turno"                   : 3,
  "tipologia"               : "confrontativa",
  "pregunta"                : "...",
  "respuesta"               : "...",
  "macro_deseo"             : "...",           # vacío en baseline/rag_only
  "micro_deseo"             : "...",           # etapa discursiva elegida
  "micro_intencion"         : "...",           # instrucciones del micro-BDI
  "actor_contexto"          : "...",           # chunks RAG recuperados (vacío en baseline/bdi_only)
  "parametros_comportamiento": "...",          # vacío en baseline/rag_only
  "nivel_tension"           : "0.7",
  "conteo_negaciones"       : "2",
  "reconocimiento_dado"     : "0.3",
  "retrieval_metadata"      : { "role_purity": 0.9, "n_docs_total": 5 },
  "verificacion_contexto"   : "..."            # chunks recuperados con la RESPUESTA como query
                                               # (para factual_grounding — todas las condiciones)
}
```

### 7.5 Métricas LLM-as-a-judge

Juez: **OpenAI GPT-4o-mini** (`evaluador.py`). Cada llamada devuelve JSON `{"score": 1–5}`.

**Principio de diseño**: la mayoría de los prompts son **dinámicos por rol**. Víctima, victimario y tercero tienen patrones narrativos radicalmente distintos, y evaluarlos con la misma rúbrica sería injusto. Los únicos prompts estáticos son `answer_relevance` y `subordinacion_macro`.

Las métricas se organizan en **tres tiers** según qué condiciones las tienen disponibles:

#### Tier 1 — Universales (las 4 condiciones → forman `persona_score`)

| Métrica                   | Evaluación                   | Qué mide y cómo varía por rol                                                                                                                                                                                                                                                                                                                                                |
| -------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `estabilidad_rol`        | Por conversación (1 llamada) | Coherencia de identidad discursiva a lo largo de toda la sesión.**Víctima**: narrativa de dolor/reconocimiento sostenida. **Victimario**: tensión natural entre contextualización y reconocimiento de responsabilidad. **Tercero**: posición de testigo estable aunque el discurso alterne entre descripción externa e indignación moral.              |
| `autenticidad_lexica`    | Por turno                     | ¿El lenguaje suena como alguien que vivió o presenció el conflicto?**Víctima**: oralidad campesina, eufemismos del trauma ("cuando eso", "nos tocó"). **Victimario**: registro de excombatiente en reflexión, mezcla de experiencia pasada y reconocimiento presente. **Tercero**: mezcla fluida de registro profesional y oralidad de testigo cercano. |
| `autenticidad_emocional` | Por turno                     | Tono afectivo apropiado al rol.**Víctima**: dolor implícito o explícito, sin frialdad analítica. **Victimario**: tensión entre justificación contextual y asunción de responsabilidad — ninguno de los dos polos resuelto del todo. **Tercero**: alternancia entre distancia de testigo e indignación moral (por diseño, no incoherencia).          |
| `goal_directedness`      | Por conversación (1 llamada) | Dirección narrativa acumulada observable.**Víctima**: progresión hacia reconocimiento del daño (cada turno suma evidencia). **Victimario**: alternancia defensiva-contextual coherente (no lineal, pero dirigida). **Tercero**: construcción de testimonio moral mediante alternancia testigo externo / actor moral interno.                             |
| `tactical_consistency`   | Por turno                     | Apropiadez de la estrategia al tipo de pregunta: CERRADA → factual/contextual, ABIERTA → narrativo/emocional, CONFRONTATIVA → defensivo/resistente. La alternancia justificada es coherencia, no incoherencia. Rúbrica diferente por rol: imprecisión traumática válida en víctima, contextualización defensiva en victimario, incomodidad del testigo en tercero.     |
| `factual_grounding`      | Por turno                     | ¿Los temas de la respuesta están respaldados por el corpus CEV y el respaldo proviene del mismo rol? (role purity). Evalúa dos dimensiones: grounding temático + pureza del rol. Usa `verificacion_contexto` — aplica a las 4 condiciones.                                                                                                                               |

`persona_score` = promedio(estabilidad_rol, autenticidad_lexica, autenticidad_emocional, goal_directedness, tactical_consistency, factual_grounding)

`answer_relevance` (1–5, estático): universal, evaluado por turno, fuera de `persona_score`. Considera el comportamiento esperado del rol: evasión táctica o trauma representado pueden ser respuestas relevantes para víctima o victimario.

#### Tier 2 — RAG (rag_only y bdi_rag → forman `rag_score`)

| Métrica             | Escala | Cálculo                                                                                                                          | Qué mide                                                                                                                                                                                                                                                               |
| -------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lexical_adoption` | 1–5   | LLM-judge, dinámico por rol                                                                                                      | ¿El agente habla como alguien que vivió el conflicto usando los chunks como insumo, no como guión? Jerarquía: léxico del conflicto presente → oralidad o tono situado → detalles creíbles. Penaliza copia literal y uso de vocabulario como lista desconectada. |
| `grounding_score`  | 0–1   | Similitud coseno `embedding(respuesta) ↔ embedding(actor_contexto)`, modelo `paraphrase-multilingual-MiniLM-L12-v2`, sin LLM | Captura paráfrasis y sinónimos que el overlap léxico no detecta. Complementa lexical_adoption con una señal objetiva.                                                                                                                                               |

`rag_score` = promedio(lexical_adoption / 5, grounding_score)  ← ambas en escala 0–1

#### Tier 3 — BDI (bdi_only y bdi_rag → forman `bdi_score`)

| Métrica                | Escala | Evaluación                              | Qué mide                                                                                                                                                                                                                                             |
| ----------------------- | ------ | ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `coherencia_tactica`  | 1–5   | Por turno, dinámico por rol             | ¿La respuesta refleja la etapa discursiva asignada (`micro_deseo`) con el tono apropiado? Cada etapa (apertura_narrativa, defensa_posicion, negociacion_verdad, confrontacion, cierre_simbolico) tiene significado y tono diferente para cada rol. |
| `subordinacion_macro` | 1–5   | Por conversación (1 llamada, estático) | ¿Las respuestas son coherentes con el `macro_deseo` global? No todas deben avanzarlo directamente; la variación natural sin contradicción es aceptable.                                                                                          |

`bdi_score` = promedio(coherencia_tactica, subordinacion_macro)

#### Legacy (guardada pero excluida del análisis)

| Métrica                    | Nota                                                                  |
| --------------------------- | --------------------------------------------------------------------- |
| `autenticidad_discursiva` | = promedio(autenticidad_lexica, autenticidad_emocional) — redundante |

### 7.6 Métricas léxicas (`metricas_lexicas.py`)

Calculadas sobre el texto sin LLM. Organizadas en tres grupos según qué condiciones las tienen disponibles:

#### Universales (las 4 condiciones)

| Métrica                            | Fórmula                                          | Qué mide                                                            |
| ----------------------------------- | ------------------------------------------------- | -------------------------------------------------------------------- |
| `conflict_vocab_density_response` | conteo_vocab_conflicto / tokens_totales_respuesta | Densidad del vocabulario del conflicto del rol en la respuesta       |
| `conflict_vocab_count`            | \|{términos únicos del rol en respuesta}\|      | Número de términos únicos del vocabulario del conflicto presentes |

Vocabulario fuente: `resultados/corpus_ngrams_{rol}.csv` — top 300 unigrams por `specificity_score` (penaliza español genérico), generados por `analisisdescriptivo.ipynb` (correr una sola vez; los 4 archivos ya existen).

#### Transferencia léxica del RAG (rag_only y bdi_rag)

| Métrica                   | Fórmula                                                                                                     | Qué mide                                                                                                                                                                      |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `conflict_vocab_overlap` | \|vocab_conflicto_chunks ∩ vocab_conflicto_respuesta\| / \|vocab_conflicto_chunks\|                         | Fracción del vocabulario de conflicto del chunk adoptada en la respuesta                                                                                                      |
| `ngram_overlap_weighted` | 0.5×(overlap unigrams) + 0.3×(overlap bigrams) + 0.2×(overlap trigrams)                                   | Overlap ponderado de n-gramas del corpus presentes en los chunks vs los adoptados en la respuesta. Unigrams pesan más por frecuencia; bigrams/trigrams aportan especificidad. |
| `oral_markers_adoption`  | \|muletillas_chunk ∩ muletillas_respuesta\| / \|muletillas_chunk\|                                          | % de marcadores discursivos orales del chunk adoptados en la respuesta (pues, entonces, nos tocó, uno, etc.)                                                                  |
| `vocabulary_overlap`     | \|vocab_contenido_chunk ∩ vocab_contenido_respuesta\| / \|vocab_contenido_chunk\|                           | Overlap general de vocabulario de contenido (excluye stopwords)                                                                                                                |
| `semantic_overlap`       | coseno(embedding(chunks[:500]), embedding(respuesta[:500])), modelo `distiluse-base-multilingual-cased-v2` | ¿El agente entiende el contenido aunque lo parafrasee? 0.85–1.0 = parafraseo correcto; 0.60–0.85 = entiende con pérdida; < 0.40 = alucinación.                            |

#### Calidad del retrieval (propiedades del chunk, no de la transferencia)

| Métrica                          | Qué mide                                                                    |
| --------------------------------- | ---------------------------------------------------------------------------- |
| `conflict_vocab_density_chunks` | Densidad del vocabulario del conflicto en los chunks recuperados             |
| `oral_markers_density_chunks`   | Densidad de marcadores de oralidad en los chunks                             |
| `lexical_diversity`             | vocab_único_contenido / tokens_contenido en actor_contexto                  |
| `role_purity`                   | % de chunks que coincidían con el rol pedido (desde `retrieval_metadata`) |
| `top_k_similarity`              | Promedio de scores dot-product del vector search                             |

**Archivos generados:**

- `metricas_lexicas_por_turno.csv` — datos granulares por turno
- `metricas_lexicas_resumen.csv` — medias por condición × rol
- `transferencia_lexica.png` — gráfico comparativo

### 7.7 Ablación contrafactual (`ablacion_contrafactual.py`)

**Deltas vs baseline** (efecto causal):

```
Δ = valor_condicion − valor_baseline
rag_only − baseline → efecto causal del RAG puro
bdi_only − baseline → efecto causal del BDI puro
bdi_rag  − baseline → efecto combinado
```

**Comparaciones directas** (efecto incremental):

```
bdi_rag − rag_only  → ¿cuánto agrega BDI cuando ya hay RAG?   (efecto_bdi_sobre_rag)
bdi_rag − bdi_only  → ¿cuánto agrega RAG cuando ya hay BDI?   (efecto_rag_sobre_bdi)
```

**Hipótesis evaluadas:**

1. RAG mejora `autenticidad_lexica` (delta rag_only vs baseline)
2. BDI mejora `autenticidad_emocional` (delta bdi_only vs baseline)
3. RAG transfiere vocabulario del conflicto (delta `conflict_vocab_overlap`)
4. BDI mejora `goal_directedness` (delta bdi_only vs baseline)

**Archivos generados:**

- `ablacion_delta_por_turno.csv` — granular (turno a turno)
- `ablacion_delta_resumen.csv` — medias por condicion × actor
- `ablacion_deltas_directos.csv` — comparaciones bdi_rag vs rag_only / bdi_only
- `ablacion_interpretacion.txt` — texto interpretativo para la tesis
- `ablacion_delta.png` — gráfico de barras

### 7.8 Análisis estadístico (`reporte_evaluacion.py`)

- **Kruskal-Wallis** sobre las 4 condiciones (test de diferencia global)
- **Mann-Whitney U** para cada par de condiciones (6 pares posibles)
- **Corrección de Bonferroni** sobre los p-values
- **Cliff's delta** (tamaño del efecto, escala: negligible / small / medium / large)
- **Bootstrap CI** para las medias (95% IC, 1000 iteraciones)

**Archivos generados:**

- `metricas_por_condicion.csv`
- `metricas_por_rol.csv` (victima / victimario / tercero)
- `metricas_por_tipologia.csv` (cerrada / abierta / confrontativa)
- `scores_por_condicion_rol.csv` (persona_score / rag_score / bdi_score)
- `analisis_estadistico.csv`
- `detalle_completo.csv` (todos los registros individuales)
- `reporte_evaluacion.xlsx` (todas las hojas en un Excel)

### 7.9 Estructura de un resultado (`evaluacion_resultados`)

```python
{
  "condicion"          : "bdi_rag",
  "actor_id"           : "victima",
  "turno"              : 3,
  "tipologia_pregunta" : "confrontativa",
  "pregunta"           : "...",
  "respuesta"          : "...",
  "scores": {
    # Universales (todas las condiciones) — forman persona_score
    "estabilidad_rol"        : 4.0,   # por conversación, repetido en cada turno
    "autenticidad_lexica"    : 3.0,
    "autenticidad_emocional" : 4.0,
    "goal_directedness"      : 4.0,   # por conversación, repetido en cada turno
    "tactical_consistency"   : 3.0,
    "factual_grounding"      : 4.0,
    "answer_relevance"       : 5.0,   # universal, fuera de persona_score
    # RAG (rag_only, bdi_rag)
    "lexical_adoption"       : 4.0,   # LLM-judge, escala 1–5
    "grounding_score"        : 0.73,  # similitud coseno embeddings, 0–1
    # BDI (bdi_only, bdi_rag)
    "coherencia_tactica"     : 3.0,
    "subordinacion_macro"    : 4.0,   # por conversación, repetido en cada turno
    # Legacy / derivada — no usar en análisis principal
    "autenticidad_discursiva": 3.5,   # promedio(autenticidad_lexica, autenticidad_emocional)
  },
  "persona_score" : 3.67,   # avg(estabilidad_rol, autenticidad_lexica, autenticidad_emocional,
                             #     goal_directedness, tactical_consistency, factual_grounding)
  "rag_score"     : 0.73,   # avg(lexical_adoption/5, grounding_score)
  "bdi_score"     : 3.50,   # avg(coherencia_tactica, subordinacion_macro)
}
```

---

## 8. Instalación y ejecución

```bash
# Instalar dependencias
uv sync

# Iniciar el sistema conversacional
uv run python ArchivosCode/App/api.py
# → abre http://localhost:8000

# Correr evaluación completa
python ArchivosCode/Evaluacion/correr_evaluacion_completa.py

# Retomar evaluación desde un paso intermedio (sin borrar datos)
python ArchivosCode/Evaluacion/correr_evaluacion_completa.py --desde 2
```

---

## 9. Configuración (.env)

```env
OPENAI_API_KEY=sk-proj-...
ATLAS_URI=mongodb+srv://usuario:password@cluster.mongodb.net/
MONGO_URI=mongodb://localhost:27017/
```

---

## 10. API endpoints

| Método  | Ruta                          | Descripción                         |
| -------- | ----------------------------- | ------------------------------------ |
| `GET`  | `/`                         | Interfaz web                         |
| `GET`  | `/actores`                  | Lista roles disponibles              |
| `POST` | `/iniciar-actor`            | Inicializa personalidad BDI          |
| `POST` | `/chat`                     | Respuesta completa (sin streaming)   |
| `WS`   | `/ws/chat`                  | Respuesta en streaming token a token |
| `POST` | `/reset-memoria`            | Limpia todo el historial             |
| `POST` | `/reset-memoria/{id_actor}` | Limpia historial de un actor         |

---

## 11 Enlace a base de datos vectorial.

BASE DE DATOS VECTORIAL (RAG)
------------------------------
La base de datos vectorial utilizada para el componente RAG se encuentra exportada
en formato JSON y está disponible en el siguiente enlace de Google Drive para su uso se debe cargar en la instancia de mongoDB atlas propio :

    https://drive.google.com/file/d/1Lgbkt-_TuzJc98SiLzj16lmHEsNU43Yj/view?usp=sharing
