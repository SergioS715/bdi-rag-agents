# Changelog

## [Unreleased]

### Reorganización del proyecto (2026-05-06)
- Estructura reorganizada siguiendo boilerplate FIS: `src/`, `conf/`, `docs/`, `jupyter/`, `scripts/`
- Módulos renombrados a snake_case: `Personalidades` → `personalidades`, `ServicioConversacion` → `servicio_conversacion`, etc.
- Notebooks consolidados en `jupyter/notebooks/`
- Corpus y datasets en `jupyter/datasets/`
- Documento de tesis movido a `docs/`
- `Makefile` agregado con comandos para servidor, evaluación y linting

### Limpieza y seguridad (2026-05-05)
- Credenciales hardcodeadas en `config.py` eliminadas — todo via `os.getenv()`
- `.gitignore` creado (excluye `.env`, `.venv/`, `__pycache__/`, etc.)
- Scripts de análisis más importantes convertidos a notebooks Jupyter
- Directorios obsoletos eliminados: `preprocesamiento/`, `memoria/`, `AgentesConversacionales/`
- Resultados de análisis estadísticos de evaluadores locales eliminados (Gemma/Llama/Mistral/Phi4 solo para promedios)

## [0.1.0] — Versión inicial del sistema

### Sistema conversacional
- Agentes BDI con personalidad MBTI → Enneagram → 24 parámetros de comportamiento
- Macro-BDI (goals de sesión) y Micro-BDI (tácticas por turno) vía LangGraph StateGraph
- RAG híbrido (vector + full-text) sobre corpus CEV (testimonios del conflicto colombiano)
- API FastAPI con WebSocket para streaming token a token
- Checkpointing de conversación en MongoDB

### Sistema de evaluación
- Pipeline 6 pasos: trazas → LLM-as-judge → métricas léxicas → reporte → ablación
- Estudio de ablación 4 condiciones: baseline / rag_only / bdi_only / bdi_rag
- Evaluación ciega con GPT-5, Gemini 3 Flash, Qwen 3.6 Plus (plantilla_anotacion.xlsx)
- Tests estadísticos: Friedman + Wilcoxon + Bonferroni sobre jueces ciegos
