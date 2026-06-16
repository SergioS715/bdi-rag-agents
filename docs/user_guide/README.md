# Guía de usuario

Esta guía explica cómo instalar, configurar y usar el sistema de agentes conversacionales EntrevistasCEV.

---

## Requisitos previos

- Python 3.11+ (recomendado 3.14)
- [uv](https://docs.astral.sh/uv/) instalado
- MongoDB Community Server corriendo localmente en `localhost:27017`
- Cuenta en MongoDB Atlas con el corpus CEV cargado (colección `ChunksProyecto.Chunks`)
- Clave de API de OpenAI

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd trabajoGradoSergioSosa

# 2. Instalar dependencias con uv
uv sync

# 3. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales
```

### Variables de entorno (`.env`)

| Variable | Descripción | Ejemplo |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Clave de API de OpenAI | `sk-proj-...` |
| `ATLAS_URI` | URI de MongoDB Atlas | `mongodb+srv://usuario:pass@cluster.mongodb.net/` |
| `MONGO_URI` | URI de MongoDB local | `mongodb://localhost:27017/` |

---

## Iniciar el sistema conversacional

```bash
uv run python src/app/api.py
```

Abre en el navegador: **http://localhost:8000**

### Usar la interfaz web

1. Selecciona un actor (Víctima, Victimario o Tercero)
2. Completa el perfil del actor (nombre, perspectiva, estilo de habla)
3. Haz clic en **Iniciar actor** — el sistema generará la personalidad BDI
4. Escribe preguntas en el campo de chat y presiona Enter
5. El agente responderá simulando al actor con su personalidad y testimonios reales del corpus CEV

### Roles disponibles

| Actor | Descripción |
|-------|-------------|
| **Víctima** | Persona que sufrió daños directos del conflicto |
| **Victimario** | Persona responsable de los hechos |
| **Tercero** | Observador o mediador del conflicto |

---

## Reiniciar la memoria

Desde la interfaz web usa el botón de reinicio, o directamente vía API:

```bash
# Reiniciar todos los actores
curl -X POST http://localhost:8000/reset-memoria

# Reiniciar un actor específico
curl -X POST http://localhost:8000/reset-memoria/victima
```

---

## Ejecutar la evaluación experimental

El pipeline de evaluación corre automáticamente las 4 condiciones (baseline, rag_only, bdi_only, bdi_rag) con 276 trazas y genera reportes en `src/evaluacion/resultados/`.

```bash
# Evaluación completa (borra datos previos, ~1 hora)
python src/evaluacion/correr_evaluacion_completa.py

# Conservar trazas existentes
python src/evaluacion/correr_evaluacion_completa.py --skip-borrar

# Retomar desde un paso intermedio (1–5)
python src/evaluacion/correr_evaluacion_completa.py --desde 2

# Sin generar gráficos
python src/evaluacion/correr_evaluacion_completa.py --sin-grafico
```

### Pasos del pipeline

| Paso | Duración est. | Descripción |
|------|--------------|-------------|
| 1 | ~20 min | Genera 276 trazas de conversación en MongoDB |
| 2 | ~30 min | Evaluación LLM-as-judge (OpenAI GPT-4o-mini) |
| 3 | ~2 min | Métricas léxicas sin LLM |
| 4 | ~1 min | Reportes CSV + Excel |
| 5 | ~1 min | Ablación contrafactual (deltas vs baseline) |

### Archivos generados

Todos en `src/evaluacion/resultados/`:

| Archivo | Contenido |
|---------|-----------|
| `detalle_completo.csv` | Todos los registros individuales |
| `metricas_por_condicion.csv` | Medias por condición experimental |
| `metricas_por_rol.csv` | Medias por actor (victima/victimario/tercero) |
| `scores_por_condicion_rol.csv` | persona_score / rag_score / bdi_score |
| `analisis_estadistico.csv` | Kruskal-Wallis, Mann-Whitney, Cliff's delta |
| `ablacion_delta_resumen.csv` | Deltas causales vs baseline |
| `reporte_evaluacion.xlsx` | Todas las hojas en un Excel |

---

## Comandos Make

```bash
make install     # uv sync
make run         # inicia la API
make eval-full   # evaluación completa
```

---

## Solución de problemas

**El agente no responde / timeout:**
- Verifica que `OPENAI_API_KEY` sea válida y tenga crédito
- Revisa los logs en la terminal donde corre `api.py`

**Error de conexión a MongoDB:**
- Verifica que MongoDB local esté corriendo: `mongod --version`
- Comprueba que `MONGO_URI` en `.env` sea correcta

**El RAG no recupera chunks:**
- Verifica que `ATLAS_URI` apunte al cluster correcto
- Confirma que la colección `ChunksProyecto.Chunks` exista y tenga el índice `hybrid_search_index`

**Error de importación de módulos:**
- Ejecuta desde la raíz del proyecto con `uv run python ...`
- Verifica que `uv sync` haya completado sin errores
