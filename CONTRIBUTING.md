# Guía de contribución

Este repositorio es el código fuente del trabajo de grado "Agentes Conversacionales con Personalidad BDI" — Sergio Sosa, Pontificia Universidad Javeriana, 2026.

---

## Antes de contribuir

1. Abre un issue describiendo el cambio propuesto
2. Espera retroalimentación antes de implementar
3. Haz fork del repositorio y trabaja en una rama con nombre descriptivo (`feature/nombre`, `fix/descripción`)

---

## Entorno de desarrollo

```bash
# 1. Instalar uv
# https://docs.astral.sh/uv/

# 2. Instalar dependencias
uv sync

# 3. Copiar y completar variables de entorno
cp .env.example .env
```

Requisitos externos:
- MongoDB Community Server en `localhost:27017`
- MongoDB Atlas con colección `ChunksProyecto.Chunks` e índice `hybrid_search_index`
- Clave de API de OpenAI

---

## Estilo de código

- **Lenguaje:** Python 3.11+
- **Formatter/Linter:** ruff (`uv run ruff check src/`)
- **Nombres:** snake_case para variables y funciones, español para identificadores de dominio
- **Sin comentarios triviales:** solo documenta el *por qué*, no el *qué*

---

## Estructura de ramas

| Rama | Propósito |
|------|-----------|
| `main` | Código estable, revisado |
| `feature/*` | Nuevas funcionalidades |
| `fix/*` | Correcciones de bugs |
| `eval/*` | Experimentos de evaluación |

---

## Pull Requests

- Un PR por cambio lógico
- Título descriptivo en español
- Incluir en la descripción: qué cambia, por qué, cómo probarlo
- El CI debe pasar antes del merge

---

## Reporte de bugs

Usa la plantilla de [bug report](.github/ISSUE_TEMPLATE/bug_report.md).
Incluye siempre: versión de Python, OS, actor afectado y condición experimental.
