#!/usr/bin/env bash
# Ejecuta el pipeline de evaluación experimental completo.
set -e

DESDE="${1:-}"

if [ -n "$DESDE" ]; then
    echo "==> Retomando evaluación desde el paso $DESDE (--skip-borrar)..."
    uv run python src/evaluacion/correr_evaluacion_completa.py --desde "$DESDE" --skip-borrar
else
    echo "==> Ejecutando evaluación completa (~1 hora)..."
    echo "    Uso: $0 [paso]  para retomar desde un paso intermedio (1-5)"
    uv run python src/evaluacion/correr_evaluacion_completa.py
fi

echo ""
echo "Resultados en src/evaluacion/resultados/"
