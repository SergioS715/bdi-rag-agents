#!/usr/bin/env bash
# Inicia el sistema conversacional (API + interfaz web).
set -e

if [ ! -f .env ]; then
    echo "ERROR: archivo .env no encontrado. Ejecuta primero: ./scripts/setup.sh"
    exit 1
fi

echo "==> Iniciando sistema conversacional..."
echo "    Interfaz web disponible en: http://localhost:8000"
echo "    Presiona Ctrl+C para detener."
echo ""

uv run python src/app/api.py
