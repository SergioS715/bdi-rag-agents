#!/usr/bin/env bash
# Configura el entorno de desarrollo del proyecto.
set -e

echo "==> Verificando que uv esté instalado..."
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv no está instalado. Instálalo en https://docs.astral.sh/uv/"
    exit 1
fi

echo "==> Instalando dependencias..."
uv sync

echo "==> Configurando .env..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Archivo .env creado desde .env.example — edítalo con tus credenciales antes de continuar."
else
    echo ".env ya existe, no se sobreescribe."
fi

echo "==> Descargando modelo de embeddings (puede tardar la primera vez)..."
uv run python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print('Modelo de embeddings listo.')
"

echo ""
echo "Entorno configurado correctamente."
echo "Edita .env con tus credenciales y luego ejecuta:"
echo "  uv run python src/app/api.py"
