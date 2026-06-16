"""
evaluador_local.py
==================
Módulo de evaluación usando un LLM local (VM) en lugar de OpenAI/Groq.
Mismos prompts, misma lógica y mismo formato de resultados que evaluador.py.

El LLM local debe exponer una API compatible con OpenAI chat completions:
    POST http://<url>/v1/chat/completions
    Body: {"model": "...", "messages": [...], "temperature": 0.1}

Los resultados se guardan en MongoDB con el prefijo del modelo:
    evaluacion_resultados_<modelo_slug>
    Ejemplo: evaluacion_resultados_mistral_7b

Uso:
    python evaluador_local.py --modelo mistral:7b
    python evaluador_local.py --modelo mistral:7b --condicion bdi_rag
    python evaluador_local.py --modelo mistral:7b --url http://192.168.1.10:10001
    python evaluador_local.py --modelo llama3:8b --condicion all
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from pymongo import MongoClient

_ROOT = Path(__file__).resolve().parent.parent.parent
_EVAL = Path(__file__).resolve().parent

for _p in [str(_ROOT), str(_EVAL)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluador import (
    evaluar_conversacion,
    COL_TRAZAS,
)

# ─── Configuración ────────────────────────────────────────────────────────────
MONGO_URI       = "mongodb://localhost:27017"
MONGO_DB        = "EntrevistasCEV"
URL_DEFAULT     = "http://localhost:10001/v1/chat/completions"
TEMPERATURA     = 0.1
SLEEP_CALLS     = 1.0   # más holgado para LLMs locales que pueden ser lentos


# ─── Cliente LLM local ────────────────────────────────────────────────────────

class _LocalLLMClient:
    """
    Wrapper compatible con la interfaz de langchain ChatOpenAI/ChatGroq.
    Implementa .invoke(messages) usando requests al LLM local.
    """

    def __init__(self, url: str, modelo: str, temperatura: float = TEMPERATURA):
        self.url        = url
        self.modelo     = modelo
        self.temperatura = temperatura

    def invoke(self, messages: list):
        data = {
            "model"      : self.modelo,
            "messages"   : messages,
            "temperature": self.temperatura,
        }
        try:
            resp = requests.post(self.url, json=data, timeout=400)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Timeout al llamar al LLM local ({self.url})")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"No se pudo conectar al LLM local en {self.url}")
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Respuesta inesperada del LLM local: {exc}")

        class _Respuesta:
            pass

        r = _Respuesta()
        r.content = content
        return r


# ─── Utilidades ───────────────────────────────────────────────────────────────

def _modelo_a_slug(modelo: str) -> str:
    """Convierte 'mistral:7b' → 'mistral_7b' para usar como sufijo de colección."""
    return re.sub(r"[^a-zA-Z0-9]", "_", modelo).strip("_")


def _nombre_coleccion(modelo: str) -> str:
    return f"evaluacion_resultados_{_modelo_a_slug(modelo)}"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(condicion: str, modelo: str, url: str) -> None:
    col_resultados = _nombre_coleccion(modelo)

    print(f"\n{'='*65}")
    print(f"  EVALUACIÓN LOCAL — {modelo}")
    print(f"  URL   : {url}")
    print(f"  Colección: {col_resultados}")
    print(f"{'='*65}")

    client    = MongoClient(MONGO_URI)
    db        = client[MONGO_DB]
    evaluador = _LocalLLMClient(url=url, modelo=modelo)

    # Índice único para idempotencia
    db[col_resultados].create_index(
        [("condicion", 1), ("actor_id", 1), ("turno", 1)],
        unique=True,
    )

    condiciones = (
        ["baseline", "rag_only", "bdi_only", "bdi_rag"]
        if condicion == "all"
        else [condicion]
    )

    for cond in condiciones:
        print(f"\n{'='*62}")
        print(f"  CONDICIÓN: {cond.upper()}")
        print(f"{'='*62}")

        for actor_id in ["victima", "victimario", "tercero"]:

            # Idempotencia — no repetir lo ya evaluado
            ya_evaluados = db[col_resultados].count_documents({
                "condicion": cond,
                "actor_id" : actor_id,
            })
            trazas_totales = db[COL_TRAZAS].count_documents({
                "condicion": cond,
                "actor_id" : actor_id,
            })

            if trazas_totales == 0:
                print(f"\n  ⚠  Sin trazas para {cond}/{actor_id}. "
                      f"Ejecuta generador_trazas.py primero.")
                continue

            if ya_evaluados >= trazas_totales:
                print(f"\n  ⏭  {cond}/{actor_id} ya evaluado "
                      f"({ya_evaluados} resultados). Saltando.")
                continue

            trazas = list(db[COL_TRAZAS].find(
                {"condicion": cond, "actor_id": actor_id},
                {"_id": 0},
            ))

            try:
                evaluar_conversacion(
                    condicion      = cond,
                    actor_id       = actor_id,
                    trazas         = trazas,
                    db             = db,
                    evaluador      = evaluador,
                    col_resultados = col_resultados,
                )
            except Exception as exc:
                print(f"\n  ✗  Error evaluando {cond}/{actor_id}: {exc}")

            time.sleep(SLEEP_CALLS)

    client.close()
    print(f"\n✅  Evaluación completada. Resultados en: {col_resultados}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación con LLM local compatible con OpenAI chat completions"
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default="mistral:7b",
        help="Nombre del modelo en el servidor local (ej: mistral:7b, llama3:8b)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=URL_DEFAULT,
        help=f"URL base del servidor LLM local (default: {URL_DEFAULT})",
    )
    parser.add_argument(
        "--condicion",
        choices=["baseline", "rag_only", "bdi_only", "bdi_rag", "all"],
        default="all",
        help="Condición experimental a evaluar (default: all)",
    )

    args = parser.parse_args()
    main(args.condicion, args.modelo, args.url)
