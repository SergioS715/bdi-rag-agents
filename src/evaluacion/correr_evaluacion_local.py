"""
correr_evaluacion_local.py
===========================
Corre evaluaciones secuenciales con 3 modelos locales de la VM.

Modelos:
  - phi4-mini:latest   (Phi-4-mini 3.8B)
  - qwen3.5:latest     (Qwen 2.5 7B)
  - mistral:7b         (Mistral 7B)

Cada modelo evalúa todas las condiciones y guarda en su propia colección:
  evaluacion_resultados_phi4_mini_latest
  evaluacion_resultados_qwen3_5_latest
  evaluacion_resultados_mistral_7b

Uso:
    python correr_evaluacion_local.py
    python correr_evaluacion_local.py --condicion bdi_rag
    python correr_evaluacion_local.py --url http://192.168.1.10:10001
"""

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_EVAL = Path(__file__).resolve().parent

for _p in [str(_ROOT), str(_EVAL)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluador_local import main as evaluar_con_modelo, URL_DEFAULT

MODELOS = [
    "phi4-mini",
    "qwen3.5",
    "mistral:7b",
]


def main(condicion: str, url: str):
    total = len(MODELOS)
    resultados = []

    for i, modelo in enumerate(MODELOS, 1):
        print(f"\n{'#'*70}")
        print(f"  MODELO {i}/{total}: {modelo}")
        print(f"{'#'*70}")

        inicio = time.time()
        try:
            evaluar_con_modelo(condicion, modelo, url)
            duracion = round(time.time() - inicio)
            resultados.append((modelo, "✓ OK", duracion))
        except Exception as exc:
            duracion = round(time.time() - inicio)
            print(f"\n  ✗  Error con {modelo}: {exc}")
            resultados.append((modelo, f"✗ ERROR: {exc}", duracion))

        if i < total:
            print(f"\n  Esperando 5 segundos antes del siguiente modelo…")
            time.sleep(5)

    # Resumen final
    print(f"\n{'='*70}")
    print("  RESUMEN FINAL")
    print(f"{'='*70}")
    for modelo, estado, secs in resultados:
        mins = secs // 60
        print(f"  {estado}  {modelo:30s}  ({mins}m {secs%60}s)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación secuencial con 3 modelos locales"
    )
    parser.add_argument(
        "--condicion",
        choices=["baseline", "rag_only", "bdi_only", "bdi_rag", "all"],
        default="all",
    )
    parser.add_argument("--url", default=URL_DEFAULT)

    args = parser.parse_args()
    main(args.condicion, args.url)
