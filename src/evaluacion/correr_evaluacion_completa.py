"""
correr_evaluacion_completa.py
==============================
Orquestador completo del pipeline de evaluación — EntrevistasCEV.

Pipeline (6 pasos):
  0. [Opcional] Borra evaluacion_trazas y evaluacion_resultados en MongoDB
  1. generador_trazas.py      → genera trazas de las 4 condiciones experimentales
  2. evaluador.py             → LLM-as-a-judge sobre las trazas (MongoDB)
  3. metricas_lexicas.py      → métricas léxicas desde los CSVs del corpus
  4. reporte_evaluacion.py    → tablas comparativas (CSV + Excel)
  5. ablacion_contrafactual.py → deltas vs baseline + comparaciones directas

Todo el output se guarda también en resultados/pipeline_YYYY-MM-DD_HH-MM.log
para que puedas revisar el resultado aunque no estés mirando la consola.

Uso típico:
    python correr_evaluacion_completa.py              # pipeline completo
    python correr_evaluacion_completa.py --skip-borrar     # sin limpiar datos previos
    python correr_evaluacion_completa.py --desde 3         # retomar desde metricas_lexicas
    python correr_evaluacion_completa.py --sin-grafico     # sin generar .png

Flags de reanudación (--desde N):
    1 = generador_trazas
    2 = evaluador
    3 = metricas_lexicas
    4 = reporte_evaluacion
    5 = ablacion_contrafactual
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
from datetime import datetime

from pymongo import MongoClient

# ─── Rutas ────────────────────────────────────────────────────────────────────
_EVAL       = Path(__file__).resolve().parent
_RESULTADOS = _EVAL / "resultados"

GENERADOR  = _EVAL / "generador_trazas.py"
EVALUADOR  = _EVAL / "evaluador.py"
LEX        = _EVAL / "metricas_lexicas.py"
REPORTE    = _EVAL / "reporte_evaluacion.py"
ABLACION   = _EVAL / "ablacion_contrafactual.py"

MONGO_URI       = "mongodb://localhost:27017"
MONGO_DB        = "EntrevistasCEV"
COL_TRAZAS      = "evaluacion_trazas"
COL_RESULTADOS  = "evaluacion_resultados"
COL_CHECKPOINTS = "langgraph_checkpoints_eval"
COL_WRITES      = "langgraph_writes_eval"


class _Tee:
    """Redirige stdout a consola Y a un archivo de log simultáneamente."""

    def __init__(self, ruta_log: Path):
        _RESULTADOS.mkdir(exist_ok=True)
        self._consola = sys.stdout
        self._archivo = open(ruta_log, "w", encoding="utf-8", buffering=1)

    def write(self, texto: str) -> None:
        self._consola.write(texto)
        self._consola.flush()
        self._archivo.write(texto)

    def flush(self) -> None:
        self._consola.flush()
        self._archivo.flush()

    def close(self) -> None:
        self._archivo.close()


# ════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _separador(n: int, titulo: str) -> None:
    ancho = 64
    print(f"\n{'═'*ancho}")
    print(f"  [{_ts()}]  PASO {n} — {titulo}")
    print(f"{'═'*ancho}")


def _borrar_colecciones() -> None:
    _separador(0, "Limpiando datos previos en MongoDB")
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    for col in [COL_TRAZAS, COL_RESULTADOS, COL_CHECKPOINTS, COL_WRITES]:
        n = db[col].delete_many({}).deleted_count
        print(f"  ✓  {col:<44s} → {n} documentos eliminados")
    client.close()
    print()


def _correr_script(
    ruta     : Path,
    args     : list[str] = None,
    timeout  : int = None,
) -> tuple[bool, float]:
    """
    Ejecuta un script Python como subproceso.
    Retorna (éxito: bool, duración_segundos: float).
    """
    cmd = [sys.executable, str(ruta)] + (args or [])
    print(f"\n  $ {' '.join(cmd)}\n")

    t0      = time.time()
    proceso = subprocess.run(
        cmd,
        cwd    = str(ruta.parent),
        timeout= timeout,
    )
    dur = time.time() - t0

    if proceso.returncode != 0:
        print(f"\n  ✗  ERROR — {ruta.name} terminó con código {proceso.returncode}")
        return False, dur

    print(f"\n  ✓  {ruta.name} completado en {dur/60:.1f} min")
    return True, dur


def _verificar_prerequisitos() -> bool:
    """Comprueba que todos los scripts del pipeline existen."""
    ok = True
    for ruta in [GENERADOR, EVALUADOR, LEX, REPORTE, ABLACION]:
        if not ruta.exists():
            print(f"  ✗  No encontrado: {ruta}")
            ok = False
    return ok


def _listar_resultados() -> None:
    """Imprime los archivos generados en resultados/."""
    if not _RESULTADOS.exists():
        return
    archivos = sorted(_RESULTADOS.iterdir())
    if not archivos:
        return
    print("\n  Archivos generados:")
    for f in archivos:
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<50s}  {size_kb:>8.1f} KB")


# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════════

PASOS = {
    1: ("Generando trazas (4 condiciones)",        GENERADOR,  ["--condicion", "all"]),
    2: ("Evaluando con LLM-as-a-judge",            EVALUADOR,  ["--condicion", "all"]),
    3: ("Calculando métricas léxicas",             LEX,        []),
    4: ("Generando reporte (CSV + Excel + tablas)", REPORTE,   ["--formato", "todos"]),
    5: ("Ablación contrafactual",                  ABLACION,   []),
}

# Estimaciones de duración (minutos) para la barra de progreso informativa
_DURACION_EST = {1: 20, 2: 30, 3: 2, 4: 1, 5: 1}


def main(
    skip_borrar  : bool = False,
    desde        : int  = 1,
    sin_grafico  : bool = False,
) -> None:
    # ── Log dual ──────────────────────────────────────────────────────────────
    _RESULTADOS.mkdir(exist_ok=True)
    marca = datetime.now().strftime("%Y-%m-%d_%H-%M")
    ruta_log = _RESULTADOS / f"pipeline_{marca}.log"
    tee = _Tee(ruta_log)
    sys.stdout = tee

    inicio_total = time.time()
    print(f"\n{'═'*64}")
    print(f"  PIPELINE DE EVALUACIÓN — EntrevistasCEV")
    print(f"  Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Log:      {ruta_log.name}")
    print(f"{'═'*64}")

    if not _verificar_prerequisitos():
        print("\n❌  Faltan scripts del pipeline. Revisa la instalación.")
        sys.exit(1)

    duraciones: dict[int, float] = {}

    # ── Paso 0: limpiar ───────────────────────────────────────────────────────
    if not skip_borrar and desde == 1:
        _borrar_colecciones()
    elif skip_borrar:
        print("\n  ⏭  --skip-borrar: datos previos conservados")
    elif desde > 1:
        print(f"\n  ⏭  Reanudando desde paso {desde} — datos previos conservados")

    # ── Pasos 1–5 ─────────────────────────────────────────────────────────────
    for n, (titulo, script, args_base) in PASOS.items():
        if n < desde:
            print(f"\n  ⏭  Paso {n} omitido (--desde {desde})")
            continue

        _separador(n, titulo)
        est = _DURACION_EST.get(n, "?")
        print(f"  (duración estimada: ~{est} min)\n")

        # Agregar --sin-grafico a los pasos que lo soporten
        args = list(args_base)
        if sin_grafico and script in (LEX, ABLACION):
            args.append("--sin-grafico")

        ok, dur = _correr_script(script, args)
        duraciones[n] = dur

        if not ok:
            print(f"\n{'═'*64}")
            print(f"  ❌  Pipeline abortado en el paso {n} ({script.name})")
            print(f"  Para retomar desde aquí ejecuta:")
            print(f"      python {Path(__file__).name} --skip-borrar --desde {n}")
            print(f"{'═'*64}\n")
            sys.stdout = tee._consola
            tee.close()
            sys.exit(1)

    # ── Resumen final ──────────────────────────────────────────────────────────
    total = time.time() - inicio_total
    print(f"\n{'═'*64}")
    print(f"  ✅  Pipeline completado — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duración total: {total/60:.1f} min")
    print()
    for n, dur in duraciones.items():
        titulo = PASOS[n][0]
        print(f"    Paso {n}  {titulo:<45s}  {dur/60:5.1f} min")
    _listar_resultados()
    print(f"\n  Log guardado en: {ruta_log}")
    print(f"{'═'*64}\n")

    sys.stdout = tee._consola
    tee.close()


# ════════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline completo de evaluación — EntrevistasCEV",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--skip-borrar",
        action="store_true",
        help="No borrar datos previos en MongoDB.\n"
             "Útil si quieres evaluar condiciones adicionales sin repetir las ya generadas.",
    )
    parser.add_argument(
        "--desde",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        metavar="N",
        help=(
            "Retomar el pipeline desde el paso N (sin borrar datos):\n"
            "  1 = generador_trazas  (default)\n"
            "  2 = evaluador\n"
            "  3 = metricas_lexicas\n"
            "  4 = reporte_evaluacion\n"
            "  5 = ablacion_contrafactual"
        ),
    )
    parser.add_argument(
        "--sin-grafico",
        action="store_true",
        help="No generar archivos .png (metricas_lexicas y ablacion).",
    )

    args = parser.parse_args()

    # --desde implica --skip-borrar (no tiene sentido limpiar si reanudas)
    skip = args.skip_borrar or (args.desde > 1)

    main(
        skip_borrar = skip,
        desde       = args.desde,
        sin_grafico = args.sin_grafico,
    )
