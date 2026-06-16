"""
reporte_evaluacion.py
=====================
Lee los resultados de evaluacion_resultados (MongoDB) y genera tablas
comparativas entre las cuatro condiciones experimentales.

Estructura de métricas:
  Universales (todas las condiciones — base de persona_score):
    estabilidad_rol, autenticidad_lexica, autenticidad_emocional,
    goal_directedness, tactical_consistency, factual_grounding, answer_relevance

  RAG (rag_only, bdi_rag — forman rag_score):
    lexical_adoption, grounding_score

  BDI (bdi_only, bdi_rag — forman bdi_score):
    coherencia_tactica, subordinacion_macro

Salidas (en ArchivosCode/Evaluacion/resultados/):
  metricas_por_condicion.csv   — promedio de cada métrica por condición (A/B/C/D)
  metricas_por_rol.csv         — promedio de métricas universales por rol de actor
  metricas_por_tipologia.csv   — promedio por tipología de pregunta
  scores_por_condicion_rol.csv — PersonaScore / RAG_Score / BDI_Score por condición+rol
  detalle_completo.csv         — todos los registros individuales

Uso:
    python reporte_evaluacion.py
    python reporte_evaluacion.py --formato csv         (default)
    python reporte_evaluacion.py --formato tabla        (imprime en consola)
"""

import argparse
import sys
from pathlib import Path

# ─── Ajuste de rutas ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_EVAL = Path(__file__).resolve().parent

for _p in [str(_ROOT), str(_EVAL)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas no está instalado. Ejecuta: pip install pandas")
    sys.exit(1)

try:
    import numpy as np
    from scipy import stats as _scipy_stats
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    print("⚠  scipy/numpy no encontrado — análisis estadístico deshabilitado. "
          "Instala con: pip install scipy numpy")

from pymongo import MongoClient
from config import MONGO_URI as _MONGO_URI_CONFIG

# ─── Configuración ────────────────────────────────────────────────────────────
MONGO_URI      = "mongodb://localhost:27017"
MONGO_DB       = "EntrevistasCEV"
COL_RESULTADOS = "evaluacion_resultados"

RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

# ── Métricas universales: comparables entre las 4 condiciones ────────────────
# Estas son las únicas que deben usarse en la tabla principal de comparación.
# Forman persona_score cuando se promedian.
METRICAS_UNIVERSALES = [
    "estabilidad_rol",
    "autenticidad_lexica",
    "autenticidad_emocional",
    "goal_directedness",
    "tactical_consistency",
    "factual_grounding",
    "answer_relevance",   # universal pero no parte de persona_score
]

# ── Métricas específicas de condición (calidad interna del mecanismo) ─────────
METRICAS_RAG = ["lexical_adoption", "grounding_score"]                  # rag_only, bdi_rag
METRICAS_BDI = ["coherencia_tactica", "subordinacion_macro"]          # bdi_only, bdi_rag

SCORES_AGREGADOS = ["persona_score", "rag_score", "bdi_score"]

CONDICION_LABEL = {
    "baseline" : "A — Baseline",
    "rag_only" : "B — RAG only",
    "bdi_only" : "C — BDI only",
    "bdi_rag"  : "D — BDI+RAG",
}


# ════════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ════════════════════════════════════════════════════════════════════════════════

def cargar_resultados() -> pd.DataFrame:
    """
    Lee evaluacion_resultados y devuelve un DataFrame plano con todas
    las métricas como columnas numéricas.
    """
    client = MongoClient(MONGO_URI)
    db     = client[MONGO_DB]

    docs = list(db[COL_RESULTADOS].find({}, {"_id": 0}))
    client.close()

    if not docs:
        print("⚠  No hay resultados en evaluacion_resultados. "
              "Ejecuta evaluador.py primero.")
        return pd.DataFrame()

    # Aplanar el sub-diccionario 'scores'
    filas = []
    for doc in docs:
        fila = {
            "condicion"         : doc["condicion"],
            "actor_id"          : doc["actor_id"],
            "turno"             : doc["turno"],
            "tipologia_pregunta": doc.get("tipologia_pregunta", ""),
            "pregunta"          : doc.get("pregunta", ""),
            "respuesta"         : doc.get("respuesta", ""),
            "persona_score"     : doc.get("persona_score"),
            "bdi_score"         : doc.get("bdi_score"),
            "rag_score"         : doc.get("rag_score"),
        }
        scores = doc.get("scores", {})
        # ── Universales ──────────────────────────────────────────────────────
        fila["estabilidad_rol"]         = scores.get("estabilidad_rol")
        fila["autenticidad_lexica"]     = scores.get("autenticidad_lexica")
        fila["autenticidad_emocional"]  = scores.get("autenticidad_emocional")
        fila["goal_directedness"]       = scores.get("goal_directedness")
        fila["tactical_consistency"]    = scores.get("tactical_consistency")
        fila["factual_grounding"]       = scores.get("factual_grounding")
        fila["answer_relevance"]        = scores.get("answer_relevance")
        # ── RAG (rag_only, bdi_rag) ───────────────────────────────────────────
        fila["lexical_adoption"]            = scores.get("lexical_adoption")
        # ── BDI (bdi_only, bdi_rag) ───────────────────────────────────────────
        fila["coherencia_tactica"]      = scores.get("coherencia_tactica")
        fila["subordinacion_macro"]     = scores.get("subordinacion_macro")
        # ── Legacy (no usar en análisis principal) ────────────────────────────
        fila["autenticidad_discursiva"] = scores.get("autenticidad_discursiva")
        filas.append(fila)

    df = pd.DataFrame(filas)

    # Asegurar tipos numéricos
    cols_num = [
        # Universales
        "estabilidad_rol", "autenticidad_lexica", "autenticidad_emocional",
        "goal_directedness", "tactical_consistency", "factual_grounding",
        "answer_relevance",
        # RAG
        "lexical_adoption",
        # BDI
        "coherencia_tactica", "subordinacion_macro",
        # Agregados
        "persona_score", "bdi_score", "rag_score",
        # Legacy
        "autenticidad_discursiva",
    ]
    for col in cols_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ════════════════════════════════════════════════════════════════════════════════
# TABLAS COMPARATIVAS
# ════════════════════════════════════════════════════════════════════════════════

def _cols_universales(df: pd.DataFrame) -> list[str]:
    """Columnas universales disponibles en el DataFrame."""
    return [c for c in METRICAS_UNIVERSALES + ["persona_score"] if c in df.columns]


def _cols_todas(df: pd.DataFrame) -> list[str]:
    """Todas las columnas de métricas disponibles (universales + específicas)."""
    todas = (
        METRICAS_UNIVERSALES
        + METRICAS_RAG
        + METRICAS_BDI
        + SCORES_AGREGADOS
    )
    return [c for c in todas if c in df.columns]


def tabla_por_condicion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Promedio de cada métrica agrupado por condición experimental.
    Muestra primero las métricas universales (comparables entre las 4 condiciones),
    luego las específicas de mecanismo (RAG / BDI), y finalmente los scores agregados.
    """
    cols = _cols_todas(df)
    resultado = (
        df.groupby("condicion")[cols]
        .mean()
        .round(3)
        .reset_index()
    )
    resultado["condicion"] = resultado["condicion"].map(
        lambda c: CONDICION_LABEL.get(c, c)
    )
    return resultado


def tabla_por_rol(df: pd.DataFrame) -> pd.DataFrame:
    """Promedio de métricas universales agrupado por rol de actor."""
    cols = _cols_universales(df)
    return (
        df.groupby("actor_id")[cols]
        .mean()
        .round(3)
        .reset_index()
        .rename(columns={"actor_id": "rol"})
    )


def tabla_por_condicion_y_rol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores agregados y métricas clave por condición × rol.
    Útil para ver si el efecto de cada condición varía según el rol.
    """
    cols_clave = [c for c in
        ["persona_score", "rag_score", "bdi_score"]
        + METRICAS_UNIVERSALES
        if c in df.columns
    ]
    resultado = (
        df.groupby(["condicion", "actor_id"])[cols_clave]
        .mean()
        .round(3)
        .reset_index()
    )
    resultado["condicion"] = resultado["condicion"].map(
        lambda c: CONDICION_LABEL.get(c, c)
    )
    resultado = resultado.rename(columns={"actor_id": "rol"})
    return resultado


def tabla_por_tipologia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Promedio de métricas universales por condición × tipología de pregunta.
    Útil para comparar si las preguntas confrontativas degradan métricas específicas.
    """
    cols = _cols_universales(df)
    return (
        df.groupby(["condicion", "tipologia_pregunta"])[cols]
        .mean()
        .round(3)
        .reset_index()
    )


# ════════════════════════════════════════════════════════════════════════════════
# ANÁLISIS ESTADÍSTICO
# ════════════════════════════════════════════════════════════════════════════════

# Universales primero (comparación cruzada), luego específicas de condición
METRICAS_STAT = [
    # Universales — base de la comparación entre condiciones
    "estabilidad_rol",
    "autenticidad_lexica",
    "autenticidad_emocional",
    "goal_directedness",
    "tactical_consistency",
    "factual_grounding",
    "answer_relevance",
    "persona_score",
    # Específicas de condición — calidad interna del mecanismo
    "lexical_adoption",
    "coherencia_tactica",
    "subordinacion_macro",
    "rag_score",
    "bdi_score",
]
CONDICIONES_STAT = ["baseline", "rag_only", "bdi_only", "bdi_rag"]
PARES_STAT = [
    ("baseline", "rag_only"),
    ("baseline", "bdi_only"),
    ("baseline", "bdi_rag"),
    ("rag_only", "bdi_only"),
    ("rag_only", "bdi_rag"),
    ("bdi_only", "bdi_rag"),
]


def _cliff_delta(x: "np.ndarray", y: "np.ndarray") -> float:
    """Cliff's Delta: magnitud del efecto para datos ordinales. Rango [-1, 1]."""
    dominance = sum(
        1 if xi > yi else -1 if xi < yi else 0
        for xi in x for yi in y
    )
    return dominance / (len(x) * len(y))


def _magnitud_cliff(d: float) -> str:
    """Interpreta la magnitud de Cliff's Delta."""
    ad = abs(d)
    if ad >= 0.474: return "grande"
    if ad >= 0.330: return "mediano"
    if ad >= 0.147: return "pequeño"
    return "negligible"


def _sig_label(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _bootstrap_ci(data: "np.ndarray", n_boot: int = 2000,
                  ci: float = 0.95, seed: int = 42) -> tuple:
    """IC bootstrap percentil para la media."""
    rng  = np.random.default_rng(seed)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return (float("nan"), float("nan"))
    boots = [rng.choice(data, size=len(data), replace=True).mean()
             for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(boots, alpha * 100)),
        float(np.percentile(boots, (1 - alpha) * 100)),
    )


def tabla_kruskal_wallis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kruskal-Wallis por métrica.
    H0: las 3 condiciones tienen la misma distribución.
    Columnas: metrica | H | p_value | significancia
    """
    if not _SCIPY_OK:
        return pd.DataFrame()

    filas = []
    for metrica in METRICAS_STAT:
        grupos = [
            df[df["condicion"] == c][metrica].dropna().values
            for c in CONDICIONES_STAT
        ]
        grupos_validos = [g for g in grupos if len(g) >= 2]
        if len(grupos_validos) < 2:
            continue
        try:
            H, p = _scipy_stats.kruskal(*grupos_validos)
            filas.append({
                "metrica"       : metrica,
                "H"             : round(H, 3),
                "p_value"       : round(p, 4),
                "significancia" : _sig_label(p),
                "conclusion"    : "diferencia significativa" if p < 0.05 else "sin diferencia significativa",
            })
        except Exception:
            pass

    return pd.DataFrame(filas)


def tabla_mann_whitney(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mann-Whitney U pairwise con corrección Bonferroni + Cliff's Delta.
    Columnas: metrica | comparacion | U | p_raw | p_bonferroni | significancia
              cliff_delta | magnitud
    """
    if not _SCIPY_OK:
        return pd.DataFrame()

    n_pares = len(PARES_STAT)
    filas = []

    for metrica in METRICAS_STAT:
        for c1, c2 in PARES_STAT:
            x = df[df["condicion"] == c1][metrica].dropna().values
            y = df[df["condicion"] == c2][metrica].dropna().values
            if len(x) < 2 or len(y) < 2:
                continue
            try:
                U, p_raw  = _scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
                p_bonf    = min(float(p_raw) * n_pares, 1.0)
                d         = _cliff_delta(x, y)
                filas.append({
                    "metrica"      : metrica,
                    "comparacion"  : f"{c1}  vs  {c2}",
                    "U"            : round(U, 1),
                    "p_raw"        : round(float(p_raw), 4),
                    "p_bonferroni" : round(p_bonf, 4),
                    "significancia": _sig_label(p_bonf),
                    "cliff_delta"  : round(d, 3),
                    "magnitud"     : _magnitud_cliff(d),
                })
            except Exception:
                pass

    return pd.DataFrame(filas)


def tabla_bootstrap_ci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Media ± IC 95% bootstrap por métrica × condición.
    Columnas: metrica | condicion | n | media | IC_95_low | IC_95_high | IC_95
    """
    if not _SCIPY_OK:
        return pd.DataFrame()

    filas = []
    for metrica in METRICAS_STAT:
        for cond in CONDICIONES_STAT:
            datos = df[df["condicion"] == cond][metrica].dropna().values
            if len(datos) == 0:
                continue
            media        = float(np.mean(datos))
            ci_low, ci_hi = _bootstrap_ci(datos)
            filas.append({
                "metrica"    : metrica,
                "condicion"  : cond,
                "n"          : len(datos),
                "media"      : round(media, 3),
                "IC_95_low"  : round(ci_low,  3),
                "IC_95_high" : round(ci_hi,   3),
                "IC_95"      : f"[{ci_low:.3f} – {ci_hi:.3f}]",
            })

    return pd.DataFrame(filas)


# ════════════════════════════════════════════════════════════════════════════════
# IMPRESIÓN EN CONSOLA
# ════════════════════════════════════════════════════════════════════════════════

def _imprimir_tabla(titulo: str, df: pd.DataFrame) -> None:
    ancho = max(len(titulo) + 4, 80)
    print(f"\n{'='*ancho}")
    print(f"  {titulo}")
    print(f"{'='*ancho}")
    print(df.to_string(index=False))
    print()


def imprimir_resumen(df: pd.DataFrame) -> None:
    if df.empty:
        return

    _imprimir_tabla("MÉTRICAS POR CONDICIÓN EXPERIMENTAL",   tabla_por_condicion(df))
    _imprimir_tabla("MÉTRICAS POR ROL DE ACTOR",             tabla_por_rol(df))
    _imprimir_tabla("SCORES AGREGADOS POR CONDICIÓN Y ROL",  tabla_por_condicion_y_rol(df))
    _imprimir_tabla("MÉTRICAS POR TIPOLOGÍA DE PREGUNTA",    tabla_por_tipologia(df))

    if _SCIPY_OK:
        _imprimir_tabla("KRUSKAL-WALLIS — ¿Hay diferencia significativa entre condiciones?",
                        tabla_kruskal_wallis(df))
        _imprimir_tabla("MANN-WHITNEY U + BONFERRONI + CLIFF'S DELTA — Comparaciones pairwise",
                        tabla_mann_whitney(df))
        _imprimir_tabla("INTERVALOS DE CONFIANZA 95% BOOTSTRAP — Media por métrica × condición",
                        tabla_bootstrap_ci(df))
        print("  Leyenda significancia: *** p<0.001 | ** p<0.01 | * p<0.05 | ns no significativo")
        print("  Cliff's delta magnitud: negligible <0.147 | pequeño <0.330 | mediano <0.474 | grande ≥0.474\n")

    # Conteo de registros
    print("REGISTROS TOTALES POR CONDICIÓN:")
    print(df.groupby("condicion").size().rename("n_turnos").to_string())
    print()


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN A CSV
# ════════════════════════════════════════════════════════════════════════════════

def exportar_csv(df: pd.DataFrame) -> None:
    if df.empty:
        return

    tablas = {
        "metricas_por_condicion.csv"  : tabla_por_condicion(df),
        "metricas_por_rol.csv"        : tabla_por_rol(df),
        "scores_por_condicion_rol.csv": tabla_por_condicion_y_rol(df),
        "metricas_por_tipologia.csv"  : tabla_por_tipologia(df),
        "detalle_completo.csv"        : df.drop(columns=["pregunta", "respuesta"],
                                                  errors="ignore"),
    }

    if _SCIPY_OK:
        tablas["estadistica_kruskal_wallis.csv"] = tabla_kruskal_wallis(df)
        tablas["estadistica_mann_whitney.csv"]   = tabla_mann_whitney(df)
        tablas["estadistica_bootstrap_ci.csv"]   = tabla_bootstrap_ci(df)

    for nombre, tabla in tablas.items():
        ruta = RESULTADOS_DIR / nombre
        tabla.to_csv(ruta, index=False, encoding="utf-8-sig")
        print(f"  ✓  {nombre:40s} → {ruta}")

    print(f"\n  Carpeta: {RESULTADOS_DIR}")


def exportar_excel(df: pd.DataFrame) -> None:
    """
    Exporta todas las tablas en un único archivo Excel con múltiples hojas.
    Requiere openpyxl: pip install openpyxl
    """
    if df.empty:
        return

    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("ERROR: openpyxl no está instalado. Ejecuta: pip install openpyxl")
        return

    ruta = RESULTADOS_DIR / "reporte_evaluacion.xlsx"

    hojas = {
        "Por Condición"       : tabla_por_condicion(df),
        "Por Rol"             : tabla_por_rol(df),
        "Por Condición y Rol" : tabla_por_condicion_y_rol(df),
        "Por Tipología"       : tabla_por_tipologia(df),
        "Detalle Completo"    : df.drop(columns=["pregunta", "respuesta"], errors="ignore"),
    }

    if _SCIPY_OK:
        hojas["Kruskal-Wallis"]   = tabla_kruskal_wallis(df)
        hojas["Mann-Whitney"]     = tabla_mann_whitney(df)
        hojas["Bootstrap IC 95%"] = tabla_bootstrap_ci(df)

    with pd.ExcelWriter(ruta, engine="openpyxl") as writer:
        for nombre_hoja, tabla in hojas.items():
            tabla.to_excel(writer, sheet_name=nombre_hoja, index=False)

            # Ajustar ancho de columnas automáticamente
            ws = writer.sheets[nombre_hoja]
            for col in ws.columns:
                max_len = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in col
                )
                ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 50)

    print(f"  ✓  reporte_evaluacion.xlsx              → {ruta}")


# ════════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════════════════════════

def generar_reporte(formato: str = "ambos") -> None:
    """
    Genera el reporte comparativo desde MongoDB.

    Args:
        formato: "csv"   → CSVs individuales
                 "excel" → un solo .xlsx con múltiples hojas
                 "tabla" → imprime en consola
                 "todos" → csv + excel + tabla
                 "ambos" → csv + tabla (compatibilidad)
    """
    print("Cargando resultados desde MongoDB…")
    df = cargar_resultados()

    if df.empty:
        return

    n = len(df)
    condiciones_presentes = df["condicion"].unique().tolist()
    print(f"  {n} registros cargados. Condiciones: {condiciones_presentes}\n")

    if formato in ("tabla", "ambos", "todos"):
        imprimir_resumen(df)

    if formato in ("csv", "ambos", "todos"):
        print("Exportando CSVs…")
        exportar_csv(df)

    if formato in ("excel", "todos"):
        print("Exportando Excel…")
        exportar_excel(df)

    print("\n✅  Reporte generado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reporte comparativo de evaluación — EntrevistasCEV"
    )
    parser.add_argument(
        "--formato",
        choices=["csv", "excel", "tabla", "ambos", "todos"],
        default="todos",
        help="Formato de salida (default: todos)",
    )
    args = parser.parse_args()
    generar_reporte(args.formato)
