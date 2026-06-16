"""
ablacion_contrafactual.py
=========================
Ablación contrafactual del RAG — compara, turno a turno, las métricas
de respuestas CON RAG vs SIN RAG (baseline) para el mismo actor.

Responde directamente: "¿qué causa el RAG, controlando todo lo demás?"

Métricas comparadas (las que ya existen en el sistema):
  Léxicas (desde metricas_lexicas_por_turno.csv):
    - conflict_vocab_overlap     → transferencia de vocab del rol
    - ngram_overlap_weighted     → transferencia de frases características
    - conflict_vocab_count       → términos únicos del rol (absoluto)
    - conflict_vocab_density_response → densidad relativa

  LLM-judge (desde MongoDB evaluacion_resultados):
    - autenticidad_lexica        → vocab oral situado (el más sensible al RAG)
    - autenticidad_emocional     → tono afectivo (más sensible al BDI)
    - goal_directedness          → coherencia narrativa observable
    - tactical_consistency       → consistencia táctica observable

  vocab_novelty (desde vocab_novelty.csv):
    - vocab_novelty              → % vocab RAG que baseline no usa

Delta calculado:
    Δ_metrica = valor_rag - valor_baseline   (positivo = RAG mejora)

Genera:
  resultados/ablacion_delta_por_turno.csv  — granular (turno a turno)
  resultados/ablacion_delta_resumen.csv    — medias por condicion × actor
  resultados/ablacion_interpretacion.txt   — resumen textual para tesis

Uso:
    python ablacion_contrafactual.py
    python ablacion_contrafactual.py --sin-grafico
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ─── Rutas ───────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parent.parent.parent
_EVAL       = Path(__file__).resolve().parent
_RESULTADOS = _EVAL / "resultados"

for _p in [str(_ROOT), str(_EVAL)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pymongo import MongoClient

MONGO_URI      = "mongodb://localhost:27017"
MONGO_DB       = "EntrevistasCEV"
COL_RESULTADOS = "evaluacion_resultados"

# ─── Métricas léxicas a comparar ─────────────────────────────────────────────
# Columnas que existen en metricas_lexicas_por_turno.csv
_METRICAS_LEXICAS_DELTA = [
    "conflict_vocab_overlap",
    "ngram_overlap_weighted",
    "conflict_vocab_count",
    "conflict_vocab_density_response",
]

# Scores LLM-judge y embeddings desde MongoDB
_METRICAS_JUDGE_DELTA = [
    "autenticidad_lexica",     # favorecido por RAG — hipótesis principal
    "autenticidad_emocional",  # favorecido por BDI
    "goal_directedness",       # coherencia narrativa observable
    "tactical_consistency",    # consistencia táctica observable
    "factual_grounding",       # anclaje factual en corpus CEV (todas las condiciones)
    "grounding_score",         # similitud semántica respuesta ↔ chunks RAG (rag_only, bdi_rag)
]


# ════════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS
# ════════════════════════════════════════════════════════════════════════════════

def _cargar_metricas_lexicas() -> pd.DataFrame:
    ruta = _RESULTADOS / "metricas_lexicas_por_turno.csv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró {ruta}. "
            "Ejecuta metricas_lexicas.py primero."
        )
    df = pd.read_csv(ruta)
    cols = ["condicion", "actor_id", "turno"] + [
        c for c in _METRICAS_LEXICAS_DELTA if c in df.columns
    ]
    return df[cols]


def _cargar_scores_judge() -> pd.DataFrame:
    """Lee evaluacion_resultados de MongoDB y devuelve DataFrame con scores."""
    client = MongoClient(MONGO_URI)
    db     = client[MONGO_DB]
    docs   = list(db[COL_RESULTADOS].find({}, {"_id": 0}))
    client.close()

    if not docs:
        print("Advertencia: no hay resultados en evaluacion_resultados.")
        return pd.DataFrame()

    filas = []
    for doc in docs:
        scores = doc.get("scores", {})
        fila   = {
            "condicion": doc["condicion"],
            "actor_id" : doc["actor_id"],
            "turno"    : doc["turno"],
        }
        for m in _METRICAS_JUDGE_DELTA:
            fila[m] = scores.get(m)
        filas.append(fila)
    return pd.DataFrame(filas)


def _cargar_vocab_novelty() -> pd.DataFrame:
    ruta = _RESULTADOS / "vocab_novelty.csv"
    if not ruta.exists():
        return pd.DataFrame()
    return pd.read_csv(ruta)[["actor_id", "condicion", "vocab_novelty",
                               "n_terminos_nuevos", "n_terminos_rag"]]


# ════════════════════════════════════════════════════════════════════════════════
# CÁLCULO DE DELTAS
# ════════════════════════════════════════════════════════════════════════════════

def calcular_deltas_por_turno(
    df_lex   : pd.DataFrame,
    df_judge : pd.DataFrame,
) -> pd.DataFrame:
    """
    Para cada (actor_id, turno, condicion_rag), calcula:
        Δ_metrica = valor_condicion_rag - valor_baseline_mismo_turno

    Un Δ positivo significa que la condición RAG mejoró esa métrica.
    """
    # Combinar métricas léxicas y judge
    if not df_judge.empty:
        cols_judge = ["condicion", "actor_id", "turno"] + [
            c for c in _METRICAS_JUDGE_DELTA if c in df_judge.columns
        ]
        df_all = df_lex.merge(df_judge[cols_judge], on=["condicion", "actor_id", "turno"], how="left")
    else:
        df_all = df_lex.copy()

    # Separar baseline de condiciones RAG
    df_base = df_all[df_all["condicion"] == "baseline"].copy()
    df_rag  = df_all[df_all["condicion"] != "baseline"].copy()

    if df_base.empty:
        print("Advertencia: no hay datos baseline para calcular deltas.")
        return pd.DataFrame()

    # Todas las métricas numéricas disponibles
    metricas = [c for c in df_all.columns
                if c not in ("condicion", "actor_id", "turno")]

    # Join por actor_id + turno para comparar mismo turno entre condiciones
    df_merged = df_rag.merge(
        df_base[["actor_id", "turno"] + metricas],
        on     = ["actor_id", "turno"],
        suffixes = ("_rag", "_base"),
    )

    filas = []
    for _, row in df_merged.iterrows():
        fila = {
            "condicion": row["condicion"],
            "actor_id" : row["actor_id"],
            "turno"    : row["turno"],
        }
        for m in metricas:
            v_rag  = row.get(f"{m}_rag")
            v_base = row.get(f"{m}_base")
            fila[f"{m}_rag"]   = v_rag
            fila[f"{m}_base"]  = v_base
            fila[f"delta_{m}"] = (
                round(float(v_rag) - float(v_base), 4)
                if v_rag is not None and v_base is not None
                and not (isinstance(v_rag, float) and np.isnan(v_rag))
                and not (isinstance(v_base, float) and np.isnan(v_base))
                else None
            )
        filas.append(fila)

    return pd.DataFrame(filas)


# ════════════════════════════════════════════════════════════════════════════════
# COMPARACIÓN DIRECTA ENTRE CONDICIONES (sin baseline como referencia)
# ════════════════════════════════════════════════════════════════════════════════

def calcular_deltas_directos(
    df_lex   : pd.DataFrame,
    df_judge : pd.DataFrame,
) -> pd.DataFrame:
    """
    Compara condiciones entre sí sin usar baseline como pivote:

      bdi_rag − rag_only  → efecto INCREMENTAL del BDI sobre el RAG puro
      bdi_rag − bdi_only  → efecto INCREMENTAL del RAG sobre el BDI puro

    Un Δ positivo significa que bdi_rag supera a la condición de referencia.

    Estas comparaciones responden preguntas complementarias al análisis vs baseline:
      - ¿Cuánto agrega el BDI cuando ya hay RAG?
      - ¿Cuánto agrega el RAG cuando ya hay BDI?
    """
    if not df_judge.empty:
        cols_judge = ["condicion", "actor_id", "turno"] + [
            c for c in _METRICAS_JUDGE_DELTA if c in df_judge.columns
        ]
        df_all = df_lex.merge(df_judge[cols_judge], on=["condicion", "actor_id", "turno"], how="left")
    else:
        df_all = df_lex.copy()

    metricas = [c for c in df_all.columns
                if c not in ("condicion", "actor_id", "turno")]

    pares = [
        ("bdi_rag", "rag_only", "efecto_bdi_sobre_rag"),
        ("bdi_rag", "bdi_only", "efecto_rag_sobre_bdi"),
    ]

    filas = []
    for cond_a, cond_b, etiqueta in pares:
        df_a = df_all[df_all["condicion"] == cond_a]
        df_b = df_all[df_all["condicion"] == cond_b]

        if df_a.empty or df_b.empty:
            continue

        df_merged = df_a.merge(
            df_b[["actor_id", "turno"] + metricas],
            on       = ["actor_id", "turno"],
            suffixes = ("_a", "_b"),
        )

        for _, row in df_merged.iterrows():
            fila = {
                "comparacion": etiqueta,
                "actor_id"   : row["actor_id"],
                "turno"      : row["turno"],
            }
            for m in metricas:
                v_a = row.get(f"{m}_a")
                v_b = row.get(f"{m}_b")
                fila[f"delta_{m}"] = (
                    round(float(v_a) - float(v_b), 4)
                    if v_a is not None and v_b is not None
                    and not (isinstance(v_a, float) and np.isnan(v_a))
                    and not (isinstance(v_b, float) and np.isnan(v_b))
                    else None
                )
            filas.append(fila)

    return pd.DataFrame(filas)


# ════════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN E INTERPRETACIÓN
# ════════════════════════════════════════════════════════════════════════════════

def exportar(
    df_delta    : pd.DataFrame,
    df_novelty  : pd.DataFrame,
    df_directos : pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _RESULTADOS.mkdir(exist_ok=True)

    # CSV granular — deltas vs baseline
    path_turno = _RESULTADOS / "ablacion_delta_por_turno.csv"
    df_delta.to_csv(path_turno, index=False)
    print(f"  ablacion_delta_por_turno.csv  ({len(df_delta)} filas)")

    # Resumen por condicion × actor (medias de los deltas vs baseline)
    delta_cols = [c for c in df_delta.columns if c.startswith("delta_")]
    resumen = (
        df_delta.groupby(["condicion", "actor_id"])[delta_cols]
        .mean()
        .round(4)
        .reset_index()
    )

    # Agregar vocab_novelty al resumen
    if not df_novelty.empty:
        resumen = resumen.merge(
            df_novelty[["condicion", "actor_id", "vocab_novelty"]],
            on=["condicion", "actor_id"],
            how="left",
        )

    path_resumen = _RESULTADOS / "ablacion_delta_resumen.csv"
    resumen.to_csv(path_resumen, index=False)
    print(f"  ablacion_delta_resumen.csv    ({len(resumen)} filas)")

    # Comparaciones directas entre condiciones (bdi_rag vs rag_only / bdi_only)
    if not df_directos.empty:
        delta_cols_d = [c for c in df_directos.columns if c.startswith("delta_")]
        resumen_directos = (
            df_directos.groupby(["comparacion", "actor_id"])[delta_cols_d]
            .mean()
            .round(4)
            .reset_index()
        )
        path_directos = _RESULTADOS / "ablacion_deltas_directos.csv"
        resumen_directos.to_csv(path_directos, index=False)
        print(f"  ablacion_deltas_directos.csv  ({len(resumen_directos)} filas)")
    else:
        resumen_directos = pd.DataFrame()

    return resumen, resumen_directos


def interpretar(
    resumen          : pd.DataFrame,
    df_novelty       : pd.DataFrame,
    resumen_directos : pd.DataFrame = None,
) -> None:
    """
    Genera una interpretación textual del ablación para facilitar la redacción
    de la sección de resultados en la tesis.
    """
    if resumen_directos is None:
        resumen_directos = pd.DataFrame()

    sep = "═" * 64
    lineas = [sep, "  ABLACIÓN CONTRAFACTUAL — EFECTO CAUSAL DE RAG Y BDI", sep, ""]
    lineas.append(
        "Δ = valor_condicion − valor_baseline\n"
        "Positivo = la condición mejora la métrica respecto al modelo sin RAG ni BDI.\n"
        "  rag_only − baseline → efecto causal del RAG puro\n"
        "  bdi_only − baseline → efecto causal del BDI puro\n"
        "  bdi_rag  − baseline → efecto combinado (RAG + BDI)\n"
    )

    for condicion in ("rag_only", "bdi_only", "bdi_rag"):
        sub = resumen[resumen["condicion"] == condicion]
        if sub.empty:
            continue
        lineas.append(f"\n  CONDICIÓN: {condicion.upper()}")
        lineas.append("-" * 48)

        for _, fila in sub.iterrows():
            actor = fila["actor_id"]
            lineas.append(f"\n  Actor: {actor}")

            # Léxicas
            for m in _METRICAS_LEXICAS_DELTA:
                col = f"delta_{m}"
                if col in fila and fila[col] is not None and not (isinstance(fila[col], float) and np.isnan(fila[col])):
                    signo = "+" if fila[col] >= 0 else ""
                    lineas.append(f"    {m:<38}: {signo}{fila[col]:.4f}")

            # Judge
            for m in _METRICAS_JUDGE_DELTA:
                col = f"delta_{m}"
                if col in fila and fila[col] is not None and not (isinstance(fila[col], float) and np.isnan(fila[col])):
                    signo = "+" if fila[col] >= 0 else ""
                    lineas.append(f"    {m:<38}: {signo}{fila[col]:.4f}")

            # vocab_novelty
            if "vocab_novelty" in fila and fila["vocab_novelty"] is not None:
                lineas.append(f"    {'vocab_novelty':<38}: {fila['vocab_novelty']:.4f}")

    # ── Comparaciones directas entre condiciones (sin baseline como pivote) ──
    if not resumen_directos.empty:
        lineas.append(f"\n{sep}")
        lineas.append("  COMPARACIONES DIRECTAS ENTRE CONDICIONES")
        lineas.append(sep)
        lineas.append(
            "Δ positivo = bdi_rag supera a la condición de referencia.\n"
            "  efecto_bdi_sobre_rag → bdi_rag − rag_only  (¿cuánto agrega BDI al RAG puro?)\n"
            "  efecto_rag_sobre_bdi → bdi_rag − bdi_only  (¿cuánto agrega RAG al BDI puro?)\n"
        )
        for comparacion in ("efecto_bdi_sobre_rag", "efecto_rag_sobre_bdi"):
            sub = resumen_directos[resumen_directos["comparacion"] == comparacion]
            if sub.empty:
                continue
            lineas.append(f"\n  {comparacion.upper().replace('_', ' ')}")
            lineas.append("-" * 48)
            for _, fila in sub.iterrows():
                lineas.append(f"\n  Actor: {fila['actor_id']}")
                for m in _METRICAS_LEXICAS_DELTA + _METRICAS_JUDGE_DELTA:
                    col = f"delta_{m}"
                    if col in fila and fila[col] is not None and not (isinstance(fila[col], float) and np.isnan(fila[col])):
                        signo = "+" if fila[col] >= 0 else ""
                        lineas.append(f"    {m:<38}: {signo}{fila[col]:.4f}")

    # Diagnóstico de hipótesis principal
    lineas.append(f"\n{sep}")
    lineas.append("  DIAGNÓSTICO DE HIPÓTESIS")
    lineas.append(sep)

    hipotesis = [
        ("Hipótesis 1 (RAG): mejora autenticidad_lexica",
         "delta_autenticidad_lexica", "rag_only"),
        ("Hipótesis 2 (BDI): mejora autenticidad_emocional",
         "delta_autenticidad_emocional", "bdi_only"),   # BDI puro, no bdi_rag
        ("Hipótesis 3 (RAG): transfiere vocabulario del conflicto",
         "delta_conflict_vocab_overlap", "rag_only"),
        ("Hipótesis 4 (BDI): mejora coherencia narrativa (goal_directedness)",
         "delta_goal_directedness", "bdi_only"),        # BDI puro, no bdi_rag
    ]

    for desc, col, cond in hipotesis:
        sub = resumen[resumen["condicion"] == cond]
        if sub.empty or col not in sub.columns:
            lineas.append(f"\n  {desc}: DATOS NO DISPONIBLES")
            continue
        media = sub[col].mean()
        if pd.isna(media):
            lineas.append(f"\n  {desc}: DATOS NO DISPONIBLES")
            continue
        veredicto = "CONFIRMADA" if media > 0 else "NO CONFIRMADA"
        lineas.append(f"\n  {desc}")
        lineas.append(f"    Delta promedio = {media:+.4f}  → {veredicto}")

    lineas.append(f"\n{sep}\n")

    texto = "\n".join(lineas)
    print(texto)

    path_txt = _RESULTADOS / "ablacion_interpretacion.txt"
    path_txt.write_text(texto, encoding="utf-8")
    print(f"  ablacion_interpretacion.txt")


def generar_grafico(df_delta: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    delta_plot = [c for c in df_delta.columns if c.startswith("delta_")
                  and c in [f"delta_{m}" for m in
                             _METRICAS_LEXICAS_DELTA + _METRICAS_JUDGE_DELTA]]
    if not delta_plot:
        return

    actores     = ["victima", "victimario", "tercero"]
    condiciones = ["rag_only", "bdi_only", "bdi_rag"]
    colores     = {
        "rag_only": "#3498db",
        "bdi_only": "#2ecc71",
        "bdi_rag" : "#e74c3c",
    }

    fig, axes = plt.subplots(1, len(delta_plot),
                              figsize=(4 * len(delta_plot), 5))
    if len(delta_plot) == 1:
        axes = [axes]

    fig.suptitle("Ablación contrafactual — Δ (condición − baseline) por métrica",
                 fontsize=13, fontweight="bold")

    for ax, col in zip(axes, delta_plot):
        x      = np.arange(len(actores))
        width  = 0.25                    # 3 condiciones — ancho ajustado
        offset = -width                  # centrar el grupo de 3 barras

        for cond in condiciones:
            sub  = df_delta[df_delta["condicion"] == cond]
            vals = []
            for actor in actores:
                v = sub[sub["actor_id"] == actor][col].mean()
                vals.append(v if pd.notna(v) else 0)
            ax.bar(x + offset, vals, width, label=cond,
                   color=colores[cond], alpha=0.85)
            offset += width

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        nombre = col.replace("delta_", "").replace("_", "\n")
        ax.set_title(nombre, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(actores, fontsize=8)
        ax.set_ylabel("Δ promedio")
        ax.grid(axis="y", alpha=0.3)

    parches = [mpatches.Patch(color=colores[c], label=c) for c in condiciones]
    fig.legend(handles=parches, loc="lower center", ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    path = _RESULTADOS / "ablacion_delta.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ablacion_delta.png")


# ════════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════════════════════════

def main(con_grafico: bool = True) -> None:
    print("Cargando datos...")
    df_lex     = _cargar_metricas_lexicas()
    df_judge   = _cargar_scores_judge()
    df_novelty = _cargar_vocab_novelty()
    print(f"   {len(df_lex)} filas lexicas | "
          f"{len(df_judge)} filas judge | "
          f"{len(df_novelty)} filas novelty")

    print("\nCalculando deltas contrafactuales (vs baseline)...")
    df_delta = calcular_deltas_por_turno(df_lex, df_judge)
    if df_delta.empty:
        print("No se pudieron calcular deltas. Verifica que existan trazas baseline.")
        return

    print("\nCalculando comparaciones directas entre condiciones...")
    df_directos = calcular_deltas_directos(df_lex, df_judge)
    if df_directos.empty:
        print("  ⚠  No hay datos para comparaciones directas (se requieren rag_only, bdi_only y bdi_rag).")

    print("\nExportando:")
    resumen, resumen_directos = exportar(df_delta, df_novelty, df_directos)

    if con_grafico:
        generar_grafico(df_delta)

    interpretar(resumen, df_novelty, resumen_directos)
    print("\nListo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablación contrafactual del RAG — EntrevistasCEV"
    )
    parser.add_argument("--sin-grafico", action="store_true")
    args = parser.parse_args()
    main(con_grafico=not args.sin_grafico)
