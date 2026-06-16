#!/usr/bin/env python3
"""
Exporta los resultados del análisis estadístico de evaluación ciega LLM-as-judge
a archivos CSV en promedios/resultados_jueces_llm/
"""

import json
import openpyxl
import csv
import os
from pathlib import Path

# ── Rutas ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent.parent
EXCEL = BASE / "plantilla_anotacion.xlsx"
MAPEO = BASE / "conf" / "mapeo_condiciones.json"
OUT_DIR = Path(__file__).parent / "promedios" / "resultados_jueces_llm"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Cargar datos ─────────────────────────────────────────────────────────────
with open(MAPEO, encoding="utf-8") as f:
    mapeo = json.load(f)

wb = openpyxl.load_workbook(EXCEL)

JUECES = {
    "GPT-5":          "Resultados_GPT5",
    "Gemini3Flash":   "Resultados_Gemini3Flash",
    "Qwen36Plus":     "Resultados_Qwen36Plus",
}
LETRAS = ["A", "B", "C", "D"]
CONDICIONES = ["baseline", "rag_only", "bdi_only", "bdi_rag"]
N_PROMPTS = 39

def leer_scores(sheet_name):
    """
    Retorna dict: {prompt_num (1-39): {condicion: score}}
    Columnas: C=Ranking, D=PuntA, E=PuntB, F=PuntC, G=PuntD
    """
    ws = wb[sheet_name]
    scores = {}
    for p in range(1, N_PROMPTS + 1):
        row = p + 1
        clave = f"pregunta_{p}"
        mapeo_p = mapeo[clave]["mapeo"]
        letra_to_cond = {L: mapeo_p[L] for L in LETRAS}
        cols = {"A": "D", "B": "E", "C": "F", "D": "G"}
        scores[p] = {}
        for letra, col in cols.items():
            val = ws[f"{col}{row}"].value
            if val is not None:
                scores[p][letra_to_cond[letra]] = int(val)
    return scores

# Cargar scores de todos los jueces
datos = {}
for juez, sheet in JUECES.items():
    datos[juez] = leer_scores(sheet)

# ── 1. Scores crudos por juez, prompt y condición ────────────────────────────
print("Exportando scores crudos...")
with open(OUT_DIR / "scores_crudos.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["juez", "prompt", "actor", "pregunta_texto", "condicion", "score"])
    for juez in JUECES:
        for p in range(1, N_PROMPTS + 1):
            clave = f"pregunta_{p}"
            actor = mapeo[clave]["actor"]
            texto = mapeo[clave]["pregunta_texto"]
            for cond in CONDICIONES:
                score = datos[juez][p].get(cond)
                if score is not None:
                    w.writerow([juez, p, actor, texto, cond, score])

# ── 2. Medias por condición por juez ─────────────────────────────────────────
print("Exportando medias por condición...")
medias = {}
for juez in JUECES:
    medias[juez] = {}
    for cond in CONDICIONES:
        vals = [datos[juez][p][cond] for p in range(1, N_PROMPTS + 1) if cond in datos[juez][p]]
        medias[juez][cond] = round(sum(vals) / len(vals), 4) if vals else None

with open(OUT_DIR / "medias_por_condicion.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["juez"] + CONDICIONES + ["promedio_condiciones"])
    for juez in JUECES:
        row = [juez] + [medias[juez][c] for c in CONDICIONES]
        prom = round(sum(medias[juez][c] for c in CONDICIONES if medias[juez][c] is not None) / 4, 4)
        row.append(prom)
        w.writerow(row)
    # Promedio entre jueces
    prom_row = ["promedio_jueces"]
    for cond in CONDICIONES:
        vals = [medias[j][cond] for j in JUECES if medias[j][cond] is not None]
        prom_row.append(round(sum(vals) / len(vals), 4))
    prom_row.append(round(sum(prom_row[1:]) / 4, 4))
    w.writerow(prom_row)

# ── 3. Tests estadísticos ─────────────────────────────────────────────────────
try:
    from scipy.stats import friedmanchisquare, wilcoxon
    import numpy as np
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("AVISO: scipy no disponible, omitiendo tests estadísticos")

if HAS_SCIPY:
    print("Ejecutando tests de Friedman...")
    friedman_rows = []
    for juez in JUECES:
        bloques = {cond: [] for cond in CONDICIONES}
        for p in range(1, N_PROMPTS + 1):
            for cond in CONDICIONES:
                bloques[cond].append(datos[juez][p].get(cond, np.nan))
        arrs = [np.array(bloques[c]) for c in CONDICIONES]
        # Filtrar prompts con datos completos
        mask = np.all(~np.isnan(np.column_stack(arrs)), axis=1)
        arrs_clean = [a[mask] for a in arrs]
        stat, p_val = friedmanchisquare(*arrs_clean)
        friedman_rows.append({
            "juez": juez,
            "n_prompts": int(mask.sum()),
            "chi2_friedman": round(stat, 3),
            "p_valor": f"{p_val:.6f}",
            "significativo": "SI" if p_val < 0.05 else "NO"
        })

    with open(OUT_DIR / "friedman.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["juez", "n_prompts", "chi2_friedman", "p_valor", "significativo"])
        w.writeheader()
        w.writerows(friedman_rows)

    print("Ejecutando tests de Wilcoxon por pares...")
    from itertools import combinations
    ALPHA_BONFERRONI = 0.05 / 6

    wilcoxon_rows = []
    rb_acum = {}  # para rank-biserial promediado

    for juez in JUECES:
        for c1, c2 in combinations(CONDICIONES, 2):
            x, y = [], []
            for p in range(1, N_PROMPTS + 1):
                s1 = datos[juez][p].get(c1)
                s2 = datos[juez][p].get(c2)
                if s1 is not None and s2 is not None:
                    x.append(s1)
                    y.append(s2)
            x, y = np.array(x), np.array(y)
            diffs = x - y
            if np.all(diffs == 0):
                stat, p_val = np.nan, 1.0
                rb = 0.0
            else:
                try:
                    stat, p_val = wilcoxon(x, y)
                    # Rank-biserial: r_rb = 1 - 2W / (n*(n+1)/2)
                    n = len(x)
                    rb = round(1 - (2 * stat) / (n * (n + 1) / 2), 4)
                except Exception:
                    stat, p_val, rb = np.nan, np.nan, np.nan

            par = f"{c1} vs {c2}"
            wilcoxon_rows.append({
                "juez": juez,
                "comparacion": par,
                "n": len(x),
                "W_stat": round(float(stat), 3) if not np.isnan(stat) else "",
                "p_valor": round(float(p_val), 6) if not np.isnan(p_val) else "",
                "sig_bonferroni": "SI" if (not np.isnan(p_val) and p_val < ALPHA_BONFERRONI) else "NO",
                "rank_biserial": rb
            })
            rb_acum.setdefault(par, []).append(rb)

    with open(OUT_DIR / "wilcoxon_pares.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["juez", "comparacion", "n", "W_stat", "p_valor", "sig_bonferroni", "rank_biserial"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(wilcoxon_rows)

    print("Calculando rank-biserial promediado...")
    rb_prom_rows = []
    for par, rbs in rb_acum.items():
        rbs_clean = [r for r in rbs if r is not None and not (isinstance(r, float) and np.isnan(r))]
        prom = round(sum(rbs_clean) / len(rbs_clean), 4) if rbs_clean else ""
        mag = ""
        if isinstance(prom, float):
            a = abs(prom)
            mag = "Grande" if a > 0.5 else ("Mediano" if a > 0.3 else ("Pequeno" if a > 0.1 else "Negligible"))
        rb_prom_rows.append({"comparacion": par, "rb_promedio_3jueces": prom, "magnitud": mag})

    with open(OUT_DIR / "rank_biserial_promedio.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["comparacion", "rb_promedio_3jueces", "magnitud"])
        w.writeheader()
        w.writerows(rb_prom_rows)

    print("Calculando concordancia entre jueces...")
    from scipy.stats import spearmanr

    # Ordenamiento de condiciones por cada juez (de mayor a menor media)
    ordenamientos = []
    for juez in JUECES:
        orden = sorted(CONDICIONES, key=lambda c: medias[juez][c] or 0, reverse=True)
        ordenamientos.append(orden)

    # Construir matriz de rangos: filas=jueces, cols=condiciones
    # Rango 1 = mejor (mayor media), 4 = peor (menor media)
    rank_matrix = []
    for orden in ordenamientos:
        fila = [orden.index(c) + 1 for c in CONDICIONES]
        rank_matrix.append(fila)

    mat = np.array(rank_matrix, dtype=float)
    k = mat.shape[0]  # jueces
    n = mat.shape[1]  # condiciones

    # Calcular Kendall W (por completitud)
    R = mat.mean(axis=0)
    R_bar = (n + 1) / 2
    S = np.sum((R - R_bar) ** 2)
    W = 12 * S / (k ** 2 * (n ** 3 - n))

    # Calcular correlaciones de Spearman entre pares de jueces
    jueces_list = list(JUECES.keys())
    rho_pares = []
    for i in range(len(jueces_list)):
        for j in range(i+1, len(jueces_list)):
            rho, p = spearmanr(mat[i], mat[j])
            rho_pares.append(rho)

    rho_promedio = np.mean(rho_pares)

    with open(OUT_DIR / "concordancia_jueces.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "valor", "interpretacion"])
        w.writerow(["Kendall W (nota: max con 3 jueces ≈ 0.5)", round(W, 4), ""])
        w.writerow(["Spearman rho promedio (mejor interpretable)", round(rho_promedio, 4), "Concordancia perfecta" if rho_promedio >= 0.9 else ("Muy alta" if rho_promedio >= 0.7 else "Alta")])
        w.writerow([])
        w.writerow(["juez"] + CONDICIONES + ["ordenamiento"])
        for juez, orden in zip(JUECES, ordenamientos):
            w.writerow([juez] + [medias[juez][c] for c in CONDICIONES] + [" > ".join(orden)])
        w.writerow([])
        w.writerow(["Pares de jueces", "Spearman rho", "p-valor"])
        for i, juez1 in enumerate(jueces_list):
            for j, juez2 in enumerate(jueces_list[i+1:], i+1):
                rho, p = spearmanr(mat[i], mat[j])
                w.writerow([f"{juez1} vs {juez2}", round(rho, 4), round(p, 4)])

# ── 4. Scores por actor ──────────────────────────────────────────────────────
print("Exportando medias por actor y condición...")
actores = ["victima", "victimario", "tercero"]
with open(OUT_DIR / "medias_por_actor.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["juez", "actor"] + CONDICIONES)
    for juez in JUECES:
        for actor in actores:
            prompts_actor = [p for p in range(1, N_PROMPTS + 1)
                             if mapeo[f"pregunta_{p}"]["actor"] == actor]
            row = [juez, actor]
            for cond in CONDICIONES:
                vals = [datos[juez][p][cond] for p in prompts_actor if cond in datos[juez][p]]
                row.append(round(sum(vals) / len(vals), 4) if vals else "")
            w.writerow(row)

print(f"\nResultados guardados en: {OUT_DIR}")
print("Archivos generados:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}")
