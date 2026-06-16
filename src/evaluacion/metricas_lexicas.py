"""
metricas_lexicas.py
===================
Métricas de transferencia léxica del RAG — EntrevistasCEV.

Lee evaluacion_trazas desde MongoDB y computa, por cada turno:

UNIVERSALES (todas las condiciones):
  1. conflict_vocab_density_response   -> % palabras del rol en la respuesta
  2. conflict_vocab_count              -> n° términos únicos del rol en respuesta

TRANSFERENCIA RAG (requieren chunks — rag_only y bdi_rag):
  3. rag_transfer_limpio               -> overlap vocab significativo (completo)
  4. rag_transfer_lexical              -> overlap LÉXICO DEL CONFLICTO
  5. rag_transfer_eufemismos           -> overlap EUFEMISMOS PARA LO INNOMBRABLE
  6. rag_transfer_expresiones          -> overlap EXPRESIONES LITERALES

CALIDAD DEL RETRIEVAL (propiedades del chunk — RAG):
  7. top_k_similarity                  -> similitud dot-product promedio
  8. role_purity                       -> % chunks del rol correcto

NOVELTY Y ANÁLISIS CRUZADO:
  9. vocab_novelty                     -> % vocabulario RAG exclusivo del RAG

Genera:
  resultados/metricas_lexicas_por_turno.csv   — datos granulares
  resultados/metricas_lexicas_resumen.csv     — medias por condicion x rol
  resultados/vocab_novelty.csv                — análisis de vocabulario nuevo
  resultados/transferencia_lexica.png         — gráfico comparativo

Uso:
    python metricas_lexicas.py
    python metricas_lexicas.py --sin-grafico
"""

import argparse
import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

#  Rutas
_ROOT = Path(__file__).resolve().parent.parent.parent
_EVAL = Path(__file__).resolve().parent
_RESULTADOS = _EVAL / "resultados"

for _p in [str(_ROOT), str(_EVAL)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pymongo import MongoClient

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB  = "EntrevistasCEV"
COL_TRAZAS = "evaluacion_trazas"

#  Vocabulario del conflicto colombiano 
# Vocabulario de respaldo (hardcodeado) — se usa cuando los CSVs del notebook
# analisisdescriptivo.ipynb aún no han sido generados.
# Una vez que el notebook corra, _cargar_vocabulario_por_rol() carga vocabularios
# específicos por rol desde corpus_ngrams_{rol}.csv, que son empíricamente más
# representativos y corrigen el desbalance de clases del corpus.
_VOCABULARIO_CONFLICTO_FALLBACK = {
    # Actores armados (como los llama la gente, no los documentos)
    "paracos", "paras", "paramilitares", "guerrilla", "guerrilleros",
    "milicianos", "tombos", "ejército", "soldados", "raspachines",
    "frente", "bloque", "columna", "grupo",
    # Hechos del conflicto
    "desplazamiento", "desplazados", "masacre", "reclutamiento", "tortura",
    "desaparición", "desaparecidos", "secuestro", "extorsión", "fumigación",
    "retorno", "retornamos", "refugio", "éxodo",
    # Geografía rural del conflicto
    "vereda", "veredas", "finca", "monte", "selva", "río", "pueblo",
    "corregimiento", "municipio", "casco", "paramo",
    # Verbos situados del conflicto
    "desplazaron", "desplazamos", "mataron", "asesinaron", "amenazaron",
    "reclutaron", "secuestraron", "quemaron", "saquearon", "llegaron",
    "huyeron", "huimos", "escapamos", "nos fuimos", "nos tocó",
    # Formas de nombrar la violencia
    "bala", "tiro", "disparo", "bomba", "mina", "granada",
    "muerto", "muertos", "cuerpo", "fosa", "tumba",
    # Instituciones y procesos de memoria
    "comisión", "verdad", "reparación", "reconocimiento", "testimonio",
    "declaración", "víctima", "victimario", "excombatiente",
}


def _cargar_vocabulario_por_rol(
    top_n_unigrams: int   = 300,
    top_n_bigrams:  int   = 150,
    top_n_trigrams: int   = 75,
    use_specificity: bool = True,
) -> dict[str, dict[int, set[str]]]:
    """
    Carga vocabulario característico POR ROL y POR TAMAÑO DE N-GRAMA desde
    los CSVs generados por analisisdescriptivo.ipynb (corpus_ngrams_{rol}.csv).

    Retorna dict[rol, dict[ngram_size, set[str]]]:
      resultado["victima"][1] -> set de unigrams de víctima
      resultado["victima"][2] -> set de bigrams de víctima
      resultado["victima"][3] -> set de trigrams de víctima

    Para unigrams (n=1): si el CSV tiene columna 'specificity_score'
    (generada por wordfreq en el notebook), se usa esa para ranking en lugar
    de tfidf puro. Esto penaliza palabras comunes en español genérico.
    Para bigrams/trigrams: se usa tfidf (ya son inherentemente específicos).

    Fallback: si los CSVs no existen, unigrams usa _VOCABULARIO_CONFLICTO_FALLBACK,
    bigrams/trigrams vacíos (comportamiento previo para unigrams).

    CAMBIO: Ahora toma los top_n directamente SIN filtro de TF-IDF, para incluir
    términos menos frecuentes pero relevantes (ej: desplazamiento, masacre, etc.)
    """
    _top_n     = {1: top_n_unigrams, 2: top_n_bigrams,  3: top_n_trigrams}
    roles      = ("victima", "victimario", "tercero")
    resultado: dict[str, dict[int, set[str]]] = {}

    for rol in roles:
        ruta = _RESULTADOS / f"corpus_ngrams_{rol}.csv"
        if not ruta.exists():
            resultado[rol] = {1: _VOCABULARIO_CONFLICTO_FALLBACK, 2: set(), 3: set()}
            continue
        try:
            df = pd.read_csv(ruta)
            resultado[rol] = {}
            for n in (1, 2, 3):
                subset = df[df["ngram_size"] == n]
                sort_col = (
                    "specificity_score"
                    if use_specificity and n == 1 and "specificity_score" in df.columns
                    else "tfidf"
                )
                top = subset.nlargest(_top_n[n], sort_col)["ngrama"].tolist()
                resultado[rol][n] = set(top) if top else set()

            if not resultado[rol].get(1):
                resultado[rol][1] = _VOCABULARIO_CONFLICTO_FALLBACK

            n1 = len(resultado[rol][1])
            n2 = len(resultado[rol].get(2, set()))
            n3 = len(resultado[rol].get(3, set()))
            print(f"   vocab '{rol}': {n1} unigrams, {n2} bigrams, {n3} trigrams (desde CSV, sin filtro TF-IDF)")
        except Exception as exc:
            print(f"   vocab '{rol}': error leyendo CSV ({exc}) — usando fallback")
            resultado[rol] = {1: _VOCABULARIO_CONFLICTO_FALLBACK, 2: set(), 3: set()}

    return resultado


# Cargado una sola vez al importar el módulo.
# Si los CSVs no existen todavía, usa fallback para unigrams y sets vacíos para n>1.
_VOCABULARIO_POR_ROL: dict[str, dict[int, set[str]]] = _cargar_vocabulario_por_rol()

# Vocabulario global para retrocompatibilidad (baseline sin rol específico)
VOCABULARIO_CONFLICTO = _VOCABULARIO_CONFLICTO_FALLBACK

#  Marcadores de oralidad (discourse markers) 
# Solo muletillas y marcadores discursivos. Se mantienen para compatibilidad
# pero ya no son la métrica principal — el conflicto vocabulary lo reemplaza.
ORAL_MARKERS = {
    # Muletillas y arranques
    "pues", "entonces", "mire", "bueno", "a ver", "fijese", "fijate",
    # Expresiones de verdad/certeza oral
    "la verdad", "yo creo", "me parece", "si mal no recuerdo",
    "es que", "o sea", "eso sí",
    # Marcadores de narrativa oral
    "ahí", "ahí fue", "eso fue", "así fue", "eso", "nos tocó",
    "tocó", "me tocó", "nos fuimos", "llegaron", "quedó",
    # Validación y cierre
    "cierto", "no más", "así",
    # Regionalismos de dificultad/situación
    "verraco", "encartado", "embejucado",
    # Pronombre genérico narrativo oral
    "uno",
}

# Stopwords básicas para vocabulario de contenido (español)
_STOPWORDS = {
    "de", "la", "el", "en", "y", "a", "que", "los", "las", "del",
    "un", "una", "con", "por", "para", "es", "se", "su", "lo", "le",
    "al", "me", "mi", "nos", "te", "tu", "si", "no", "más", "ya",
    "fue", "era", "hay", "son", "muy", "pero", "como", "cuando",
    "también", "todo", "este", "esta", "ese", "esa", "eso", "esto",
    "porque", "que", "hasta", "donde", "quien", "como",
}


# 
# FUNCIONES DE TOKENIZACIÓN Y MÉTRICAS
# 

def _tokenizar(texto: str) -> list[str]:
    """Tokeniza texto en palabras minúsculas, sin puntuación."""
    if not texto:
        return []
    texto = texto.lower()
    texto = re.sub(r"[^\w\sáéíóúüñ]", " ", texto)
    return [t for t in texto.split() if len(t) > 1]


def _vocab_contenido(tokens: list[str]) -> set[str]:
    """Vocabulario de contenido: tokens no-stopword."""
    return {t for t in tokens if t not in _STOPWORDS}


def _vocab_significativo(texto: str) -> set[str]:
    """
    Vocabulario significativo: palabras sin stopwords, >2 caracteres.

    Usado para métricas de overlap "limpio" sin ruido de palabras funcionales.

    Args:
        texto: texto a procesar

    Returns:
        Set de palabras significativas (minúsculas)
    """
    if not texto.strip():
        return set()

    tokens = _tokenizar(texto)
    return {
        t for t in tokens
        if t not in _STOPWORDS and len(t) > 2
    }


def _contar_vocab(tokens: list[str], texto_original: str, vocab_set: set) -> int:
    """Cuenta ocurrencias de palabras de un set. Bigramas en texto, unigramas en tokens."""
    texto_lower = texto_original.lower()
    count = 0
    for word in vocab_set:
        if " " in word:
            count += texto_lower.count(word)
        else:
            count += tokens.count(word)
    return count


def calcular_rag_transfer_limpio(
    actor_contexto: str,
    respuesta: str,
) -> float | None:
    """
    Overlap de vocabulario SIGNIFICATIVO entre actor_contexto y respuesta.

    Mide: de las palabras significativas (sin stopwords, >2 chars) que aparecen
    en el actor_contexto recuperado, cuántas también aparecen en la respuesta.

    Efecto puro del RAG: ¿El agente integró el material que recibió?

    Args:
        actor_contexto: fragmentos RAG recuperados
        respuesta:      respuesta del agente

    Returns:
        float (0-1): proporción de vocabulario significativo del chunk
                     que aparece en la respuesta.
        None si no hay actor_contexto o sin vocabulario significativo.
    """
    if not actor_contexto.strip() or not respuesta.strip():
        return None

    vocab_chunk = _vocab_significativo(actor_contexto)
    vocab_resp = _vocab_significativo(respuesta)

    if not vocab_chunk:
        return None

    overlap = vocab_chunk & vocab_resp
    return round(len(overlap) / len(vocab_chunk), 4)


def _extraer_seccion(actor_contexto: str, seccion_titulo: str) -> str:
    """
    Extrae una sección específica del actor_contexto.

    El LLM genera encabezados con número variable de # (## o ###), nombres largos/cortos
    (ej: "EUFEMISMOS PARA LO INNOMBRABLE" o solo "Eufemismos"). Se intenta búsqueda
    por nombre completo; si falla, por palabra clave principal.

    Args:
        actor_contexto: texto con secciones marcadas (## o ### NOMBRE)
        seccion_titulo: nombre de la sección (ej: "LÉXICO DEL CONFLICTO")

    Returns:
        Texto de la sección (sin encabezado), o string vacío si no existe.
    """
    if not actor_contexto:
        return ""

    # Intentar búsqueda por nombre completo primero
    patron = re.compile(
        r"(#{1,4})\s*(?:\d+[\.\)]\s*)?" + re.escape(seccion_titulo),
        re.IGNORECASE,
    )
    m = patron.search(actor_contexto)

    # Si falla, intentar búsqueda por palabra clave principal
    if not m:
        # Extraer palabra clave (primera palabra significativa del título)
        palabras_clave = {
            "LÉXICO": r"L[EÉ]XICO",
            "EUFEMISMO": r"EUFEMISMO",
            "EXPRESIÓN": r"EXPRESI[ÓO]N",
            "INNOMBRABLE": r"INNOMBRABLE",
        }
        for palabra in ["EUFEMISMO", "EXPRESIÓN", "LÉXICO"]:
            if palabra in seccion_titulo.upper():
                keyword = palabras_clave.get(palabra, palabra)
                patron_alt = re.compile(
                    r"(#{1,4})\s*(?:\d+[\.\)]\s*)?" + keyword,
                    re.IGNORECASE,
                )
                m = patron_alt.search(actor_contexto)
                if m:
                    break

    if m:
        # Encontró el encabezado parent — extraer normalmente
        nivel = len(m.group(1))
        inicio = m.end()

        patron_siguiente = re.compile(r"^#{1," + str(nivel) + r"}\s+\S", re.MULTILINE)
        sig = patron_siguiente.search(actor_contexto, inicio)
        if sig:
            seccion = actor_contexto[inicio : sig.start()]
        else:
            seccion = actor_contexto[inicio:]

        return seccion.strip()

    # Fallback para LÉXICO DEL CONFLICTO sin encabezado parent
    # Tomar todas las subsecciones ### hasta EUFEMISMOS o EXPRESIONES
    if "LÉXICO" in seccion_titulo.upper():
        primer_subsec = re.search(r"^###\s+", actor_contexto, re.MULTILINE)
        if primer_subsec:
            inicio = primer_subsec.start()
            # Fin: siguiente ## O una subsección que mencione EUFEMISMO/EXPRESIÓN
            fin_lexical = re.search(
                r"^(?:##\s+\S|###\s+.*(?:EUFEMISMO|EXPRESIÓN|EXPRESION))",
                actor_contexto[inicio:],
                re.MULTILINE | re.IGNORECASE
            )
            if fin_lexical:
                return actor_contexto[inicio : inicio + fin_lexical.start()].strip()
            else:
                return actor_contexto[inicio:].strip()

    return ""


def calcular_rag_transfer_lexical(
    actor_contexto: str,
    respuesta: str,
) -> float | None:
    """
    Overlap de LÉXICO DEL CONFLICTO (palabras clave) entre chunk y respuesta.

    Mide: ¿El agente integra las palabras clave específicas que aparecen
    en el material RAG?
    """
    if not actor_contexto.strip() or not respuesta.strip():
        return None

    seccion_lexico = _extraer_seccion(actor_contexto, "LÉXICO DEL CONFLICTO")
    if not seccion_lexico:
        return None

    vocab_chunk = _vocab_significativo(seccion_lexico)
    vocab_resp = _vocab_significativo(respuesta)

    if not vocab_chunk:
        return None

    overlap = vocab_chunk & vocab_resp
    return round(len(overlap) / len(vocab_chunk), 4)


def calcular_rag_transfer_eufemismos(
    actor_contexto: str,
    respuesta: str,
) -> float | None:
    """
    Overlap de EUFEMISMOS (frases indirectas) entre chunk y respuesta.

    Mide: ¿El agente integra las formas indirectas de hablar sobre violencia
    que aparecen en el material RAG?
    """
    if not actor_contexto.strip() or not respuesta.strip():
        return None

    seccion_eufemismos = _extraer_seccion(
        actor_contexto, "EUFEMISMOS PARA LO INNOMBRABLE"
    )
    if not seccion_eufemismos:
        return None

    # Buscar frases entrecomilladas; si no, buscar bullets (- líneas)
    frases = re.findall(r'"([^"]+)"', seccion_eufemismos)
    if not frases:
        # Fallback: buscar líneas que comienzan con `-` como frases
        frases = re.findall(r'^-\s+(.+)$', seccion_eufemismos, re.MULTILINE)

    if not frases:
        return None

    respuesta_lower = respuesta.lower()
    coincidencias = sum(1 for frase in frases if frase.lower() in respuesta_lower)

    return round(coincidencias / len(frases), 4)


def calcular_rag_transfer_expresiones(
    actor_contexto: str,
    respuesta: str,
) -> float | None:
    """
    Overlap de EXPRESIONES LITERALES (registro oral auténtico) entre chunk y respuesta.

    Mide: ¿El agente integra las expresiones orales auténticas que aparecen
    en el material RAG?
    """
    if not actor_contexto.strip() or not respuesta.strip():
        return None

    seccion_expr = _extraer_seccion(actor_contexto, "EXPRESIONES LITERALES")
    if not seccion_expr:
        return None

    # Buscar frases entrecomilladas; si no, buscar bullets (- líneas)
    frases = re.findall(r'"([^"]+)"', seccion_expr)
    if not frases:
        # Fallback: buscar líneas que comienzan con `-` como frases
        frases = re.findall(r'^-\s+(.+)$', seccion_expr, re.MULTILINE)

    if not frases:
        return None

    respuesta_lower = respuesta.lower()
    coincidencias = sum(1 for frase in frases if frase.lower() in respuesta_lower)

    return round(coincidencias / len(frases), 4)


def calcular_metricas_lexicas(
    actor_contexto: str,
    respuesta: str,
    rol: str | None = None,
) -> dict:
    """
    Calcula métricas de transferencia léxica para un turno.

    Args:
        actor_contexto: texto de los chunks RAG entregados al actor.
        respuesta:      respuesta generada por el actor.
        rol:            'victima', 'victimario' o 'tercero'.
                        Si se provee, usa el vocabulario característico de ese
                        rol (cargado desde corpus_ngrams_{rol}.csv).
                        Si es None, usa el vocabulario global de fallback.

    UNIVERSALES (todas las condiciones — comparables cross-condition):
      conflict_vocab_density_response -> % palabras del rol en la respuesta
      conflict_vocab_count            -> n° único de términos del rol en respuesta

    TRANSFERENCIA RAG (requieren chunks — rag_only y bdi_rag):
      rag_transfer_limpio     -> |vocab_significativo_chunks ∩ vocab_significativo_respuesta| / |vocab_significativo_chunks|
                               (efecto PURO del RAG: palabras >2 chars, sin stopwords)
      rag_transfer_lexical    -> overlap sección LÉXICO DEL CONFLICTO (core)
      rag_transfer_eufemismos -> overlap sección EUFEMISMOS (frases exactas)
      rag_transfer_expresiones-> overlap sección EXPRESIONES LITERALES (frases exactas)
    """
    # Seleccionar vocabulario específico del rol
    _fallback_n = {1: VOCABULARIO_CONFLICTO, 2: set(), 3: set()}
    vocab_por_n  = _VOCABULARIO_POR_ROL.get(rol, _fallback_n) if rol else _fallback_n
    vocab_rol    = vocab_por_n.get(1, VOCABULARIO_CONFLICTO)   # unigrams

    tokens_resp  = _tokenizar(respuesta)
    tokens_chunk = _tokenizar(actor_contexto)

    n_resp  = len(tokens_resp)  or 1

    #  Densidad de vocabulario de conflicto en respuesta
    cv_resp  = _contar_vocab(tokens_resp,  respuesta,       vocab_rol)
    conflict_density_resp  = round(cv_resp  / n_resp,  4)

    # cv_resp_set se computa aquí para estar disponible en ambas ramas (baseline y RAG)
    # Necesario para conflict_vocab_count y conflict_vocab_words_response
    cv_resp_set = {w for w in vocab_rol if respuesta.lower().count(w) > 0}

    if not actor_contexto.strip():
        # Sin chunks (baseline / bdi_only): solo métricas universales.
        # Las de transferencia RAG son None.
        return {
            # Universales
            "conflict_vocab_density_response" : conflict_density_resp,
            "conflict_vocab_count"            : len(cv_resp_set),
            "conflict_vocab_words_response"   : "|".join(sorted(cv_resp_set)),
            # Transferencia RAG — no aplica sin chunks
            "rag_transfer_limpio"             : None,
            "rag_transfer_lexical"            : None,
            "rag_transfer_eufemismos"         : None,
            "rag_transfer_expresiones"        : None,
        }

    return {
        # Universales
        "conflict_vocab_density_response" : conflict_density_resp,
        "conflict_vocab_count"            : len(cv_resp_set),
        "conflict_vocab_words_response"   : "|".join(sorted(cv_resp_set)),
        # Transferencia RAG (requieren chunks)
        "rag_transfer_limpio"             : calcular_rag_transfer_limpio(actor_contexto, respuesta),
        "rag_transfer_lexical"            : calcular_rag_transfer_lexical(actor_contexto, respuesta),
        "rag_transfer_eufemismos"         : calcular_rag_transfer_eufemismos(actor_contexto, respuesta),
        "rag_transfer_expresiones"        : calcular_rag_transfer_expresiones(actor_contexto, respuesta),
    }


# 
# LECTURA DESDE MONGODB Y CÓMPUTO
# 

def _metricas_retrieval(retrieval_metadata: dict, actor_contexto: str) -> dict:
    """
    Extrae métricas de calidad del retrieval desde metadata guardada en MongoDB.
    Si retrieval_metadata está vacío (trazas generadas antes de este cambio),
    deja los campos de retrieval en None pero calcula lexical_diversity igual.

    Campos cubiertas:
      - top_k_similarity : promedio de scores dot-product del vector search
      - scores_por_chunk : lista de scores individuales (para distribución)
      - role_purity      : % de chunks que coincidían con el rol pedido
      - n_docs_total     : chunks devueltos por el retriever antes del filtro
      - n_docs_rol       : chunks que pasaron el filtro de rol
      - usado_fallback   : True si ningún chunk coincidió y se usaron todos
      - lexical_diversity: vocab_único_contenido / tokens_contenido en actor_contexto
    """
    lex_div = _calcular_lexical_diversity(actor_contexto)

    if not retrieval_metadata:
        return {
            "top_k_similarity": None,
            "scores_por_chunk": None,
            "role_purity"     : None,
            "n_docs_total"    : None,
            "n_docs_rol"      : None,
            "usado_fallback"  : None,
            "lexical_diversity": lex_div,
        }

    return {
        "top_k_similarity": retrieval_metadata.get("top_k_similarity"),
        "scores_por_chunk": retrieval_metadata.get("scores_por_chunk"),
        "role_purity"     : retrieval_metadata.get("role_purity"),
        "n_docs_total"    : retrieval_metadata.get("n_docs_total"),
        "n_docs_rol"      : retrieval_metadata.get("n_docs_rol"),
        "usado_fallback"  : retrieval_metadata.get("usado_fallback"),
        "lexical_diversity": lex_div,
    }


def _calcular_lexical_diversity(texto: str) -> float | None:
    """Vocabulario único de contenido / total de tokens de contenido."""
    if not texto.strip():
        return None
    tokens = _tokenizar(texto)
    vocab  = _vocab_contenido(tokens)
    if not tokens:
        return None
    return round(len(vocab) / len(tokens), 4)


def calcular_todas_las_metricas() -> pd.DataFrame:
    """Lee evaluacion_trazas y devuelve DataFrame con métricas léxicas."""
    client = MongoClient(MONGO_URI)
    db     = client[MONGO_DB]

    trazas = list(db[COL_TRAZAS].find({}, {"_id": 0}))
    client.close()

    if not trazas:
        raise RuntimeError(
            "No hay trazas en evaluacion_trazas. "
            "Ejecuta generador_trazas.py primero."
        )

    filas = []
    for t in trazas:
        actor_contexto      = t.get("actor_contexto", "")      or ""
        respuesta           = t.get("respuesta", "")            or ""
        retrieval_metadata  = t.get("retrieval_metadata", {})   or {}

        metricas_lex  = calcular_metricas_lexicas(actor_contexto, respuesta, rol=t.get("actor_id"))
        metricas_retr = _metricas_retrieval(retrieval_metadata, actor_contexto)

        filas.append({
            "condicion"           : t.get("condicion"),
            "actor_id"            : t.get("actor_id"),
            "turno"               : t.get("turno"),
            "tipologia"           : t.get("tipologia"),
            "n_palabras_chunks"   : len(_tokenizar(actor_contexto)),
            "n_palabras_respuesta": len(_tokenizar(respuesta)),
            **metricas_lex,
            **metricas_retr,
        })

    df = pd.DataFrame(filas)
    df = df.sort_values(["condicion", "actor_id", "turno"]).reset_index(drop=True)
    return df


# 
# RESUMEN Y EXPORTACIÓN
# 

#  Universales: comparables entre las 4 condiciones 
_METRICAS_UNIVERSALES_LEX = [
    "conflict_vocab_density_response",  # % vocab del rol en respuesta (relativo)
    "conflict_vocab_count",             # n° único de términos del rol (absoluto)
]

#  Transferencia RAG: solo rag_only y bdi_rag (requieren chunks)
#  Nota: Incluye las métricas de transfer completas para análisis detallado.
#  En el MD de la tesis se reportan solo: rag_transfer_lexical, y vocab_novelty
_METRICAS_TRANSFERENCIA_RAG = [
    "rag_transfer_limpio",          # overlap vocab significativo (actor_contexto completo)
    "rag_transfer_lexical",         # overlap sección LÉXICO DEL CONFLICTO (core)
    "rag_transfer_eufemismos",      # overlap sección EUFEMISMOS (frases exactas)
    "rag_transfer_expresiones",     # overlap sección EXPRESIONES LITERALES (frases exactas)
]

#  Calidad del retrieval: propiedades del chunk, no de la transferencia
#  Nota: Solo se reportan top_k_similarity y role_purity en el MD
_METRICAS_CALIDAD_RETRIEVAL = [
    "top_k_similarity",               # similitud dot-product de retrieval (core)
    "role_purity",                    # % chunks del rol correcto
]

# Lista combinada para resumen CSV (compatibilidad con exportar_csvs y generar_grafico)
_METRICAS_TRANSFERENCIA = (
    _METRICAS_UNIVERSALES_LEX
    + _METRICAS_TRANSFERENCIA_RAG
    + _METRICAS_CALIDAD_RETRIEVAL
)

# conflict_vocab_words_response es string (no numérico) — va al CSV detalle pero no al resumen


def calcular_vocab_novelty(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula vocab_novelty por actor_id x condición RAG.

    Definición:
        vocab_novelty = |vocab_rol_RAG - vocab_rol_baseline| / |vocab_rol_RAG|

    Es decir: qué fracción del vocabulario de conflicto que usa el actor en la
    condición RAG NO aparece en sus respuestas baseline.

    Responde la pregunta central: "¿qué vocabulario introduce el RAG que
    GPT-4o-mini no usaría solo?"

    Un valor de 0.0 significa que el RAG no aporta vocabulario nuevo.
    Un valor de 1.0 significa que todo el vocabulario de conflicto en la respuesta
    RAG es exclusivo de esa condición.

    Requiere la columna 'conflict_vocab_words_response' (palabras pipe-separadas)
    generada por calcular_metricas_lexicas().

    Retorna un DataFrame con columnas: actor_id, condicion, vocab_novelty
    """
    if "conflict_vocab_words_response" not in df.columns:
        return pd.DataFrame(columns=["actor_id", "condicion", "vocab_novelty"])

    def palabras(serie: pd.Series) -> set:
        """Une todos los sets de palabras de una serie de strings pipe-separados."""
        palabras_set = set()
        for val in serie.dropna():
            if val:
                palabras_set.update(val.split("|"))
        return palabras_set - {""}

    # Vocabulario baseline por actor (unión de todos los turnos)
    baseline_df = df[df["condicion"] == "baseline"]
    vocab_baseline: dict[str, set] = {
        actor: palabras(grupo["conflict_vocab_words_response"])
        for actor, grupo in baseline_df.groupby("actor_id")
    }

    filas = []
    for (actor, condicion), grupo in df[df["condicion"] != "baseline"].groupby(["actor_id", "condicion"]):
        vocab_rag      = palabras(grupo["conflict_vocab_words_response"])
        vocab_base     = vocab_baseline.get(actor, set())
        vocab_exclusivo = vocab_rag - vocab_base

        novelty = (
            round(len(vocab_exclusivo) / len(vocab_rag), 4)
            if vocab_rag else None
        )
        filas.append({
            "actor_id"          : actor,
            "condicion"         : condicion,
            "vocab_novelty"     : novelty,
            "n_terminos_rag"    : len(vocab_rag),
            "n_terminos_nuevos" : len(vocab_exclusivo),
            "terminos_nuevos"   : ", ".join(sorted(vocab_exclusivo)[:20]),  # top-20 para auditoría
        })

    return pd.DataFrame(filas)


def exportar_csvs(df: pd.DataFrame) -> None:
    _RESULTADOS.mkdir(exist_ok=True)

    # Datos granulares (incluye conflict_vocab_words_response para auditoría)
    path_detalle = _RESULTADOS / "metricas_lexicas_por_turno.csv"
    df.to_csv(path_detalle, index=False)
    print(f"  [OK] {path_detalle.name}")

    # Resumen por condicion x actor_id (solo métricas numéricas)
    resumen = (
        df.groupby(["condicion", "actor_id"])[_METRICAS_TRANSFERENCIA]
        .mean()
        .round(4)
        .reset_index()
    )

    # Agregar vocab_novelty al resumen — requiere comparación cruzada entre condiciones
    novelty_df = calcular_vocab_novelty(df)
    if not novelty_df.empty:
        resumen = resumen.merge(
            novelty_df[["actor_id", "condicion", "vocab_novelty",
                         "n_terminos_rag", "n_terminos_nuevos", "terminos_nuevos"]],
            on=["actor_id", "condicion"],
            how="left",
        )
        # Exportar tabla de novelty por separado para auditoría de términos
        novelty_df.to_csv(_RESULTADOS / "vocab_novelty.csv", index=False)
        print(f"  [OK] vocab_novelty.csv")

    path_resumen = _RESULTADOS / "metricas_lexicas_resumen.csv"
    resumen.to_csv(path_resumen, index=False)
    print(f"  [OK] {path_resumen.name}")

    # Resumen solo por condicion
    resumen_cond = (
        df.groupby("condicion")[_METRICAS_TRANSFERENCIA]
        .mean()
        .round(4)
        .reset_index()
    )
    if not novelty_df.empty:
        nov_cond = novelty_df.groupby("condicion")[["vocab_novelty"]].mean().round(4).reset_index()
        resumen_cond = resumen_cond.merge(nov_cond, on="condicion", how="left")

    print("\nRESUMEN POR CONDICION:")
    print(resumen_cond.to_string(index=False))


def generar_grafico(df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("    matplotlib no disponible. Omitiendo gráfico.")
        return

    metricas_plot = [
        ("conflict_vocab_density_response", "Conflict Vocab Density\n(vocab conflicto / palabras — universal)"),
        ("conflict_vocab_count",            "Conflict Vocab Count\n(términos únicos del rol — universal)"),
        ("rag_transfer_lexical",            "RAG Transfer Lexical\n(overlap LÉXICO DEL CONFLICTO — RAG)"),
        ("top_k_similarity",                "Top-K Similarity\n(similitud retrieval — RAG)"),
    ]

    condiciones  = ["baseline", "rag_only", "bdi_only", "bdi_rag"]
    colores      = {
        "baseline": "#95a5a6",
        "rag_only": "#3498db",
        "bdi_only": "#2ecc71",
        "bdi_rag" : "#e74c3c",
    }
    actores      = ["victima", "victimario", "tercero"]

    fig, axes = plt.subplots(1, len(metricas_plot), figsize=(16, 5))
    fig.suptitle("Métricas de Transferencia Léxica del RAG", fontsize=14, fontweight="bold")

    for ax, (metrica, titulo) in zip(axes, metricas_plot):
        x      = np.arange(len(actores))
        width  = 0.20   # 4 condiciones — barras más angostas
        offset = 0

        for cond in condiciones:
            sub  = df[df["condicion"] == cond]
            vals = []
            for actor in actores:
                v = sub[sub["actor_id"] == actor][metrica].mean()
                vals.append(v if pd.notna(v) else 0)

            ax.bar(x + offset, vals, width,
                   label=cond, color=colores[cond], alpha=0.85)
            offset += width

        ax.set_title(titulo, fontsize=10)
        ax.set_xticks(x + width * 1.5)   # centro del grupo de 4 barras
        ax.set_xticklabels(actores, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Valor promedio")
        ax.grid(axis="y", alpha=0.3)

    parches = [
        mpatches.Patch(color=colores[c], label=c, alpha=0.85)
        for c in condiciones
    ]
    fig.legend(handles=parches, loc="lower center", ncol=4, fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    path_img = _RESULTADOS / "transferencia_lexica.png"
    plt.savefig(path_img, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {path_img.name}")


# 
# INTERPRETACIÓN AUTOMÁTICA
# 

def interpretar(df: pd.DataFrame) -> None:
    """
    Imprime una interpretación automática de los resultados para facilitar
    su redacción en la tesis.
    """
    print("\n" + "" * 64)
    print("  INTERPRETACIÓN AUTOMÁTICA")
    print("" * 64)

    cond_media = df.groupby("condicion")[
        [m for m in _METRICAS_TRANSFERENCIA if m in df.columns]
    ].mean()

    # 0. top_k_similarity (calidad del retrieval — independiente de la respuesta)
    if "top_k_similarity" in cond_media.columns:
        sim_rag = cond_media.loc["rag_only",  "top_k_similarity"] if "rag_only"  in cond_media.index else None
        sim_bdi = cond_media.loc["bdi_rag",   "top_k_similarity"] if "bdi_rag"   in cond_media.index else None

        if sim_rag is not None and not np.isnan(sim_rag):
            print(f"\n  top_k_similarity (dot product promedio):")
            print(f"    rag_only = {sim_rag:.4f} | bdi_rag = {sim_bdi:.4f}")
            if sim_rag > 0.5:
                print("    -> El RAG recupera chunks semánticamente relevantes (sim > 0.5).")
            elif sim_rag > 0.3:
                print("    -> Similitud moderada — los chunks son pertinentes pero no altamente específicos.")
            else:
                print("    -> Similitud baja — considerar ajustar la query construction o el corpus.")

            if sim_bdi is not None and not np.isnan(sim_bdi) and abs(sim_rag - sim_bdi) > 0.03:
                if sim_bdi > sim_rag:
                    print("    -> bdi_rag recupera chunks más similares que rag_only.")
                    print("      El BDI mejora la especificidad de las queries al tener más contexto.")
                else:
                    diff = round((sim_rag - sim_bdi) / sim_rag * 100, 1)
                    print(f"    -> rag_only es {diff}% más similar que bdi_rag.")
                    print("      Posible causa: micro_intencion BDI genera queries más abstractas.")
        else:
            print("\n  top_k_similarity: datos no disponibles (requiere nueva corrida)")

    # 1. conflict_vocab_density (comparación directa con baseline)
    if "conflict_vocab_density_response" in cond_media.columns:
        cvd_base = cond_media.loc["baseline", "conflict_vocab_density_response"] if "baseline" in cond_media.index else None
        cvd_rag  = cond_media.loc["rag_only",  "conflict_vocab_density_response"] if "rag_only"  in cond_media.index else None
        cvd_bdi  = cond_media.loc["bdi_rag",   "conflict_vocab_density_response"] if "bdi_rag"   in cond_media.index else None

        print(f"\n  conflict_vocab_density_response (densidad vocab conflicto en respuesta):")
        print(f"    baseline = {cvd_base:.4f} | rag_only = {cvd_rag:.4f} | bdi_rag = {cvd_bdi:.4f}")

        mejor = max(
            [("baseline", cvd_base), ("rag_only", cvd_rag), ("bdi_rag", cvd_bdi)],
            key=lambda x: x[1] if x[1] is not None and not np.isnan(x[1]) else -1
        )
        print(f"    -> Mayor densidad de vocabulario del conflicto: {mejor[0]}")
        if mejor[0] != "baseline":
            print("      [OK] El RAG sí aporta vocabulario específico del conflicto.")
        else:
            print("      [X] El modelo base genera más vocab del conflicto sin RAG — revisar corpus o summarización.")

    # 2. conflict_vocab_count vs density — desambiguar efecto longitud de respuesta
    if "conflict_vocab_count" in cond_media.columns:
        cvc_base = cond_media.loc["baseline", "conflict_vocab_count"] if "baseline" in cond_media.index else None
        cvc_rag  = cond_media.loc["rag_only",  "conflict_vocab_count"] if "rag_only"  in cond_media.index else None
        cvc_bdi  = cond_media.loc["bdi_rag",   "conflict_vocab_count"] if "bdi_rag"   in cond_media.index else None

        print(f"\n  conflict_vocab_count (terminos unicos del rol en respuesta — absoluto):")
        print(f"    baseline = {cvc_base:.1f} | rag_only = {cvc_rag:.1f} | bdi_rag = {cvc_bdi:.1f}")

        cvd_base = cond_media.loc["baseline", "conflict_vocab_density_response"] if "baseline" in cond_media.index else None
        cvd_rag  = cond_media.loc["rag_only",  "conflict_vocab_density_response"] if "rag_only"  in cond_media.index else None
        cvd_bdi  = cond_media.loc["bdi_rag",   "conflict_vocab_density_response"] if "bdi_rag"   in cond_media.index else None

        # Diagnostico: si count sube pero density baja, el RAG solo alarga la respuesta
        if (cvc_rag is not None and cvc_base is not None and
                not np.isnan(cvc_rag) and not np.isnan(cvc_base)):
            if cvc_rag > cvc_base and cvd_rag is not None and cvd_rag < cvd_base:
                print("      RAG: count sube pero density baja -> el RAG alarga las respuestas")
                print("       más de lo que agrega vocabulario situado. Revisar chunking o prompt.")
            elif cvc_rag > cvc_base and cvd_rag is not None and cvd_rag >= cvd_base:
                print("    [OK]  RAG añade más términos del conflicto Y mantiene o mejora la densidad.")

    # 3. vocab_novelty — métrica central de aporte del RAG
    novelty_df = calcular_vocab_novelty(df)
    if not novelty_df.empty:
        print(f"\n  vocab_novelty (% vocab RAG que NO aparece en baseline — aporte exclusivo del RAG):")
        for _, fila in novelty_df.iterrows():
            print(f"    {fila['condicion']:10s} | {fila['actor_id']:12s}: "
                  f"{fila['vocab_novelty']:.3f}  "
                  f"({fila['n_terminos_nuevos']}/{fila['n_terminos_rag']} terminos nuevos)")
            if fila["terminos_nuevos"]:
                print(f"      Ejemplos: {fila['terminos_nuevos']}")

        nov_media = novelty_df.groupby("condicion")["vocab_novelty"].mean()
        if "rag_only" in nov_media.index and "bdi_rag" in nov_media.index:
            if nov_media["bdi_rag"] > nov_media["rag_only"]:
                print("    -> BDI+RAG introduce vocabulario más exclusivo que RAG solo.")
                print("      El contexto BDI dirige al RAG hacia términos más específicos del rol.")
            else:
                diff = round((nov_media["rag_only"] - nov_media["bdi_rag"]) * 100, 1)
                print(f"    -> RAG solo introduce {diff}pp más vocabulario nuevo que BDI+RAG.")

    print("\n" + "" * 64)


# 
# PUNTO DE ENTRADA
# 

def main(con_grafico: bool = True) -> None:
    print("Calculando metricas de transferencia lexica...")

    df = calcular_todas_las_metricas()
    print(f"   {len(df)} trazas cargadas.\n")

    print("Exportando CSVs:")
    exportar_csvs(df)

    if con_grafico:
        print("\nGenerando grafico:")
        generar_grafico(df)

    interpretar(df)
    print("\n Listo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Métricas de transferencia léxica del RAG — EntrevistasCEV"
    )
    parser.add_argument(
        "--sin-grafico",
        action="store_true",
        help="Omitir generación del gráfico matplotlib",
    )
    args = parser.parse_args()
    main(con_grafico=not args.sin_grafico)
