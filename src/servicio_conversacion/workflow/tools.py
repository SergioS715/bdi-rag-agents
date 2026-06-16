from langchain_core.tools import tool
from rag.retrievers import obtener_recuperador
from config import RAG_TEXT_EMBEDDING_MODEL_ID, RAG_TOP_K, RAG_DEVICE

recuperador = obtener_recuperador(
    id_modelo_embedding=RAG_TEXT_EMBEDDING_MODEL_ID,
    k=RAG_TOP_K,
    dispositivo=RAG_DEVICE
)

# Vector store directo — lo usamos con pre_filter para garantizar que los
# chunks recuperados sean del rol correcto. El hybrid retriever de Atlas
# no expone pre_filter de forma estable entre versiones, así que hacemos
# la búsqueda vectorial con filtro duro y nos apoyamos en dotProduct.
_vectorstore = recuperador.vectorstore

# ─── Metadata del último retrieval (para métricas de calidad) ────────────────
_ultimo_retrieval_metadata: dict = {}


def get_ultimo_retrieval_metadata() -> dict:
    """
    Devuelve la metadata del último retrieval ejecutado.
    Campos:
      - n_docs_total:      chunks recuperados con el pre-filtro de rol activo
      - n_docs_rol:        igual a n_docs_total (el filtro es estricto por rol_llm)
      - role_purity:       1.0 si hubo matches, 0.0 si no se encontró nada del rol
      - usado_fallback:    True si el pre-filtro devolvió vacío y el sistema
                           no tiene chunks del rol (no se mezclan otros roles)
      - roles_encontrados: lista de rol_llm de los chunks devueltos
      - top_k_similarity:  promedio dot-product de los k chunks
      - scores_por_chunk:  scores individuales
    """
    return dict(_ultimo_retrieval_metadata)


# Mapa de equivalencias: rol del sistema → valores que aparecen en rol_llm en la BD
# rol_llm se llenó con el output del clasificador LLM en procesamientoTexto.ipynb
# y puede tener variantes ortográficas. Filtramos por todos los alias del rol.
_ALIAS_POR_ROL: dict[str, list[str]] = {
    "victima"   : ["victima", "víctima"],
    "victimario": ["victimario", "actor_armado"],
    "tercero"   : ["tercero"],
}


def _alias_rol(rol: str) -> list[str]:
    """Devuelve todos los alias válidos en rol_llm para un rol del sistema."""
    return _ALIAS_POR_ROL.get(rol.strip().lower(), [rol.strip().lower()])


def _construir_pre_filter(rol: str) -> dict:
    """
    Construye el pre_filter para MongoDB Atlas Vector Search (formato `fields`).
    Solo se usa si el índice es tipo `vectorSearch`. Si es Atlas Search clásico
    (`mappings`), esta llamada falla y caemos al post-filtro de _post_filtrar_por_rol.
    """
    alias = _alias_rol(rol)
    return {
        "$or": [
            {"rol_llm"            : {"$in": alias}},
            {"clasificacion_final": {"$in": alias}},
        ]
    }


def _doc_es_del_rol(doc, rol: str) -> bool:
    """True si el doc tiene rol_llm o clasificacion_final coincidente con el rol."""
    alias = set(_alias_rol(rol))
    rol_doc = (doc.metadata.get("rol_llm") or
               doc.metadata.get("clasificacion_final") or "").strip().lower()
    return rol_doc in alias


# Multiplicador de k para post-filtrado: pedimos K_AMPLIADO chunks sin filtro
# y nos quedamos con los primeros RAG_TOP_K que sean del rol correcto.
# 8x suele bastar para roles minoritarios; aumentar si el corpus está muy sesgado.
_FACTOR_EXPANSION_K = 8


def _expandir_query(query: str) -> str:
    """
    Devuelve el query tal como lo construyó el LLM, sin prefijo estático.
    El filtro por rol en _buscar_con_filtro ya garantiza chunks del rol correcto,
    y el prefijo fijo diluía la señal semántica específica de cada búsqueda.
    """
    return query.strip()


# Flag de runtime: una vez que confirmamos que el índice no soporta pre_filter,
# no volvemos a intentarlo en futuras llamadas (evita ruido en consola).
_pre_filter_soportado: bool | None = None


def _buscar_con_filtro(query_expandida: str, rol: str, k: int) -> tuple[list, list[float]]:
    """
    Devuelve (docs, scores) garantizando que TODOS los docs sean del rol pedido.

    Estrategia:
      1. Si el índice soporta pre_filter (Atlas Vector Search nuevo) → lo usamos.
      2. Si no (Atlas Search clásico con mappings) → pedimos k * _FACTOR_EXPANSION_K
         chunks sin filtro y filtramos en cliente, conservando los k mejores
         del rol correcto.

    Nunca mezcla roles. Si no hay chunks del rol, devuelve listas vacías.
    """
    global _pre_filter_soportado

    # Camino 1 — pre_filter nativo (más eficiente, requiere índice vectorSearch)
    if _pre_filter_soportado is not False:
        try:
            score_docs = _vectorstore.similarity_search_with_score(
                query_expandida,
                k=k,
                pre_filter=_construir_pre_filter(rol),
            )
            _pre_filter_soportado = True
            if not score_docs:
                return [], []
            docs   = [d for d, _ in score_docs]
            scores = [round(float(s), 4) for _, s in score_docs]
            return docs, scores
        except Exception as exc:
            if _pre_filter_soportado is None:
                # Primera vez que falla — informar y desactivar para futuras llamadas
                print(f"ℹ  pre_filter no soportado por el índice actual ({type(exc).__name__}). "
                      f"Usando post-filtrado en cliente con k expandido.")
            _pre_filter_soportado = False

    # Camino 2 — post-filtrado: pedimos más, filtramos por rol en cliente
    k_ampliado = k * _FACTOR_EXPANSION_K
    try:
        score_docs = _vectorstore.similarity_search_with_score(
            query_expandida,
            k=k_ampliado,
        )
    except Exception as exc:
        print(f"⚠  similarity_search_with_score falló: {exc}")
        return [], []

    if not score_docs:
        return [], []

    # Filtrar conservando orden por similitud
    filtrados = [(d, s) for d, s in score_docs if _doc_es_del_rol(d, rol)]
    if not filtrados:
        return [], []

    filtrados = filtrados[:k]
    docs   = [d for d, _ in filtrados]
    scores = [round(float(s), 4) for _, s in filtrados]
    return docs, scores


@tool
def recuperar_contexto_actor(query: str, rol: str = "") -> str:
    """Recupera fragmentos de testimonios reales del conflicto armado colombiano
    del MISMO ROL del personaje para enriquecer la respuesta con vocabulario,
    emociones y formas de expresión auténticas. Úsala en casi todas las respuestas
    — los fragmentos aportan registro oral, tono emocional y vocabulario del
    conflicto que hacen el testimonio auténtico. Solo omítela para saludos o
    frases puramente sociales ("hola", "gracias", "adiós"). Para cualquier otra
    cosa — emociones, recuerdos, opiniones, hechos, familia, vida cotidiana,
    violencia, perdón — úsala siempre. Los fragmentos contienen lenguaje oral
    real con descripciones de hechos del conflicto — ese es su valor. Absórbelos
    para sonar como alguien que vivió eso.

    Args:
        query: Descripción de lo que buscas en términos del relato.
               Ejemplos: 'cómo expresa emociones una víctima del desplazamiento',
               'vocabulario de excombatiente hablando de operaciones',
               'cómo habla un testigo sobre pérdidas en su comunidad'.
        rol:   Rol del actor actual ('victima', 'victimario' o 'tercero').
               OBLIGATORIO — es lo que garantiza que los fragmentos sean del
               mismo rol y no se mezclen testimonios entre perfiles.
    """
    global _ultimo_retrieval_metadata

    rol_norm = rol.strip().lower()
    query_expandida = _expandir_query(query)

    docs, scores = _buscar_con_filtro(query_expandida, rol_norm, k=RAG_TOP_K)
    top_k_sim = round(sum(scores) / len(scores), 4) if scores else None

    if not docs:
        _ultimo_retrieval_metadata = {
            "n_docs_total"     : 0,
            "n_docs_rol"       : 0,
            "role_purity"      : 0.0,
            "usado_fallback"   : True,   # no había chunks del rol — sin mezclar
            "roles_encontrados": [],
            "top_k_similarity" : top_k_sim,
            "scores_por_chunk" : scores,
        }
        return "No se encontró información relevante."

    roles_encontrados = [
        d.metadata.get("rol_llm") or d.metadata.get("clasificacion_final", "")
        for d in docs
    ]

    # Como el filtro es estricto, role_purity == 1.0 por construcción. La dejamos
    # por compatibilidad con la evaluación, pero ahora refleja "hubo matches".
    _ultimo_retrieval_metadata = {
        "n_docs_total"     : len(docs),
        "n_docs_rol"       : len(docs),
        "role_purity"      : 1.0,
        "usado_fallback"   : False,
        "roles_encontrados": roles_encontrados,
        "top_k_similarity" : top_k_sim,
        "scores_por_chunk" : scores,
    }

    print(f"   📐 top_k_similarity={top_k_sim}  n_docs={len(docs)}  rol={rol_norm}")

    fragmentos = []
    for i, d in enumerate(docs, 1):
        fragmentos.append(f"--- FRAGMENTO DE TESTIMONIO {i} ---\n{d.page_content}")
    return "\n\n".join(fragmentos)


def _limpiar_tokens_anonimizacion(texto: str) -> str:
    """
    Elimina tokens de anonimización y las líneas que quedan vacías tras la limpieza.
    Maneja: {TOKEN}, {{TOKEN}}, [TOKEN], [[TOKEN]], y palabras TODO-MAYÚSCULAS conocidas.
    """
    import re

    # 1. Eliminar {TOKEN} y {{TOKEN}}
    texto = re.sub(r'\{+[^{}]*\}+', '', texto)
    # 2. Eliminar [TOKEN] y [[TOKEN]] — corchetes simples o dobles
    texto = re.sub(r'\[+[^\]]*\]+', '', texto)
    # 3. Eliminar MAYUSCULA+NÚMERO (ej: PARAMILITAR1, RANGO2)
    texto = re.sub(r'\b[A-Z_][A-Z0-9_]*\d+\b', '', texto)
    # 4. Eliminar palabras TODO-MAYÚSCULAS que son tokens de anonimización conocidos
    tokens_conocidos = r'\b(LUGAR|PERSONA|VEREDA|MUNICIPIO|ORGANIZACION|ORG_SOCIAL|RANGO|ACTOR|GRUPO|OPERACION|FECHA)\b'
    texto = re.sub(tokens_conocidos, '', texto)

    # 5. Filtrar líneas que quedaron inútiles tras la limpieza:
    #    líneas vacías, solo puntuación/espacios, o bullet points sin contenido real
    lineas_limpias = []
    for linea in texto.splitlines():
        linea_sin_puntuacion = re.sub(r'[\s\-\*\:\"\'\(\)\.,;]+', '', linea)
        if linea_sin_puntuacion:
            lineas_limpias.append(re.sub(r'\s+', ' ', linea).strip())

    return '\n'.join(lineas_limpias)


herramientas = [recuperar_contexto_actor]