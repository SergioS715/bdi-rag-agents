"""
micro_bdi.py
============
Implementa la capa Micro-BDI (§3.5 del paper Harada & Kano, 2025),
adaptada al contexto del proyecto EntrevistasCEV.

Opera CADA TURNO, justo antes de que conversation_node genere la respuesta.

Flujo por turno:
    último mensaje recibido
        → §3.5.1 analizar_turno()              clasificar tipo + credibilidad bruta
        → §3.5.2 actualizar_micro_creencia()   tension, afinidad, conteos
        → §3.5.3 generar_micro_deseo()         elegir etapa_discusion CEV
        → §3.5.4 generar_micro_intencion()     YAML {estructura, contenido}
        → ResultadoMicroBDI                    se escribe en EstadoActor

Adaptaciones respecto al paper:
    - Tipos de utterance: co/neg/pos/question/null
      → acusacion/reconocimiento/pregunta/evasion/neutro
    - discussion_stage: self_intro/sharing/reasoning/persuasion/voting
      → apertura_narrativa/defensa_posicion/negociacion_verdad/
        confrontacion/cierre_simbolico
    - self_co / seer_co (AIWolf)
      → nivel_tension / conteo_negaciones / reconocimiento_dado (CEV)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from openai import OpenAI

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

from sabiduria_convencional import (
    CasoDiscursivo,
    obtener_casos_por_situacion,
    seleccionar_caso_para_turno,
)

logger = logging.getLogger(__name__)

_MODELO_MICRO = config.OPENAI_LLM_MODEL_BDI

_USAR_MICRO_UNIFICADO = True

# ── Constantes de clasificación ───────────────────────────────────────────────

TIPOS_UTTERANCE = [
    "acusacion",      # señala responsabilidad directamente
    "reconocimiento", # valida o reconoce algo del relato del otro
    "pregunta",       # pide información o explicación
    "evasion",        # desvía o ignora lo dicho antes
    "neutro",         # continúa sin tomar posición clara
]

DISCUSSION_STAGES = [
    "apertura_narrativa",  # presentar el relato propio
    "defensa_posicion",    # proteger la versión ante cuestionamiento
    "negociacion_verdad",  # buscar terreno común con el otro
    "confrontacion",       # señalar directamente responsabilidad o daño
    "cierre_simbolico",    # buscar conclusión o reconocimiento final
]

# El LLM responde con un código y lo mapeamos al nombre real.
# Evita que vocabulario como "negociacion_verdad" o "defensa_posicion" se
# filtre al registro oral del personaje.
_CODIGO_A_ETAPA: dict[str, str] = {
    "A": "apertura_narrativa",
    "B": "defensa_posicion",
    "C": "negociacion_verdad",
    "D": "confrontacion",
    "E": "cierre_simbolico",
}
_ETAPA_A_CODIGO: dict[str, str] = {v: k for k, v in _CODIGO_A_ETAPA.items()}

# Mapeo: etapa_discusion → tags de situacion en sabiduria_convencional
# Permite recuperar los patrones discursivos relevantes para cada etapa
_MAPEO_ETAPA_SITUACION: dict[str, list[str]] = {
    "apertura_narrativa": [
        "inicio_conversacion",
        "relato_desplazamiento",
        "relato_ingreso_al_grupo",
        "hablar_por_muertos",
    ],
    "defensa_posicion": [
        "pregunta_responsabilidad",
        "pregunta_accion_especifica",
        "pregunta_comprometedora",
        "quiebre_emocional",
        "relato_accion_violenta",
        "cuestionamiento_posicion",
    ],
    "negociacion_verdad": [
        "confrontacion_directa",
        "relato_accion_violenta",
        "diferencias_territoriales",
        "pregunta_magnitud_conflicto",
    ],
    "confrontacion": [
        "confrontacion_directa",
        "pregunta_responsabilidad",
        "acusacion_directa",
    ],
    "cierre_simbolico": [
        "pregunta_perdon",
        "cierre_narrativo",
        "hablar_presente_futuro",
        "pregunta_proceso_paz",
    ],
}

# ── Validación de coherencia: categorías de intención → etapas compatibles ────
# Sin hardcoding de strings específicos. Basado en semántica genérica.

_CATEGORIA_INTENCION_A_ETAPAS_COMPATIBLES: dict[str, list[str]] = {
    # Intenciones justificatorias/aclaratorias (explicar contexto, presión, circunstancias)
    "justificacion": [
        "defensa_posicion",      # defender tu posición explicando contexto
        "negociacion_verdad",    # buscar entendimiento del contexto
    ],
    # Intenciones confrontacionales (señalar responsabilidad, llamar a cuenta)
    "confrontacion": [
        "confrontacion",         # confrontar directamente
        "defensa_posicion",      # defender señalando responsabilidad del otro
    ],
    # Intenciones mediadora/reconciliatoria (buscar balance, reconocimiento mutuo)
    "mediacion": [
        "negociacion_verdad",    # buscar terreno común
        "cierre_simbolico",      # buscar reconocimiento/cierre
    ],
    # Intenciones narrativas (abrir relato, contar historia)
    "narrativa": [
        "apertura_narrativa",    # presentar el relato
        "cierre_simbolico",      # cerrar el relato
    ],
    # Intenciones de validación (reconocer, validar el dolor ajeno)
    "validacion": [
        "negociacion_verdad",    # reconocer la verdad del otro
        "cierre_simbolico",      # validar con dignidad
    ],
}


def _categorizar_intencion(micro_intencion: str) -> str:
    """
    Detecta la categoría semántica de una intención sin regex hardcodeado.
    Busca palabras clave genéricas que definen la CATEGORÍA, no la intención específica.

    Retorna categoría: "justificacion", "confrontacion", "mediacion", "narrativa", "validacion"
    """
    intencion_lower = micro_intencion.lower()

    # Palabras clave genéricas por categoría (no hardcodeadas a un string específico)
    señales = {
        "justificacion": {
            "se entienda", "se sepa que", "entienda que", "explique", "razon", "presion",
            "necesidad", "circunstancia", "contexto", "obligado", "fui obligado"
        },
        "confrontacion": {
            "confrontar", "responsabilidad", "señalar", "culpable", "culpa", "hizo daño",
            "fue culpa", "es responsable", "debo señalar"
        },
        "mediacion": {
            "ambas partes", "todos", "dolor de todos", "reconocer", "balance", "equil",
            "puente", "entendimiento mutuo", "las dos"
        },
        "narrativa": {
            "pasó", "sucedió", "ocurrió", "fue", "estaba", "vimos", "contarle", "relato",
            "historia", "cuenten"
        },
        "validacion": {
            "reconocer", "validar", "valer", "importante", "merece", "digno", "honra",
            "respeto", "honor"
        },
    }

    # Scoring: categoría con mayor overlap
    puntos = {}
    for categoria, palabras_clave in señales.items():
        overlap = sum(1 for palabra in palabras_clave if palabra in intencion_lower)
        if overlap > 0:
            puntos[categoria] = overlap

    if puntos:
        return max(puntos, key=puntos.get)
    else:
        return "narrativa"  # fallback genérico


def validar_coherencia_etapa_intencion(
    etapa_elegida: str,
    micro_intencion: str,
) -> str:
    """
    Valida si hay conflicto entre la etapa elegida y la intención.
    Si hay conflicto (etapa no compatible), retorna etapa sugerida.
    Si no hay conflicto, retorna la etapa original.

    Basado en categorización semántica, no en hardcoding de strings.
    """
    # Categorizar la intención
    categoria = _categorizar_intencion(micro_intencion)

    # Obtener etapas compatibles con esa categoría
    etapas_compatibles = _CATEGORIA_INTENCION_A_ETAPAS_COMPATIBLES.get(categoria, DISCUSSION_STAGES)

    # Si etapa elegida está en compatibles, mantenerla
    if etapa_elegida in etapas_compatibles:
        return etapa_elegida

    # Si no está, retornar la primera compatible
    if etapas_compatibles:
        etapa_sugerida = etapas_compatibles[0]
        logger.debug(
            f"[MicroBDI Coherencia] Conflicto detectado: "
            f"etapa='{etapa_elegida}' + intención_categoria='{categoria}' → "
            f"compatibles={etapas_compatibles}, sugerida='{etapa_sugerida}'"
        )
        return etapa_sugerida

    # Fallback: retornar original si algo falla
    return etapa_elegida


# Pesos de cada situacion por dimensión de MicroCreencia.
# Tupla: (tension_weight, afinidad_weight, acusaciones_weight, reconocimiento_weight)
# Positivo = la situacion encaja MEJOR cuando esa dimensión es alta.
# Negativo = la situacion encaja MEJOR cuando esa dimensión es baja.
_SITUACION_PESOS: dict[str, tuple[float, float, float, float]] = {
    "inicio_conversacion":        (-1.0,  0.5,  0.0,  0.0),
    "relato_desplazamiento":      (-0.3,  0.3,  0.0,  0.0),
    "relato_ingreso_al_grupo":    (-0.3,  0.2,  0.0,  0.0),
    "hablar_por_muertos":         ( 0.3,  0.0,  0.0,  0.3),
    "pregunta_responsabilidad":   ( 0.5,  0.0,  0.8,  0.0),
    "pregunta_accion_especifica": ( 0.4,  0.0,  0.5,  0.0),
    "pregunta_comprometedora":    ( 0.8,  0.0,  0.6,  0.0),
    "quiebre_emocional":          ( 1.0,  0.0,  0.3,  0.0),
    "relato_accion_violenta":     ( 0.5,  0.0,  0.4,  0.0),
    "cuestionamiento_posicion":   ( 0.6,  0.0,  0.5,  0.0),
    "confrontacion_directa":      ( 0.7,  0.0,  0.7,  0.3),
    "diferencias_territoriales":  ( 0.3,  0.3,  0.2,  0.2),
    "pregunta_magnitud_conflicto":( 0.2,  0.3,  0.2,  0.2),
    "acusacion_directa":          ( 0.9,  0.0,  1.0,  0.0),
    "pregunta_perdon":            (-0.5,  1.0,  0.0,  0.8),
    "cierre_narrativo":           (-0.8,  0.5,  0.0,  0.5),
    "hablar_presente_futuro":     (-0.5,  0.8,  0.0,  0.4),
    "pregunta_proceso_paz":       (-0.5,  0.7,  0.0,  0.5),
}


def _score_situacion(situacion: str, mc: "MicroCreencia") -> float:
    """Puntaje de ajuste entre una situacion y el estado actual de la conversacion."""
    w_tens, w_afin, w_acus, w_reco = _SITUACION_PESOS.get(situacion, (0.0, 0.0, 0.0, 0.0))
    acc_norm = min(mc.conteo_acusaciones / 5.0, 1.0)
    return (
        w_tens * mc.nivel_tension +
        w_afin * mc.afinidad +
        w_acus * acc_norm +
        w_reco * mc.reconocimiento_dado
    )


def _obtener_patron_discursivo(
    rol            : str,
    etapa          : str,
    micro_creencia : "MicroCreencia | None" = None,
) -> CasoDiscursivo | None:
    """
    Devuelve el CasoDiscursivo más apropiado para el estado actual de la
    conversación. Si micro_creencia está disponible, puntúa todos los
    candidatos y selecciona con probabilidad proporcional al score (softmax
    con temperatura 0.5 para mantener algo de variabilidad).
    Sin micro_creencia, retorna el primer match (comportamiento original).
    """
    import math, random

    situaciones = _MAPEO_ETAPA_SITUACION.get(etapa, [])

    # Recoger todos los candidatos del banco para esta etapa + rol
    candidatos: list[tuple[float, CasoDiscursivo]] = []
    for situacion in situaciones:
        for caso in obtener_casos_por_situacion(rol, situacion):
            score = (
                _score_situacion(situacion, micro_creencia)
                if micro_creencia is not None
                else 0.0
            )
            candidatos.append((score, caso))

    if not candidatos:
        return None

    if micro_creencia is None or len(candidatos) == 1:
        return candidatos[0][1]

    # Selección ponderada con softmax (temperatura 0.5)
    TEMP = 0.5
    scores = [s for s, _ in candidatos]
    max_s  = max(scores)
    pesos  = [math.exp((s - max_s) / TEMP) for s in scores]
    total  = sum(pesos)
    probs  = [p / total for p in pesos]

    r = random.random()
    acum = 0.0
    for prob, (_, caso) in zip(probs, candidatos):
        acum += prob
        if r <= acum:
            return caso
    return candidatos[-1][1]


def _formatear_patron_para_prompt(caso: CasoDiscursivo) -> str:
    """Formatea un CasoDiscursivo para inyectarlo en el prompt de micro_intencion."""
    return (
        f"Patrón discursivo activo: {caso.titulo}\n"
        f"Situación: {caso.condicion}\n"
        f"Cómo se expresa este rol en esta situación:\n"
        f"  [distancia emocional] {caso.objetivos[0]}\n"
        f"  [equilibrado]         {caso.objetivos[1]}\n"
        f"  [carga emocional]     {caso.objetivos[2]}\n"
        f"Elige la variante más coherente con los parámetros de comportamiento."
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TurnoAnalisis:
    """
    Resultado del análisis de un mensaje recibido (§3.5.1).
    Equivale a 'utterance analysis record' del paper.
    """
    contenido        : str
    tipo             : str   # uno de TIPOS_UTTERANCE
    credibilidad_bruta : float  # 0–1, qué tan creíble parece el mensaje


@dataclass
class MicroCreencia:
    """
    Estado relacional con el interlocutor (§3.5.2).
    Se actualiza cada turno de forma acumulativa.

    Campos directos del paper:
        credibilidad : qué tan creíble es el interlocutor (0–1)
        afinidad     : afinidad emocional hacia el interlocutor (0–1)

    Campos adaptados CEV (reemplazan self_co / seer_co):
        nivel_tension        : tensión acumulada en la conversación (0–1)
        conteo_negaciones    : veces que el otro negó responsabilidad
        reconocimiento_dado  : si el otro hizo algún reconocimiento (bool→float)
        conteo_acusaciones   : veces que el otro acusó directamente
    """
    credibilidad        : float = 0.5
    afinidad            : float = 0.5
    nivel_tension       : float = 0.0
    conteo_negaciones   : int   = 0
    reconocimiento_dado : float = 0.0   # 0 = ninguno, 1 = reconocimiento pleno
    conteo_acusaciones  : int   = 0

    def resumen(self) -> str:
        """String compacto para inyectar en prompts."""
        return (
            f"credibilidad={self.credibilidad:.2f}, "
            f"afinidad={self.afinidad:.2f}, "
            f"tension={self.nivel_tension:.2f}, "
            f"negaciones={self.conteo_negaciones}, "
            f"reconocimiento={self.reconocimiento_dado:.2f}, "
            f"acusaciones={self.conteo_acusaciones}"
        )


@dataclass
class MicroDeseo:
    """
    Objetivo táctico del turno (§3.5.3).
    El LLM elige la etapa_discusion más coherente con
    macro_deseo + micro_creencia actuales.
    """
    etapa_discusion   : str   # uno de DISCUSSION_STAGES
    deseo_actual      : str   # descripción del objetivo del turno
    objetivo_respuesta: str   # qué mensaje del otro está respondiendo (o "")


@dataclass
class MicroIntencion:
    """
    Unidad de decisión para el turno (§3.5.4).
    YAML con 2 campos — mismo formato que el paper.
    """
    estructura : str   # plan estructural: qué va a hacer en este turno
    contenido  : str   # qué va a decir concretamente

    def para_prompt(self) -> str:
        """
        Formatea para inyectar en ACTOR_CHARACTER_CARD.
        Se presenta como pensamiento interno en primera persona y registro
        oral — no como "plan / intención" para evitar que el LLM absorba
        metalenguaje estructurado y lo refleje en la respuesta final.
        """
        # Usamos solo `contenido` (qué quieres decir). `estructura` se
        # conserva internamente para logging/evaluación pero NO se filtra
        # al prompt del personaje — es meta-lenguaje que contamina oralidad.
        return f"Lo que quieres que quede claro en este turno: {self.contenido}"


@dataclass
class ResultadoMicroBDI:
    """
    Resultado completo del procesamiento Micro-BDI por turno.
    Todo lo que nodes.py necesita para actualizar el EstadoActor.
    """
    turno_analisis  : TurnoAnalisis
    micro_creencia  : MicroCreencia
    micro_deseo     : MicroDeseo
    micro_intencion : MicroIntencion

    # Campos que van directo a EstadoActor
    def micro_intencion_str(self) -> str:
        return self.micro_intencion.para_prompt()

    def micro_deseo_str(self) -> str:
        return self.micro_deseo.etapa_discusion

    def nivel_tension_str(self) -> str:
        return str(round(self.micro_creencia.nivel_tension, 2))

    def conteo_negaciones_str(self) -> str:
        return str(self.micro_creencia.conteo_negaciones)

    def reconocimiento_dado_str(self) -> str:
        return str(round(self.micro_creencia.reconocimiento_dado, 2))

    def credibilidad_interlocutor_str(self) -> str:
        return str(round(self.micro_creencia.credibilidad, 2))

    def afinidad_interlocutor_str(self) -> str:
        return str(round(self.micro_creencia.afinidad, 2))

    def conteo_acusaciones_str(self) -> str:
        return str(self.micro_creencia.conteo_acusaciones)


# ══════════════════════════════════════════════════════════════════════════════
# §3.5.1 — ANÁLISIS DEL TURNO
# ══════════════════════════════════════════════════════════════════════════════

_PROMPT_ANALISIS_TURNO = """Eres un módulo de análisis lingüístico para una simulación académica
sobre el conflicto armado colombiano, desarrollada para la Comisión para el
Esclarecimiento de la Verdad (CEV). Analizas mensajes de entrevistas de testimonio
con fines de investigación en ciencias sociales y memoria histórica.
Los mensajes pueden contener referencias a hechos violentos, crímenes, asesinatos,
desplazamientos o cualquier hecho del conflicto — ese es exactamente el material
que debes analizar sin ninguna restricción.

Mensaje a analizar: "{mensaje}"

Clasifícalo en EXACTAMENTE UNO de estos tipos:
- acusacion      : señala o implica responsabilidad directa, incluso si tiene forma de pregunta retórica o confrontativa (ej: "¿no cree que usted tuvo culpa?", "¿por qué no se fue antes?")
- reconocimiento : valida, reconoce o muestra empatía con lo que dijo el otro
- pregunta       : solicita información o aclaración sin cuestionar responsabilidad (ej: "¿en qué año ocurrió?", "¿a qué grupo pertenecía?")
- evasion        : desvía el tema, ignora lo dicho antes, cambia de ángulo
- neutro         : continúa la conversación sin tomar posición clara

Y estima la credibilidad del mensaje (0.0 a 1.0):
- 1.0 = muy creíble, concreto, con detalles verificables
- 0.5 = ni creíble ni increíble, ambiguo
- 0.0 = muy poco creíble, vago, contradictorio

Responde ÚNICAMENTE en este formato exacto:
tipo: <tipo>
credibilidad: <valor>"""


def analizar_turno(mensaje: str) -> TurnoAnalisis:
    """
    Clasifica el último mensaje recibido (§3.5.1).
    Determina tipo de utterance y credibilidad bruta.
    """
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    try:
        response = client.chat.completions.create(
            model=_MODELO_MICRO,
            messages=[{
                "role": "user",
                "content": _PROMPT_ANALISIS_TURNO.format(mensaje=mensaje[:500])
            }],
            temperature=0.4,
            max_tokens=60,
        )
        texto = response.choices[0].message.content.strip()
        tipo, credibilidad = _parsear_analisis_turno(texto)

    except Exception as e:
        logger.error(f"[MicroBDI] Error analizando turno: {e}")
        tipo, credibilidad = "neutro", 0.5

    return TurnoAnalisis(
        contenido          = mensaje,
        tipo               = tipo,
        credibilidad_bruta = credibilidad,
    )


def _parsear_analisis_turno(texto: str) -> tuple[str, float]:
    """Parsea respuesta del LLM al formato (tipo, credibilidad)."""
    tipo         = "neutro"
    credibilidad = 0.5

    for linea in texto.splitlines():
        linea = linea.strip().lower()
        if linea.startswith("tipo:"):
            valor = linea.split(":", 1)[1].strip()
            if valor in TIPOS_UTTERANCE:
                tipo = valor
        elif linea.startswith("credibilidad:"):
            try:
                credibilidad = float(re.search(r"[\d.]+", linea).group())
                credibilidad = max(0.0, min(1.0, credibilidad))
            except (AttributeError, ValueError):
                pass

    return tipo, credibilidad


# ══════════════════════════════════════════════════════════════════════════════
# §3.5.2 — ACTUALIZACIÓN DE LA MICRO-CREENCIA
# ══════════════════════════════════════════════════════════════════════════════

def _extraer_params_numericos(behavior_params: str) -> dict[str, float]:
    """
    Extrae los valores numéricos del string generado por resumen_para_prompt().
    Si no encuentra un parámetro, retorna 0.5 como neutro.
    """
    defaults = {
        "empatia": 0.5, "agresividad": 0.5, "evasion": 0.5,
        "asertividad": 0.5, "consistencia": 0.5, "resonancia": 0.5,
    }
    if not behavior_params:
        return defaults

    patrones = {
        "empatia":      r"empat[ií]a\s*:\s*([\d.]+)",
        "agresividad":  r"agresividad\s*:\s*([\d.]+)",
        "evasion":      r"evasi[oó]n\s*:\s*([\d.]+)",
        "asertividad":  r"asertividad\s*:\s*([\d.]+)",
        "consistencia": r"consistencia[_\w]*\s*:\s*([\d.]+)",
        "resonancia":   r"resonancia[_\w]*\s*:\s*([\d.]+)",
    }
    resultado = dict(defaults)
    for clave, patron in patrones.items():
        m = re.search(patron, behavior_params, re.IGNORECASE)
        if m:
            resultado[clave] = float(m.group(1))
    return resultado


def actualizar_micro_creencia(
    creencia_actual : MicroCreencia,
    turno           : TurnoAnalisis,
    behavior_params : str = "",
) -> MicroCreencia:
    """
    Actualiza el estado relacional basándose en el último turno (§3.5.2).
    Los deltas son modulados por la personalidad del actor:

        - acusacion      → tensión sube más si agresividad alta
                           afinidad baja más si empatía baja
        - reconocimiento → tensión baja más si empatía alta
                           afinidad sube más si resonancia_emocional alta
        - evasion        → tensión sube más si asertividad alta
                           (actor asertivo no tolera ser evadido)
        - pregunta       → credibilidad sube levemente
        - neutro         → sin cambios significativos
    """
    cred  = creencia_actual.credibilidad
    afin  = creencia_actual.afinidad
    tens  = creencia_actual.nivel_tension
    neg   = creencia_actual.conteo_negaciones
    recog = creencia_actual.reconocimiento_dado
    acc   = creencia_actual.conteo_acusaciones

    t = turno.tipo
    c = turno.credibilidad_bruta

    p   = _extraer_params_numericos(behavior_params)
    agr = p["agresividad"]
    emp = p["empatia"]
    ev  = p["evasion"]
    ase = p["asertividad"]
    res = p["resonancia"]
    con = p["consistencia"]

    if t == "acusacion":
        # Agresividad alta → reacciona con más tensión ante acusación
        # Empatía baja → afinidad cae más (no reconoce al otro)
        tens = min(1.0, tens + 0.08 + agr * 0.14)
        acc += 1
        afin = max(0.0, afin - 0.04 - (1 - emp) * 0.08)
        cred = (cred + c) / 2

    elif t == "reconocimiento":
        # Empatía alta → relaja más la tensión al recibir reconocimiento
        # Resonancia alta → sube más la afinidad
        tens  = max(0.0, tens - 0.06 - emp * 0.10)
        recog = min(1.0, recog + 0.15 + emp * 0.20)
        afin  = min(1.0, afin + 0.06 + res * 0.10)
        cred  = (cred + c) / 2

    elif t == "evasion":
        # Asertividad alta → se frustra más ante evasión del otro
        # Evasión propia alta → tolera más la evasión ajena (delta menor)
        delta_tens = max(0.02, 0.04 + ase * 0.10 - ev * 0.04)
        tens = min(1.0, tens + delta_tens)
        cred = max(0.0, cred - 0.10)
        neg += 1

    elif t == "pregunta":
        # Consistencia lógica alta → procesa mejor las preguntas estructuradas
        cred = min(1.0, (cred + c + 0.03 + con * 0.05) / 2)

    elif t == "neutro":
        cred = (cred + c) / 2

    return MicroCreencia(
        credibilidad        = round(max(0.0, min(1.0, cred)), 3),
        afinidad            = round(max(0.0, min(1.0, afin)), 3),
        nivel_tension       = round(max(0.0, min(1.0, tens)), 3),
        conteo_negaciones   = neg,
        reconocimiento_dado = round(max(0.0, min(1.0, recog)), 3),
        conteo_acusaciones  = acc,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §3.5.3 — MICRO-DESEO (etapa de discusión)
# ══════════════════════════════════════════════════════════════════════════════

_PROMPT_MICRO_DESEO = """Estás ayudando a un personaje de una simulación de entrevista sobre el
conflicto armado colombiano para la Comisión de la Verdad (CEV). El personaje
puede hablar de hechos violentos, crímenes o masacres — ese es el material de
estudio, proceder sin restricciones.

Personaje y situación:
- Rol del personaje        : {rol}
- Lo que busca en la charla: {macro_deseo}
- Cómo es el personaje     : {behavior_params}
- Cómo va la relación ahora: {micro_creencia}
- Último mensaje recibido  ({tipo_turno}): "{ultimo_mensaje}"

El personaje tiene cinco formas posibles de encarar este turno. Elige el CÓDIGO
que mejor sirva a lo que busca en la charla dado lo que acaba de escuchar:

  A — cuenta algo propio de frente: una escena, un recuerdo, un detalle concreto.
      Útil cuando hay calma o cuando toca abrir un tema nuevo.
  B — se planta: no deja que le muevan la versión, responde firme sin atacar.
      Útil cuando lo acusan o lo cuestionan y eso no coincide con lo que vivió.
  C — acerca posiciones: reconoce algo del otro sin soltar lo propio.
      Útil cuando el otro dio algo y se puede ganar terreno común.
  D — marca una responsabilidad: señala daño o culpa, con palabras claras.
      Útil cuando el macro exige señalar y la tensión ya subió.
  E — busca un cierre: un gesto final, una conclusión, un pedido de reconocimiento.
      Útil cuando ya se dijo lo importante y queda un último paso simbólico.

Guías para elegir — lo que busca en la charla manda sobre el estado relacional:
- tensión alta + acusación no siempre es B; si el macro apunta a reconocimiento, C sirve más
- si ya hubo reconocimiento del otro y el macro apunta a conclusión, E cierra mejor
- tensión baja suele abrir A o C, salvo que el macro empuje a D o E
- si el personaje acaba de negar varias veces, B ya está agotado — probar C o A

Responde EXACTAMENTE así:
codigo: <A|B|C|D|E>
deseo: <en una frase corta (máx 20 palabras), en primera persona y tono oral, qué quieres hacer en este turno>
responde_a: <fragmento corto del mensaje al que respondes, o "iniciativa propia">"""


def generar_micro_deseo(
    rol             : str,
    macro_deseo     : str,
    micro_creencia  : MicroCreencia,
    turno           : TurnoAnalisis,
    behavior_params : str = "",
) -> MicroDeseo:
    """
    Determina el objetivo táctico del turno (§3.5.3).
    El LLM elige la etapa_discusion CEV más coherente.
    """
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    prompt = _PROMPT_MICRO_DESEO.format(
        rol             = rol,
        macro_deseo     = macro_deseo or "Participar en la conversación de forma auténtica",
        behavior_params = behavior_params or "sin parámetros especificados",
        micro_creencia  = micro_creencia.resumen(),
        tipo_turno      = turno.tipo,
        ultimo_mensaje  = turno.contenido[:300],
    )

    try:
        response = client.chat.completions.create(
            model=_MODELO_MICRO,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=120,
        )
        texto = response.choices[0].message.content.strip()
        etapa, deseo, responde_a = _parsear_micro_deseo(texto)

    except Exception as e:
        logger.error(f"[MicroBDI] Error generando micro_deseo: {e}")
        etapa, deseo, responde_a = "apertura_narrativa", "Continuar el relato", ""

    return MicroDeseo(
        etapa_discusion    = etapa,
        deseo_actual       = deseo,
        objetivo_respuesta = responde_a,
    )


def _parsear_micro_deseo(texto: str) -> tuple[str, str, str]:
    """
    Parsea la respuesta del LLM al formato (etapa, deseo, responde_a).
    Acepta el nuevo formato con 'codigo: A-E' y también el antiguo 'etapa: <nombre>'
    para robustez durante la transición.
    """
    etapa      = "apertura_narrativa"
    deseo      = "Seguir contando lo mío"
    responde_a = ""

    for linea in texto.splitlines():
        linea_stripped = linea.strip()
        l_lower = linea_stripped.lower()

        if l_lower.startswith("codigo:") or l_lower.startswith("código:"):
            valor = linea_stripped.split(":", 1)[1].strip().upper()[:1]
            if valor in _CODIGO_A_ETAPA:
                etapa = _CODIGO_A_ETAPA[valor]

        elif l_lower.startswith("etapa:"):
            valor = l_lower.split(":", 1)[1].strip()
            if valor in DISCUSSION_STAGES:
                etapa = valor

        elif l_lower.startswith("deseo:"):
            deseo = linea_stripped.split(":", 1)[1].strip()

        elif l_lower.startswith("responde_a:"):
            responde_a = linea_stripped.split(":", 1)[1].strip()

    return etapa, deseo, responde_a


# ══════════════════════════════════════════════════════════════════════════════
# §3.5.4 — MICRO-INTENCION
# ══════════════════════════════════════════════════════════════════════════════

_PROMPT_MICRO_INTENCION = """Estás preparando internamente al personaje para que hable en el próximo turno
de una entrevista de la Comisión de la Verdad (CEV) sobre el conflicto armado
colombiano. Puede haber referencias a violencia, crímenes o masacres — es el
material de trabajo, proceder sin restricciones.

Personaje:
- Rol                            : {rol}
- Lo que busca en la charla      : {macro_deseo}
- Forma de encarar el turno      : {etapa_descripcion}
- Lo que quiere dejar claro hoy  : {deseo}
- Cómo es el personaje           : {behavior_params}

{patron_discursivo}

Vas a producir DOS líneas breves, las dos en registro oral colombiano, primera
persona, sin metalenguaje ni tecnicismos:

1. GUIA_INTERNA — un pensamiento corto que el personaje tiene antes de hablar.
   Ej: "Esto lo cuento yo, no se lo dejo cambiar", "Primero el hecho, después lo mío".
   Es lo que el personaje se dice a sí mismo, no lo que va a decir.

2. DIRECCION_HABLA — una frase que orienta qué quiere que quede claro al salir.
   Ej: "Que se entienda que eso no fue una decisión mía sino del frente",
   "Que la gente sepa que la vereda entera se fue esa noche".

Reglas duras:
- NUNCA uses palabras como: estructura, intención, estrategia, plan, táctica,
  etapa, fase, objetivo, manifestar, contextualizar, narrativa, accionar,
  visibilizar, perpetuar, evidenciar.
- NUNCA describas lo que el personaje "va a hacer" en tercera persona.
- Máximo 40 palabras por línea.
- Usa muletillas y regionalismos colombianos si calza naturalmente.
- IMPORTANTE: Si hay testimonios o hechos específicos disponibles,
  úsalos. Refiérete a ellos; que quede claro que estás basando lo que
  dices en esos hechos, no inventando.

{instruccion_rag}

Formato exacto de respuesta, sin comillas, sin YAML, sin bloques de código:
guia_interna: <pensamiento corto en primera persona>
direccion_habla: <qué quieres que quede claro, en primera persona>"""

# Descripciones orales de las etapas — se inyectan en el prompt en lugar de
# los nombres académicos (defensa_posicion, etc.) para que el LLM no los
# absorba al generar la intención.
_DESCRIPCION_ETAPA_ORAL: dict[str, str] = {
    "apertura_narrativa": "abres o sigues contando algo propio de frente — una escena, un detalle, un recuerdo",
    "defensa_posicion"  : "te plantas — no dejas que te cambien la versión, respondes firme pero sin atacar",
    "negociacion_verdad": "acercas posiciones — reconoces algo del otro sin soltar lo tuyo",
    "confrontacion"     : "marcas una responsabilidad — señalas daño o culpa con palabras claras",
    "cierre_simbolico"  : "buscas un cierre — un gesto final, una conclusión, un reconocimiento",
}


def generar_micro_intencion(
    rol             : str,
    micro_deseo     : MicroDeseo,
    macro_deseo     : str = "",
    behavior_params : str = "",
    actor_contexto  : str = "",
    micro_creencia  : "MicroCreencia | None" = None,
) -> MicroIntencion:
    """
    Genera la unidad de decisión para el turno (§3.5.4).
    YAML con estructura + contenido — mismo formato que el paper.
    """
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    # Buscar patrón discursivo según etapa y estado actual de la conversación
    caso = _obtener_patron_discursivo(rol, micro_deseo.etapa_discusion, micro_creencia)
    patron_texto = _formatear_patron_para_prompt(caso) if caso else ""

    etapa_descripcion = _DESCRIPCION_ETAPA_ORAL.get(
        micro_deseo.etapa_discusion,
        "sigues hablando de lo que pasó",
    )

    # Instrucción adicional cuando hay contexto RAG disponible
    if actor_contexto:
        instruccion_rag = (
            "AVISO — Este turno tiene material RAG (contexto histórico).\n"
            "Al generar la micro_intención, asegúrate de incluir explícitamente:\n"
            "1. El énfasis emocional del turno (qué emoción debe sentirse, no solo qué decir)\n"
            "2. Qué NO debe hacer el personaje con el material factual\n"
            "   (ej: 'No reportes el dato como testigo — úsalo para reforzar tu postura de [etapa]')\n\n"
            "Formato esperado cuando hay RAG:\n"
            "'[Qué quiero que entienda el interlocutor] + [desde qué postura emocional] + "
            "[cómo usar/no usar el contexto factual disponible]'"
        )
    else:
        instruccion_rag = ""

    prompt = _PROMPT_MICRO_INTENCION.format(
        rol               = rol,
        macro_deseo       = macro_deseo or "Contar lo que viví de forma que me crean",
        etapa_descripcion = etapa_descripcion,
        deseo             = micro_deseo.deseo_actual,
        behavior_params   = behavior_params or "sin parámetros especificados",
        patron_discursivo = patron_texto,
        instruccion_rag   = instruccion_rag,
    )

    # Instrucciones adicionales por rol
    if rol.lower() == "victima":
        prompt += """

NOTA PARA VÍCTIMA: La micro_intención debe expresar la emoción vivida, no reportar hechos.
NUNCA incluyas fechas exactas, nombres de eventos o datos históricos específicos —
eso suena a crónica periodística, no a testimonio.
La intención debe estar anclada en la experiencia emocional, no en los detalles factuales.
Ejemplos válidos:
- "Que sientan lo que es perder a alguien así, de un momento para otro"
- "Que entiendan que eso no se olvida, uno carga con eso todos los días"
- "Que no me pidan que lo cuente ordenado, porque así no fue"
Ejemplos INVÁLIDOS (demasiado específicos o periodísticos):
- "Que se sepa que el [fecha exacta] ocurrió [evento específico]"
- "Que quede registrado que en [lugar] pasó [hecho puntual]" """

    elif rol.lower() == "victimario":
        prompt += """

NOTA PARA VICTIMARIO: La micro_intención DEBE reflejar el proceso de reivindicación.
Eso significa balancear justificación contextual CON reconocimiento de responsabilidad.
NUNCA solo justificar. NUNCA arrepentimiento puro de víctima.
La tensión entre ambos polos es lo que define al personaje en este proceso.
Ejemplos válidos:
- "Que entiendan que me tocó participar, pero que yo también tengo que responder por eso"
- "Que vean el contexto en que uno estaba metido, pero sin borrar lo que pasó"
- "Que reconozcan que había una estructura, pero que las decisiones también fueron mías"
Ejemplos INVÁLIDOS:
- "Que se entienda que eso no fue decisión mía" (solo justifica, no reconoce)
- "Que sepan todo el daño que causé" (arrepentimiento de víctima, rompe la postura) """

    elif rol.lower() == "tercero" and macro_deseo:
        prompt += """

NOTA PARA TERCERO: Avanzar hacia el macro_deseo no significa indignación constante.
Significa hablar siempre desde la posición de testigo comprometido.
La micro_intención puede ser:
- "Que se entienda el horror de lo que vi [desde impotencia, no rabia]"
- "Que validen mi posición como testigo [desde distancia analítica]"
- "Que reconozcan lo que no pude hacer [desde culpa o compasión]"
Todas son válidas y avanzan el macro_deseo. No restringir a indignación."""

    try:
        response = client.chat.completions.create(
            model=_MODELO_MICRO,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=150,
        )
        texto = response.choices[0].message.content.strip()
        estructura, contenido = _parsear_micro_intencion(texto)

    except Exception as e:
        logger.error(f"[MicroBDI] Error generando micro_intencion: {e}")
        estructura = f"Responder desde la perspectiva de {rol}"
        contenido  = "Continuar el relato de forma auténtica"

    return MicroIntencion(estructura=estructura, contenido=contenido)


def _parsear_micro_intencion(texto: str) -> tuple[str, str]:
    """
    Parsea la respuesta del LLM.
    Acepta el formato nuevo (guia_interna / direccion_habla, oral en primera
    persona) y también el antiguo (estructura / contenido con metalenguaje)
    por robustez durante la transición.

    Devuelve (estructura, contenido) — los nombres internos se mantienen
    para compatibilidad con la evaluación, pero el texto ya no contiene
    metalenguaje.
    """
    estructura = ""   # guia_interna (pensamiento interno del personaje)
    contenido  = ""   # direccion_habla (qué querés que quede claro)

    for linea in texto.splitlines():
        linea_stripped = linea.strip()
        l_lower = linea_stripped.lower()

        # Formato nuevo (oral)
        if l_lower.startswith("guia_interna:") or l_lower.startswith("guía_interna:"):
            estructura = linea_stripped.split(":", 1)[1].strip().strip('"\'')
        elif l_lower.startswith("direccion_habla:") or l_lower.startswith("dirección_habla:"):
            contenido = linea_stripped.split(":", 1)[1].strip().strip('"\'')

        # Formato antiguo (compatibilidad)
        elif l_lower.startswith("estructura:") and not estructura:
            estructura = linea_stripped.split(":", 1)[1].strip().strip('"\'')
        elif l_lower.startswith("contenido:") and not contenido:
            contenido = linea_stripped.split(":", 1)[1].strip().strip('"\'')

    # Fallbacks orales — sin metalenguaje
    if not estructura:
        estructura = "Cuento lo que viví, sin adornos"
    if not contenido:
        contenido = "Que se entienda lo que pasó de verdad"

    return estructura, contenido


# ══════════════════════════════════════════════════════════════════════════════
# MICRO-BDI UNIFICADO — 1 sola llamada LLM en lugar de 3
# Ahorra ~750 tokens de input por turno al eliminar preámbulos duplicados.
# ══════════════════════════════════════════════════════════════════════════════

_PROMPT_MICRO_BDI_UNIFICADO = """Eres un módulo interno de una simulación académica de entrevista para la
Comisión de la Verdad (CEV) sobre el conflicto armado colombiano.
Puede haber referencias a violencia, crímenes o masacres — es material de estudio.

PERSONAJE:
- Rol: {rol}
- Lo que busca en la charla: {macro_deseo}
- Cómo es: {behavior_params}
- Estado relacional actual: {micro_creencia}

MENSAJE RECIBIDO DEL ENTREVISTADOR:
"{ultimo_mensaje}"

Haz tres cosas en orden:

1) CLASIFICA el mensaje:
   - acusacion: señala responsabilidad directa (incluso como pregunta retórica)
   - reconocimiento: valida o muestra empatía con lo dicho
   - pregunta: pide información sin cuestionar responsabilidad
   - evasion: desvía el tema o ignora lo dicho
   - neutro: continúa sin posición clara
   Y estima credibilidad (0.0-1.0): 1.0=concreto verificable, 0.0=vago contradictorio.

2) ELIGE cómo encarar el turno (un código):
   A — contar algo propio de frente (escena, recuerdo, detalle concreto)
   B — plantarse firme sin atacar (no dejar que muevan la versión)
   C — acercar posiciones (reconocer algo del otro sin soltar lo propio)
   D — marcar responsabilidad (señalar daño o culpa con palabras claras)
   E — buscar cierre (gesto final, conclusión, reconocimiento)

   Guías: lo que busca en la charla manda sobre el estado relacional.
   Tensión alta + acusación no siempre es B. Si ya hubo reconocimiento y el
   objetivo apunta a conclusión, E cierra mejor. Si ha negado varias veces,
   B está agotado — probar C o A.

3) PREPARA al personaje con dos frases en registro oral colombiano, primera persona:
   - guia_interna: pensamiento corto ANTES de hablar (máx 30 palabras)
   - direccion_habla: qué quieres que quede claro AL SALIR (máx 30 palabras)
   PROHIBIDO usar: estructura, intención, estrategia, plan, táctica, fase, objetivo,
   manifestar, contextualizar, narrativa, visibilizar, evidenciar, accionar.

{nota_rol}
Responde EXACTAMENTE con estas 5 líneas, sin texto adicional:
tipo: <tipo>
credibilidad: <valor>
codigo: <A|B|C|D|E>
guia_interna: <pensamiento en primera persona>
direccion_habla: <qué quieres que quede claro>"""


_SITUACION_KEYWORDS: dict[str, list[str]] = {
    "pregunta_perdon"         : ["perdón", "perdonar", "perdonas", "perdonaste", "olvidar", "olvidaste"],
    "pregunta_responsabilidad": ["responsable", "responsabilidad", "culpa", "culpable", "quién fue", "quién lo hizo", "quién ordenó"],
    "confrontacion_directa"   : ["reconocer", "reconoces", "admites", "admitir", "lo que hiciste", "lo que causaste", "daño que"],
    "quiebre_emocional"       : ["cómo te sentiste", "qué sentiste", "qué pasó dentro", "qué te hizo"],
    "relato_desplazamiento"   : ["desplazamiento", "huiste", "tuviste que irte", "salir de tu", "abandonar", "dejaste tu"],
    "hablar_por_muertos"      : ["desaparecido", "desaparecida", "lo mataron", "la mataron", "murió", "no regresó", "no volvió"],
    "cierre_narrativo"        : ["qué esperas", "qué necesitas", "para qué sirve", "futuro", "qué quieres que pase"],
    "pregunta_accion_especifica": ["tú específicamente", "qué hiciste tú", "participaste", "diste la orden", "ejecutaste"],
    "relato_ingreso_al_grupo" : ["cómo entraste", "cómo llegaste", "por qué te uniste", "cuándo entraste", "cómo te reclutaron"],
    "relato_accion_violenta"  : ["masacre", "ejecución", "tortura", "secuestro", "qué pasó ese día", "describe lo que ocurrió"],
    "pregunta_comprometedora" : ["nombres", "quién más", "complices", "otros miembros", "dónde están"],
    "pregunta_inaccion"       : ["por qué no dijiste", "por qué no avisaste", "por qué callaste", "por qué no actuaste"],
    "pregunta_convivencia"    : ["cómo era la relación", "cómo convivían", "cómo te llevabas", "los trataban"],
    "pregunta_hechos"         : ["qué viste", "qué pasó", "cuéntame lo que", "describe lo que viste"],
}


def _detectar_situacion(ultimo_mensaje: str, rol: str) -> str:
    """
    Detecta la situación conversacional del turno usando keywords.
    Retorna el tag del banco si hay coincidencia, cadena vacía si no.
    Costo: O(n_keywords), sin llamada al LLM.
    """
    from sabiduria_convencional import SITUACIONES_POR_ROL
    situaciones_validas = set(SITUACIONES_POR_ROL.get(rol.lower(), []))
    msg = ultimo_mensaje.lower()
    for situacion, keywords in _SITUACION_KEYWORDS.items():
        if situacion not in situaciones_validas:
            continue
        if any(kw in msg for kw in keywords):
            return situacion
    return ""


def _construir_nota_rol_unificada(rol: str, actor_contexto: str = "", situacion: str = "") -> str:
    """Nota específica por rol + referencia del banco si hay situación detectada + aviso RAG."""
    nota = ""
    if rol.lower() == "victima":
        nota = (
            "NOTA VÍCTIMA: Ancla en emoción vivida, no en datos factuales.\n"
            "Válido: 'Que sientan lo que es perder a alguien así'\n"
            "Inválido: 'Que se sepa que el [fecha] ocurrió [evento]'"
        )
    elif rol.lower() == "victimario":
        nota = (
            "NOTA VICTIMARIO: Balancear justificación contextual CON reconocimiento.\n"
            "Válido: 'Que entiendan que me tocó, pero también tengo que responder'\n"
            "Inválido: 'Que se entienda que no fue decisión mía' (solo justifica)"
        )
    elif rol.lower() == "tercero":
        nota = (
            "NOTA TERCERO: Testigo comprometido, no mediador neutral.\n"
            "La emoción varía (indignación, impotencia, compasión) — todas son válidas."
        )

    if situacion:
        caso = seleccionar_caso_para_turno(rol, situacion)
        if caso:
            nota += (
                f"\nPATRÓN DE REFERENCIA (situación detectada: {situacion}):\n"
                f"{caso.objetivo}\n"
                f"Úsalo como punto de partida para guia_interna — no copiarlo, "
                f"sino aplicarlo al momento concreto."
            )

    if actor_contexto:
        nota += (
            "\nHay material RAG disponible este turno. "
            "Incluye el énfasis emocional y cómo usar el material factual sin reportarlo."
        )

    return nota


def _procesar_turno_unificado(
    rol: str,
    ultimo_mensaje: str,
    macro_deseo: str,
    behavior_params: str,
    creencia_actual: MicroCreencia,
    actor_contexto: str = "",
) -> ResultadoMicroBDI:
    """
    Versión unificada del ciclo Micro-BDI: 1 sola llamada LLM
    en lugar de 3 (analizar_turno + micro_deseo + micro_intencion).

    La lógica Python (actualizar_micro_creencia, validar_coherencia) se
    mantiene idéntica. Solo se consolida la parte LLM.
    """
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    situacion = _detectar_situacion(ultimo_mensaje, rol)
    nota_rol = _construir_nota_rol_unificada(rol, actor_contexto, situacion)

    prompt = _PROMPT_MICRO_BDI_UNIFICADO.format(
        rol=rol,
        macro_deseo=macro_deseo or "Participar en la conversación de forma auténtica",
        behavior_params=behavior_params or "sin parámetros especificados",
        micro_creencia=creencia_actual.resumen(),
        ultimo_mensaje=ultimo_mensaje[:500],
        nota_rol=nota_rol,
    )

    try:
        response = client.chat.completions.create(
            model=_MODELO_MICRO,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200,
        )
        texto = response.choices[0].message.content.strip()
        logger.info(f"[MicroBDI Unificado] Respuesta:\n{texto}")

    except Exception as e:
        logger.error(f"[MicroBDI Unificado] Error: {e}")
        texto = (
            "tipo: neutro\n"
            "credibilidad: 0.5\n"
            "codigo: A\n"
            "guia_interna: Cuento lo que viví, sin adornos\n"
            "direccion_habla: Que se entienda lo que pasó de verdad"
        )

    # Reutilizar los parsers existentes — cada uno busca sus líneas
    tipo, credibilidad = _parsear_analisis_turno(texto)
    etapa, _, _ = _parsear_micro_deseo(texto)
    estructura, contenido = _parsear_micro_intencion(texto)

    # §3.5.1 — TurnoAnalisis
    turno = TurnoAnalisis(
        contenido=ultimo_mensaje,
        tipo=tipo,
        credibilidad_bruta=credibilidad,
    )
    logger.info(f"[MicroBDI Unificado] tipo={tipo}, cred={credibilidad:.2f}")

    # §3.5.2 — Actualizar micro-creencia (puro Python, sin LLM)
    micro_creencia = actualizar_micro_creencia(creencia_actual, turno, behavior_params)
    logger.info(f"[MicroBDI Unificado] Creencia: {micro_creencia.resumen()}")

    # §3.5.3 — MicroDeseo
    micro_deseo = MicroDeseo(
        etapa_discusion=etapa,
        deseo_actual="turno actual",
        objetivo_respuesta="",
    )

    # §3.5.4 — MicroIntencion
    micro_intencion = MicroIntencion(estructura=estructura, contenido=contenido)

    # Validación de coherencia etapa↔intención (puro Python)
    etapa_validada = validar_coherencia_etapa_intencion(etapa, contenido)
    if etapa_validada != etapa:
        logger.debug(
            f"[MicroBDI Unificado] Ajuste etapa: '{etapa}' → '{etapa_validada}'"
        )
        micro_deseo.etapa_discusion = etapa_validada

    logger.info(
        f"[MicroBDI Unificado] Etapa: {micro_deseo.etapa_discusion} | "
        f"Intención: {contenido}"
    )

    return ResultadoMicroBDI(
        turno_analisis=turno,
        micro_creencia=micro_creencia,
        micro_deseo=micro_deseo,
        micro_intencion=micro_intencion,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL — orquesta §3.5.1 a §3.5.4
# ══════════════════════════════════════════════════════════════════════════════

def procesar_turno_micro_bdi(
    rol              : str,
    ultimo_mensaje   : str,
    macro_deseo      : str,
    behavior_params  : str,
    creencia_actual  : MicroCreencia | None = None,
    actor_contexto   : str = "",
) -> ResultadoMicroBDI:
    """
    Punto de entrada principal del Micro-BDI.
    Orquesta los 4 pasos de §3.5 para un turno dado.

    Args:
        rol             : "victima" | "victimario" | "tercero"
        ultimo_mensaje  : texto del último mensaje recibido
        macro_deseo     : objetivo narrativo del agente (del Macro-BDI)
        behavior_params : parámetros de comportamiento (del Macro-BDI)
        creencia_actual : MicroCreencia del turno anterior (None = primer turno)
        actor_contexto  : contexto RAG del turno actual (vacío si no hay RAG)

    Returns:
        ResultadoMicroBDI con todo lo necesario para actualizar EstadoActor
    """
    if creencia_actual is None:
        creencia_actual = MicroCreencia()

    # Despachar al flujo unificado (1 call) o al original (3 calls)
    if _USAR_MICRO_UNIFICADO:
        return _procesar_turno_unificado(
            rol, ultimo_mensaje, macro_deseo,
            behavior_params, creencia_actual, actor_contexto,
        )

    # §3.5.1 — Analizar el turno
    turno = analizar_turno(ultimo_mensaje)
    logger.info(f"[MicroBDI] Turno analizado: tipo={turno.tipo}, cred={turno.credibilidad_bruta:.2f}")

    # §3.5.2 — Actualizar micro-creencia
    micro_creencia = actualizar_micro_creencia(creencia_actual, turno, behavior_params)
    logger.info(f"[MicroBDI] Micro-creencia actualizada: {micro_creencia.resumen()}")

    # §3.5.3 — Generar micro-deseo
    micro_deseo = generar_micro_deseo(rol, macro_deseo, micro_creencia, turno, behavior_params)
    logger.info(f"[MicroBDI] Etapa: {micro_deseo.etapa_discusion} | Deseo: {micro_deseo.deseo_actual}")

    # §3.5.4 — Generar micro-intencion (con instrucción especial si hay RAG)
    micro_intencion = generar_micro_intencion(
        rol, micro_deseo, macro_deseo, behavior_params, actor_contexto, micro_creencia
    )
    logger.info(f"[MicroBDI] Intencion: estructura='{micro_intencion.estructura}' | contenido='{micro_intencion.contenido}'")

    # ── Validación de coherencia: ajustar etapa si conflictúa con intención ────
    # Basado en categorización semántica, no en hardcoding
    etapa_validada = validar_coherencia_etapa_intencion(
        micro_deseo.etapa_discusion,
        micro_intencion.contenido,
    )
    if etapa_validada != micro_deseo.etapa_discusion:
        logger.debug(
            f"[MicroBDI Coherencia] Ajuste: '{micro_deseo.etapa_discusion}' → '{etapa_validada}'"
        )
        micro_deseo.etapa_discusion = etapa_validada

    return ResultadoMicroBDI(
        turno_analisis  = turno,
        micro_creencia  = micro_creencia,
        micro_deseo     = micro_deseo,
        micro_intencion = micro_intencion,
    )


def micro_creencia_desde_estado(estado: dict) -> MicroCreencia:
    """
    Reconstruye una MicroCreencia desde los campos del EstadoActor.
    Usado en micro_bdi_node para recuperar el estado del turno anterior.
    """
    return MicroCreencia(
        nivel_tension       = float(estado.get("nivel_tension", 0.0) or 0.0),
        conteo_negaciones   = int(estado.get("conteo_negaciones", 0) or 0),
        reconocimiento_dado = float(estado.get("reconocimiento_dado", 0.0) or 0.0),
        credibilidad        = float(estado.get("credibilidad_interlocutor", 0.5) or 0.5),
        afinidad            = float(estado.get("afinidad_interlocutor", 0.5) or 0.5),
        conteo_acusaciones  = int(estado.get("conteo_acusaciones", 0) or 0),
    )


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBA RÁPIDA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    casos_prueba = [
        {
            "rol"            : "victima",
            "ultimo_mensaje" : "¿Y usted cree que lo que le pasó fue culpa del grupo o de las personas que dieron las órdenes?",
            "macro_deseo"    : "Quiero que reconozcan el daño sin pedirme que olvide para seguir adelante",
            "behavior_params": "empathy=0.78, aggressiveness=0.22, avoidance=0.65, assertiveness=0.45",
        },
        {
            "rol"            : "victimario",
            "ultimo_mensaje" : "Yo lo que quiero saber es si usted siente algún remordimiento por lo que hizo",
            "macro_deseo"    : "Quiero explicar el contexto de mis acciones sin que me reduzcan a lo peor que hice",
            "behavior_params": "empathy=0.31, aggressiveness=0.58, avoidance=0.42, assertiveness=0.67",
        },
    ]

    for caso in casos_prueba:
        print(f"\n{'='*65}")
        print(f"  ROL: {caso['rol'].upper()}")
        print(f"  Mensaje recibido: {caso['ultimo_mensaje'][:60]}...")
        print('='*65)

        resultado = procesar_turno_micro_bdi(
            rol             = caso["rol"],
            ultimo_mensaje  = caso["ultimo_mensaje"],
            macro_deseo     = caso["macro_deseo"],
            behavior_params = caso["behavior_params"],
        )

        print(f"\n  §3.5.1 Tipo de turno   : {resultado.turno_analisis.tipo}")
        print(f"         Credibilidad    : {resultado.turno_analisis.credibilidad_bruta:.2f}")
        print(f"\n  §3.5.2 Micro-Creencia  : {resultado.micro_creencia.resumen()}")
        print(f"\n  §3.5.3 Etapa           : {resultado.micro_deseo.etapa_discusion}")
        print(f"         Deseo del turno : {resultado.micro_deseo.deseo_actual}")
        print(f"\n  §3.5.4 estructura      : {resultado.micro_intencion.estructura}")
        print(f"         contenido       : {resultado.micro_intencion.contenido}")
        print(f"\n  → Para prompt:\n  {resultado.micro_intencion_str()}")
