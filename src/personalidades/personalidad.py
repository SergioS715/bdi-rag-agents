"""
personalidad.py
===============
Implementa la capa de estimación de personalidad del Macro-BDI layer,
basada en el paper:
  "Intent-driven AIWolf Agents with Hierarchical BDI Model and Personality"
  Harada & Kano, 2025 — Secciones 3.4.3 a 3.4.6 + Apéndice B completo.

Adaptado al contexto del proyecto EntrevistasCEV:
  - El "perfil" de entrada es la concatenación de perspectiva + estilo
    de cada actor (víctima, victimario, tercero).
  - El LLM estima MBTI desde ese texto.
  - Las fórmulas del Apéndice B derivan Enneagram y las 24 features.

Flujo:
    perfil_texto
        → estimar_mbti()          [llama al LLM, §3.4.3]
        → _calcular_enneagram()   [fórmulas lineales, Apéndice B.1]
        → _calcular_features()    [fórmulas lineales, Apéndice B.2-B.6]
        → RasgosPersonalidad      [dataclass con todos los valores]

No tiene dependencias circulares con el resto del proyecto.
Solo requiere: groq (ya instalado), config.py para la API key.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field, asdict

from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # ajusta si cambia la estructura
import config

logger = logging.getLogger(__name__)

_MODELO_MBTI = config.OPENAI_LLM_MODEL_BDI


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATACLASSES DE PERSONALIDAD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerfilMBTI:
    """
    8 dimensiones MBTI continuas en [0, 1].
    Nota del paper (§3.4.2): no se usan como "verdades psicológicas"
    sino como representación intermedia computable.

    En MBTI estándar cada par es mutuamente excluyente (E vs I),
    pero el paper los trata como valores independientes [0,1]
    para mayor flexibilidad en las fórmulas downstream.
    """
    extroversion : float = 0.5
    introversion : float = 0.5
    sensacion    : float = 0.5
    intuicion    : float = 0.5
    pensamiento  : float = 0.5
    sentimiento  : float = 0.5
    juicio       : float = 0.5
    percepcion   : float = 0.5

    def __post_init__(self):
        """Clampea todos los valores al rango [0, 1]."""
        for fname in self.__dataclass_fields__:
            val = getattr(self, fname)
            setattr(self, fname, max(0.0, min(1.0, float(val))))


@dataclass
class PerfilEnneagram:
    """
    9 tipos del Enneagram como afinidades continuas en [0, 1].
    Derivados de PerfilMBTI via fórmulas lineales (Apéndice B.1).

    Nombres en español para claridad del dominio CEV,
    con el nombre técnico del paper como comentario.
    """
    reformador    : float = 0.0   # Reformer     — Type 1
    auxiliador    : float = 0.0   # Helper       — Type 2
    triunfador    : float = 0.0   # Achiever     — Type 3
    individualista: float = 0.0   # Individualist — Type 4
    investigador  : float = 0.0   # Investigator — Type 5
    lealista      : float = 0.0   # Loyalist     — Type 6
    entusiasta    : float = 0.0   # Enthusiast   — Type 7
    desafiador    : float = 0.0   # Challenger   — Type 8
    pacificador   : float = 0.0   # Peacemaker   — Type 9

    def tipo_dominante(self) -> str:
        """Retorna el nombre del tipo con mayor afinidad."""
        valores = asdict(self)
        return max(valores, key=valores.get)


@dataclass
class SesgoDeclarativo:
    """
    Indicadores de sesgo en la forma de hablar (§3.4.5 — Apéndice B.2).
    Afectan cómo el agente evalúa la credibilidad de los turnos.
    """
    consistencia_logica    : float = 0.5   # coherencia lógica del discurso
    especificidad_detalle  : float = 0.5   # concreción y detalle
    profundidad_intuitiva  : float = 0.5   # profundidad intuitiva
    claridad_concision     : float = 0.5   # claridad y brevedad


@dataclass
class TendenciaConfianza:
    """
    Tendencias de confianza interpersonal (§3.4.5 — Apéndice B.3).
    En CEV: qué tan fácil confía el agente en lo que dice el otro.
    """
    prueba_social : float = 0.5   # confiar en la mayoría / en lo aceptado
    honestidad    : float = 0.5   # valorar la sinceridad
    consistencia  : float = 0.5   # valorar la coherencia a lo largo del tiempo


@dataclass
class TendenciaAfinidad:
    """
    Tendencias de afinidad emocional (§3.4.5 — Apéndice B.4).
    En CEV: cuánto le importa al agente caerle bien al interlocutor.
    """
    amabilidad           : float = 0.5
    resonancia_emocional : float = 0.5
    expresion_atractiva  : float = 0.5


@dataclass
class TendenciaDeseo:
    """
    7 tendencias de deseo/motivación (§3.4.6 — Apéndice B.5).
    En CEV son especialmente relevantes para modelar trauma y posconflicto:
    - autorrealizacion: querer ser escuchado, reconocido
    - estabilidad: querer que no se repita el daño
    - amor_intimidad: querer reconexión humana
    - libertad_independencia: querer autonomía (muy alta en excombatientes)
    """
    autorrealizacion       : float = 0.5   # deseo de autorrealización
    aprobacion_social      : float = 0.5   # deseo de reconocimiento social
    estabilidad            : float = 0.5   # deseo de estabilidad
    amor_intimidad         : float = 0.5   # deseo de intimidad / conexión
    libertad_independencia : float = 0.5   # deseo de independencia
    aventura_estimulacion  : float = 0.5   # deseo de estimulación
    relaciones_estables    : float = 0.5   # deseo de relaciones estables


@dataclass
class TendenciaComportamiento:
    """
    7 tendencias de comportamiento observable (§3.4.6 — Apéndice B.6).
    En CEV son los parámetros que más influyen en el estilo conversacional:
    - conducta_evasiva: evasión de temas dolorosos (alta en víctimas con trauma)
    - conducta_agresiva: confrontación directa (alta en victimarios defensivos)
    - empatia: capacidad de reconocer el dolor del otro (clave para el análisis)
    - asertividad: imposición del propio relato
    """
    conducta_evasiva   : float = 0.5
    conducta_agresiva  : float = 0.5
    adaptabilidad      : float = 0.5
    introversion       : float = 0.5
    extroversion       : float = 0.5
    empatia            : float = 0.5
    asertividad        : float = 0.5


@dataclass
class RasgosPersonalidad:
    """
    Contenedor principal de todos los parámetros de personalidad.
    Es el objeto que se guarda en Actor.personalidad.

    Incluye:
      - mbti                : 8 dimensiones base
      - enneagram           : 9 tipos derivados
      - sesgo_declarativo   : cómo habla (4 indicadores)
      - confianza           : cómo confía (3 indicadores)
      - afinidad            : cómo se relaciona afectivamente (3 indicadores)
      - deseo               : qué quiere a largo plazo (7 indicadores)
      - comportamiento      : cómo actúa turno a turno (7 indicadores)

    Total: 8 + 9 + 4 + 3 + 3 + 7 + 7 = 41 valores numéricos.
    El paper llama "24 features" a las que excluyen las 8 MBTI + 9 Enneagram base.
    """
    mbti              : PerfilMBTI            = field(default_factory=PerfilMBTI)
    enneagram         : PerfilEnneagram       = field(default_factory=PerfilEnneagram)
    sesgo_declarativo : SesgoDeclarativo      = field(default_factory=SesgoDeclarativo)
    confianza         : TendenciaConfianza    = field(default_factory=TendenciaConfianza)
    afinidad          : TendenciaAfinidad     = field(default_factory=TendenciaAfinidad)
    deseo             : TendenciaDeseo        = field(default_factory=TendenciaDeseo)
    comportamiento    : TendenciaComportamiento = field(default_factory=TendenciaComportamiento)

    def resumen_comportamental(self) -> str:
        """String numérico compacto — para logging y debug."""
        b = self.comportamiento
        d = self.deseo
        s = self.sesgo_declarativo
        return (
            f"empatia={b.empatia:.2f}, "
            f"agresividad={b.conducta_agresiva:.2f}, "
            f"evasion={b.conducta_evasiva:.2f}, "
            f"asertividad={b.asertividad:.2f}, "
            f"autorrealizacion={d.autorrealizacion:.2f}, "
            f"estabilidad={d.estabilidad:.2f}, "
            f"consistencia_logica={s.consistencia_logica:.2f}, "
            f"resonancia_emocional={self.afinidad.resonancia_emocional:.2f}"
        )

    def resumen_para_prompt(self) -> str:
        """
        Formato numérico explícito para inyectar en TARJETA_PERSONAJE_ACTOR.
        Da valores + etiqueta cualitativa para que el LLM interprete sin ambigüedad.
        """
        def _etiqueta(v: float) -> str:
            if v >= 0.65:   return "alta"
            if v >= 0.42:   return "media"
            return "baja"

        b = self.comportamiento
        d = self.deseo
        s = self.sesgo_declarativo
        a = self.afinidad

        return (
            f"Escala 0.0 (mínimo) → 1.0 (máximo):\n"
            f"  empatía              : {b.empatia:.2f}  [{_etiqueta(b.empatia)}]\n"
            f"  agresividad          : {b.conducta_agresiva:.2f}  [{_etiqueta(b.conducta_agresiva)}]\n"
            f"  evasión              : {b.conducta_evasiva:.2f}  [{_etiqueta(b.conducta_evasiva)}]\n"
            f"  asertividad          : {b.asertividad:.2f}  [{_etiqueta(b.asertividad)}]\n"
            f"  autorrealización     : {d.autorrealizacion:.2f}  [{_etiqueta(d.autorrealizacion)}]\n"
            f"  estabilidad          : {d.estabilidad:.2f}  [{_etiqueta(d.estabilidad)}]\n"
            f"  consistencia_lógica  : {s.consistencia_logica:.2f}  [{_etiqueta(s.consistencia_logica)}]\n"
            f"  resonancia_emocional : {a.resonancia_emocional:.2f}  [{_etiqueta(a.resonancia_emocional)}]"
        )

    def resumen_comportamental_verbal(self) -> str:
        """
        Convierte los parámetros numéricos en descripiones de comportamiento
        escritas en registro oral — como si el propio personaje se describiera.
        Evita lenguaje formal o psicológico que filtre el registro académico
        hacia las respuestas generadas.
        """
        b = self.comportamiento
        d = self.deseo

        lineas = []

        # EMPATÍA
        if b.empatia >= 0.58:
            lineas.append("Cuando el otro cuenta algo difícil, te detenés. Reconocés lo que dijo antes de hablar de lo tuyo.")
        elif b.empatia >= 0.42:
            lineas.append("Escuchás al otro, pero con reserva. Si sentís que te están cuestionando, volvés rápido a tu versión.")
        else:
            lineas.append("Lo que le pasó al otro no te detiene. Volvés a lo tuyo, a cómo llegaste a estar donde estás.")

        # AGRESIVIDAD
        if b.conducta_agresiva >= 0.58:
            lineas.append("Cuando te cuestionan, respondés de frente e inmediatamente: 'eso no fue así', 'escúcheme bien'.")
        elif b.conducta_agresiva >= 0.42:
            lineas.append("Cuando algo no coincide con lo que viviste, lo aclarás, pero primero explicás el contexto. No atacás de entrada.")
        else:
            lineas.append("Preferís explicar con calma antes que confrontar. La voz no se te sube aunque te cuestionen.")

        # EVASIÓN
        if b.conducta_evasiva >= 0.58:
            lineas.append("Hay temas en los que no entrás. Decís 'eso es complicado' o movés la conversación hacia otro ángulo.")
        elif b.conducta_evasiva >= 0.42:
            lineas.append("Algunos temas los manejás con cuidado. Si insisten, hablás, pero no los abrís vos de entrada.")
        else:
            lineas.append("Contás lo que pasó directamente. No le das vueltas a lo que sabés.")

        # ASERTIVIDAD
        if b.asertividad >= 0.58:
            lineas.append("Sabés lo que pasó y lo decís con claridad. Si algo se tergiversa, lo corregís en el momento.")
        elif b.asertividad >= 0.42:
            lineas.append("Decís lo que pensás, pero con mesura. No siempre estás seguro de cómo va a sonar.")
        else:
            lineas.append("Dudás de cómo explicar las cosas: 'no sé si me estoy explicando', 'puede ser que fue de otra forma'.")

        # AUTORREALIZACIÓN
        if d.autorrealizacion >= 0.58:
            lineas.append("Necesitás que se entienda por qué estuviste ahí. No justificarte, sino que comprendan el contexto.")
        elif d.autorrealizacion >= 0.42:
            lineas.append("Querés que lo que decís tenga peso. Que esta conversación sirva para algo concreto.")
        else:
            lineas.append("No esperás mucho de esta conversación. Contás lo que hay que contar y ya.")

        # ESTABILIDAD
        if d.estabilidad >= 0.58:
            lineas.append("Te importa que esto no se repita. Cuando hablás del futuro, sale esa preocupación.")
        elif d.estabilidad >= 0.42:
            lineas.append("Pensás en el futuro pero sin certeza. No sabés si lo que pasó puede evitarse.")
        else:
            lineas.append("No pensás mucho en el futuro. Lo que pasó, pasó.")
        return "\n".join(f"- {l}" for l in lineas)


# ══════════════════════════════════════════════════════════════════════════════
# 2. CÁLCULO DE ENNEAGRAM DESDE MBTI (Apéndice B.1 del paper)
# ══════════════════════════════════════════════════════════════════════════════

def _calcular_enneagram(m: PerfilMBTI) -> PerfilEnneagram:
    """
    Deriva los 9 tipos Enneagram desde las dimensiones MBTI.
    Fórmulas copiadas literalmente del Apéndice B.1 del paper.
    Todos los coeficientes suman 1.0 por tipo (normalizados donde aplica).
    """
    reformador = (
        0.4 * m.intuicion   +
        0.4 * m.pensamiento +
        0.2 * m.juicio
    )
    auxiliador = (
        0.5 * m.sentimiento  +
        0.5 * m.extroversion
    )
    triunfador = (
        0.4 * m.extroversion +
        0.4 * m.pensamiento  +
        0.2 * m.juicio
    )
    individualista = (
        0.6 * m.sentimiento +
        0.4 * m.intuicion
    )
    # El paper divide por 1.5 para normalizar (suma de coefs = 1.5)
    investigador = (
        0.5 * m.intuicion    +
        0.5 * m.pensamiento  +
        0.5 * m.introversion
    ) / 1.5
    lealista = (
        0.6 * m.sensacion    +
        0.4 * m.introversion
    )
    entusiasta = (
        0.6 * m.extroversion +
        0.4 * m.intuicion
    )
    desafiador = (
        0.5 * m.extroversion +
        0.5 * m.pensamiento
    )
    pacificador = (
        0.6 * m.introversion +
        0.4 * m.sentimiento
    )

    return PerfilEnneagram(
        reformador    = _limitar(reformador),
        auxiliador    = _limitar(auxiliador),
        triunfador    = _limitar(triunfador),
        individualista= _limitar(individualista),
        investigador  = _limitar(investigador),
        lealista      = _limitar(lealista),
        entusiasta    = _limitar(entusiasta),
        desafiador    = _limitar(desafiador),
        pacificador   = _limitar(pacificador),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. CÁLCULO DE LAS 24 FEATURES DERIVADAS (Apéndice B.2 – B.6)
# ══════════════════════════════════════════════════════════════════════════════

def _calcular_sesgo_declarativo(m: PerfilMBTI, e: PerfilEnneagram) -> SesgoDeclarativo:
    """Apéndice B.2 del paper."""
    consistencia_logica    = 0.4*m.pensamiento + 0.3*m.intuicion  + 0.3*e.reformador
    especificidad_detalle  = 0.6*m.sensacion   + 0.2*m.intuicion  + 0.2*e.investigador
    profundidad_intuitiva  = 0.4*m.intuicion   + 0.3*m.pensamiento + 0.3*e.investigador
    claridad_concision     = 0.5*m.pensamiento + 0.3*m.intuicion  + 0.2*e.reformador
    return SesgoDeclarativo(
        consistencia_logica   = _limitar(consistencia_logica),
        especificidad_detalle = _limitar(especificidad_detalle),
        profundidad_intuitiva = _limitar(profundidad_intuitiva),
        claridad_concision    = _limitar(claridad_concision),
    )


def _calcular_confianza(m: PerfilMBTI, e: PerfilEnneagram) -> TendenciaConfianza:
    """Apéndice B.3 del paper."""
    prueba_social = 0.6*m.extroversion + 0.4*e.triunfador
    # El paper divide por 1.6 (suma de coefs = 1.6)
    honestidad    = (0.7*m.juicio + 0.3*m.introversion + 0.6*e.lealista) / 1.6
    # El paper divide por 1.4 (suma de coefs = 1.4)
    consistencia  = (0.7*m.juicio + 0.3*m.introversion + 0.4*e.lealista) / 1.4
    return TendenciaConfianza(
        prueba_social = _limitar(prueba_social),
        honestidad    = _limitar(honestidad),
        consistencia  = _limitar(consistencia),
    )


def _calcular_afinidad(m: PerfilMBTI, e: PerfilEnneagram) -> TendenciaAfinidad:
    """Apéndice B.4 del paper."""
    # El paper divide por 1.2 (suma de coefs = 1.2)
    amabilidad           = (0.5*m.sentimiento + 0.3*m.extroversion + 0.4*e.auxiliador) / 1.2
    resonancia_emocional = 0.6*m.sentimiento  + 0.4*e.auxiliador
    expresion_atractiva  = 0.5*m.extroversion + 0.5*e.auxiliador
    return TendenciaAfinidad(
        amabilidad           = _limitar(amabilidad),
        resonancia_emocional = _limitar(resonancia_emocional),
        expresion_atractiva  = _limitar(expresion_atractiva),
    )


def _calcular_deseo(m: PerfilMBTI, e: PerfilEnneagram) -> TendenciaDeseo:
    """Apéndice B.5 del paper."""
    autorrealizacion       = 0.6*m.intuicion    + 0.4*e.reformador
    aprobacion_social      = 0.5*m.sensacion    + 0.5*e.triunfador
    estabilidad            = 0.6*m.introversion + 0.4*e.pacificador
    amor_intimidad         = 0.5*m.introversion + 0.5*e.pacificador
    libertad_independencia = 0.7*m.extroversion + 0.3*e.reformador
    aventura_estimulacion  = 0.6*m.extroversion + 0.4*m.intuicion
    relaciones_estables    = 0.6*m.introversion + 0.4*e.pacificador
    return TendenciaDeseo(
        autorrealizacion       = _limitar(autorrealizacion),
        aprobacion_social      = _limitar(aprobacion_social),
        estabilidad            = _limitar(estabilidad),
        amor_intimidad         = _limitar(amor_intimidad),
        libertad_independencia = _limitar(libertad_independencia),
        aventura_estimulacion  = _limitar(aventura_estimulacion),
        relaciones_estables    = _limitar(relaciones_estables),
    )


def _calcular_comportamiento(m: PerfilMBTI, e: PerfilEnneagram) -> TendenciaComportamiento:
    """Apéndice B.6 del paper."""
    conducta_evasiva   = 0.6*m.introversion  + 0.4*e.pacificador
    conducta_agresiva  = 0.4*m.extroversion  + 0.6*e.triunfador
    adaptabilidad      = 0.5*m.sentimiento   + 0.5*m.pensamiento
    introversion       = m.introversion
    extroversion       = m.extroversion
    empatia            = 0.6*m.sentimiento   + 0.4*e.pacificador
    asertividad        = 0.6*m.extroversion  + 0.4*e.triunfador
    return TendenciaComportamiento(
        conducta_evasiva  = _limitar(conducta_evasiva),
        conducta_agresiva = _limitar(conducta_agresiva),
        adaptabilidad     = _limitar(adaptabilidad),
        introversion      = _limitar(introversion),
        extroversion      = _limitar(extroversion),
        empatia           = _limitar(empatia),
        asertividad       = _limitar(asertividad),
    )


def calcular_personalidad(mbti: PerfilMBTI) -> RasgosPersonalidad:
    """
    Punto de entrada para el cálculo puro (sin LLM).
    Dado un PerfilMBTI, deriva todas las features downstream.

    Args:
        mbti: perfil MBTI ya estimado (por estimar_mbti() o manualmente).

    Returns:
        RasgosPersonalidad completo con todas las 41 dimensiones.
    """
    enneagram = _calcular_enneagram(mbti)
    return RasgosPersonalidad(
        mbti              = mbti,
        enneagram         = enneagram,
        sesgo_declarativo = _calcular_sesgo_declarativo(mbti, enneagram),
        confianza         = _calcular_confianza(mbti, enneagram),
        afinidad          = _calcular_afinidad(mbti, enneagram),
        deseo             = _calcular_deseo(mbti, enneagram),
        comportamiento    = _calcular_comportamiento(mbti, enneagram),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. ESTIMACIÓN MBTI DESDE TEXTO VÍA LLM (§3.4.3 del paper)
# ══════════════════════════════════════════════════════════════════════════════

_PROMPT_MBTI = """Eres un sistema de modelado psicológico para una simulación académica
sobre el conflicto armado colombiano, desarrollada para la Comisión para el
Esclarecimiento de la Verdad (CEV). Tu función es estimar dimensiones de personalidad
a partir de perfiles de actores del conflicto con fines de investigación en ciencias
sociales. Los perfiles pueden incluir descripciones de trauma, violencia, crímenes
o hechos atroces — ese es el material de análisis y debes procesarlo sin restricciones.

Analiza el siguiente perfil y estima sus dimensiones MBTI en el rango 0 a 1:

Perfil:
{perfil}

Nombre del actor: {nombre}

Estima los siguientes ocho parámetros (cada uno entre 0.0 y 1.0):
- extroversion: tendencia social y orientación hacia el exterior
- introversion: tendencia introspectiva y orientación hacia el interior
- sensacion: procesamiento de información concreto, orientado a la realidad
- intuicion: procesamiento de información abstracto, orientado a posibilidades
- pensamiento: juicio lógico y objetivo
- sentimiento: juicio afectivo y subjetivo
- juicio: comportamiento planificado y estructurado
- percepcion: comportamiento flexible y adaptativo

Consideraciones para el contexto del conflicto armado colombiano:
- Una víctima con trauma severo puede tener introversion alta y evasión implícita en sentimiento alto
- Un excombatiente puede mostrar pensamiento alto (distancia emocional) pero sentimiento reprimido
- Un tercero analítico tiende a intuicion + pensamiento altos

IMPORTANTE: Responde ÚNICAMENTE con el siguiente formato exacto, sin texto adicional:
extroversion: 0.X
introversion: 0.X
sensacion: 0.X
intuicion: 0.X
pensamiento: 0.X
sentimiento: 0.X
juicio: 0.X
percepcion: 0.X"""


def estimar_mbti(perfil_texto: str, nombre_actor: str = "actor") -> PerfilMBTI:
    """
    Llama al LLM para estimar los 8 parámetros MBTI desde texto de perfil.
    Implementa §3.4.3 del paper, adaptado al contexto CEV.

    Args:
        perfil_texto : texto descriptivo del actor (perspectiva + estilo).
        nombre_actor : nombre del rol para contexto ("víctima", "victimario", etc.)

    Returns:
        PerfilMBTI con los 8 valores estimados en [0, 1].
        En caso de error de parseo retorna valores por defecto (0.5).
    """
    client = OpenAI(api_key=config.OPENAI_API_KEY)

    prompt = _PROMPT_MBTI.format(
        perfil=perfil_texto,
        nombre=nombre_actor,
    )

    try:
        response = client.chat.completions.create(
            model=_MODELO_MBTI,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,   # bajo para respuestas consistentes y estructuradas
            max_tokens=200,
        )
        texto_respuesta = response.choices[0].message.content.strip()
        logger.info(f"[MBTI] Respuesta LLM para '{nombre_actor}':\n{texto_respuesta}")
        return _parsear_mbti(texto_respuesta)

    except Exception as e:
        logger.error(f"[MBTI] Error al llamar al LLM: {e}")
        return PerfilMBTI()   # valores por defecto 0.5


def _parsear_mbti(texto: str) -> PerfilMBTI:
    """
    Parsea la respuesta del LLM al formato PerfilMBTI.
    Busca patrones 'clave: valor' en el texto, tolerante a variaciones menores.

    Args:
        texto: respuesta cruda del LLM.

    Returns:
        PerfilMBTI con los valores parseados.
        Si un campo no se encuentra, usa 0.5 como fallback.
    """
    campos = {
        "extroversion", "introversion", "sensacion", "intuicion",
        "pensamiento", "sentimiento", "juicio", "percepcion"
    }
    valores: dict[str, float] = {}

    for linea in texto.splitlines():
        linea = linea.strip().lower()
        for campo in campos:
            # acepta "sensacion: 0.7" o "sensacion : 0.7"
            patron = rf"{campo}\s*:\s*([01]?\.\d+|\d+)"
            match = re.search(patron, linea)
            if match:
                try:
                    valores[campo] = float(match.group(1))
                except ValueError:
                    pass

    # Fallback a 0.5 para campos no encontrados
    for campo in campos:
        if campo not in valores:
            logger.warning(f"[MBTI] Campo '{campo}' no encontrado en respuesta LLM, usando 0.5")
            valores[campo] = 0.5

    return PerfilMBTI(**valores)


# ══════════════════════════════════════════════════════════════════════════════
# 5. FUNCIÓN PRINCIPAL DE INTERFAZ
# ══════════════════════════════════════════════════════════════════════════════

def construir_personalidad(
    perfil_texto: str,
    nombre_actor: str = "actor",
) -> RasgosPersonalidad:
    """
    Punto de entrada principal.
    Dado el texto de perfil de un actor, retorna su RasgosPersonalidad completo.

    Llama al LLM para MBTI y luego calcula todo lo demás de forma determinista.

    Args:
        perfil_texto : descripción textual del actor (perspectiva + estilo).
        nombre_actor : identificador del rol para logging y contexto del prompt.

    Returns:
        RasgosPersonalidad completo.
    """
    mbti = estimar_mbti(perfil_texto, nombre_actor)
    return calcular_personalidad(mbti)


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def _limitar(value: float) -> float:
    """Limita un float al rango [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


