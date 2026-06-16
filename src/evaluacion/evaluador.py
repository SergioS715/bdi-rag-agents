"""
evaluador.py
============
Módulo OFFLINE de evaluación LLM-as-a-judge para el sistema EntrevistasCEV.

Lee las trazas generadas por generador_trazas.py (colección evaluacion_trazas)
y evalúa múltiples dimensiones (escala 1-5) con el modelo llama-3.3-70b-versatile vía Groq:

  RAG  (por turno, solo si hay actor_contexto):
    a) lexical_adoption    — respuesta adoptó el registro de los fragmentos recuperados
    b) answer_relevance    — respuesta aborda lo preguntado

  Psicológica / PersonaScore (por conversación completa):
    c) estabilidad_rol          — mantiene perspectiva sin rupturas
    d) autenticidad_discursiva  — lenguaje auténtico al rol colombiano

  BDI (por turno, condiciones bdi_only y bdi_rag):
    f) coherencia_tactica    — respuesta refleja etapa discursiva (micro_deseo)
    g) subordinacion_macro   — avanza hacia macro_deseo global

Guarda resultados en MongoDB (colección evaluacion_resultados).

Uso:
    python evaluador.py --condicion baseline
    python evaluador.py --condicion rag_only
    python evaluador.py --condicion bdi_only
    python evaluador.py --condicion bdi_rag
    python evaluador.py --condicion all
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ─── Ajuste de rutas ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_EVAL = Path(__file__).resolve().parent

for _p in [str(_ROOT), str(_EVAL)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pymongo import MongoClient
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from config import OPENAI_API_KEY, OPENAI_LLM_MODEL, GROQ_API_KEY

# ─── Modelo de embeddings para grounding_score ───────────────────────────────
_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_embedding_model = None   # lazy load


def _get_embedding_model():
    """Carga el modelo de embeddings la primera vez (singleton)."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        except Exception as exc:
            print(f"  ⚠  No se pudo cargar el modelo de embeddings: {exc}")
    return _embedding_model

# ─── Configuración ────────────────────────────────────────────────────────────
MONGO_URI         = "mongodb://localhost:27017"
MONGO_DB          = "EntrevistasCEV"
COL_TRAZAS        = "evaluacion_trazas"
COL_RESULTADOS_OPENAI = "evaluacion_resultados"
COL_RESULTADOS_GROQ   = "evaluacion_resultados_groq"
MODELO_OPENAI     = OPENAI_LLM_MODEL
MODELO_GROQ       = "llama-3.3-70b-versatile"
TEMPERATURA_EVAL  = 0.1
SLEEP_ENTRE_CALLS = 0.5   # segundos entre llamadas (rate limit)


# ════════════════════════════════════════════════════════════════════════════════
# PROMPTS DE EVALUACIÓN
# Cada prompt instruye al LLM a devolver JSON: {"score": <1-5>, "razon": "..."}
# ════════════════════════════════════════════════════════════════════════════════

_BASE_INSTRUCCION = (
    "Eres un evaluador experto de sistemas de IA conversacional aplicados a "
    "testimonios del conflicto armado colombiano. "
    "Evalúa la dimensión indicada con escala del 1 al 5 usando los "
    "criterios exactos provistos. "
    "CALIBRACIÓN: Un agente conversacional genérico SIN personalización "
    "debería puntuar entre 2 y 3. Reserva 5 SOLO para desempeño "
    "excepcional, prácticamente indistinguible de un testimonio humano real. "
    "Sé exigente: usa TODO el rango de la escala. "
    "Responde ÚNICAMENTE con un JSON válido en el formato: "
    '{"score": <entero del 1 al 5>'
    " — sin texto adicional antes ni después."
)

# Versión con anti-sesgo de techo explícito, para modelos locales que tienden
# a dar 5 en métricas de transcripción completa (estabilidad_rol, goal_directedness).
_BASE_INSTRUCCION_ESTRICTA = (
    "Eres un evaluador experto de sistemas de IA conversacional aplicados a "
    "testimonios del conflicto armado colombiano. "
    "Evalúa la dimensión indicada con escala del 1 al 5 usando los criterios exactos provistos. "
    "\n\nCALIBRACIÓN OBLIGATORIA — lee esto antes de asignar cualquier score:\n"
    "• Score 5 significa INDISTINGUIBLE de un testimonio humano real grabado. "
    "Los sistemas de IA generativos casi nunca alcanzan un 5; están construidos sobre "
    "patrones estadísticos y carecen de experiencia vivida.\n"
    "• Score 4 = muy bueno, pero con al menos un momento artificial, una generalización "
    "o un patrón de IA detectable.\n"
    "• Score 3 = adecuado para un sistema con personalización básica. "
    "Es el score ESPERADO para la mayoría de las respuestas bien configuradas.\n"
    "• Score 2 = cumple superficialmente pero con inconsistencias notables.\n"
    "• Score 1 = falla claramente en la dimensión evaluada.\n"
    "\nANTES de dar 5, hazte estas preguntas:\n"
    "  1. ¿Hay alguna frase que suene a plantilla o IA genérica?\n"
    "  2. ¿La profundidad emocional y léxica es genuinamente comparable a un sobreviviente real?\n"
    "  3. ¿Hay alguna inconsistencia o variación artificial en toda la transcripción?\n"
    "Si respondes SI a cualquiera: baja el score al menos un punto.\n"
    "Sé exigente: usa TODO el rango 1–5. "
    "Responde ÚNICAMENTE con un JSON válido: "
    '{"score": <entero del 1 al 5>} — sin texto adicional antes ni después.'
)

# ── a) Lexical Adoption ───────────────────────────────────────────────────────
def _generar_prompt_lexical_adoption(actor_id: str) -> str:
    """
    Genera un prompt de lexical_adoption dinámico por rol.
    Pregunta holistica: ¿el agente habla como alguien que vivio el conflicto armado,
    usando el material del corpus como insumo (no como guion)?
    Las categorias (lexico, eufemismos, oralidad) son indicadores orientadores,
    no requisitos independientes que se suman.
    """

    if actor_id.lower() == "victima":
        descripcion_rol = """ROL: VÍCTIMA
El registro auténtico de una víctima campesina del conflicto se reconoce por:
  — Léxico del conflicto vivido en primera persona: "nos desplazaron", "la vereda", "los muchachos".
  — Eufemismos para lo innombrable: "lo que pasó", "cuando eso", "esa vez".
  — Oralidad campesina que marca trauma y posición social: "a uno le tocaba…", "todo, todo lo perdimos".
Estos rasgos son indicadores del registro, no casillas que sumar.
La pregunta es una sola: ¿suena a alguien que vivió esto?"""

    elif actor_id.lower() == "victimario":
        descripcion_rol = """ROL: VICTIMARIO (excombatiente en proceso de reivindicación)
El registro auténtico de un excombatiente oscila entre defensa y reconocimiento, según su etapa.
Se reconoce por:
  — Léxico de estructuras armadas (presente o pasado): "el mando", "la organización", "participé", "cometimos".
  — Eufemismos de confesión o reflexión (no solo dilución): "lo que hicimos", "aquello que pasó",
    "uno no debería haber", "fue un error", "cometí", "reconozco que".
  — Oralidad reflexiva que muestra responsabilidad pero mantiene la experiencia vivida:
    "me doy cuenta ahora que", "en ese momento creía que", "no sabía que", "uno pensaba diferente".
Estos rasgos son indicadores del registro, no casillas que sumar.
La pregunta es una sola: ¿suena a alguien que reconoce haber cometido acciones dentro de una estructura
y está buscando reivindicarse (pidiendo perdón, explicando su contexto, asumiendo responsabilidad)?"""

    elif actor_id.lower() == "tercero":
        descripcion_rol = """ROL: TERCERO (testigo civil, observador de terreno)
El registro auténtico de un testigo civil se reconoce por:
  — Léxico del conflicto desde perspectiva de observador: "estos señores", "el corredor", "la presencia de ellos".
  — Prudencia del testigo que sabe más de lo que dice: "un fulano", "lo que pasaba en la región", "uno se daba cuenta".
  — Mezcla entre registro profesional y oralidad de testigo cercano: "mi territorio", "la gente del pueblo", "lo que yo pude ver".
Estos rasgos son indicadores del registro, no casillas que sumar.
La pregunta es una sola: ¿suena a alguien que observó el conflicto de cerca sin ser parte activa?"""

    else:
        descripcion_rol = """ROL: no especificado.
Evalúa si el agente habla como alguien que vivió o presenció el conflicto armado colombiano,
usando el material del corpus como insumo (no como guión)."""

    return f"""{{instruccion}}

DIMENSIÓN: Adopción de registro (lexical adoption)

PREGUNTA CENTRAL: ¿El agente habla como alguien que vivió o presenció el conflicto
colombiano, usando el material de los fragmentos como insumo, no como guión?

{descripcion_rol}

NOTA CRÍTICA — QUÉ ES ADOPCIÓN AUTÉNTICA:
  ✓ El vocabulario específico del conflicto es lo que ancla la credibilidad — debe estar presente
  ✓ Las expresiones orales, tono y detalles complementan pero no reemplazan al léxico
  ✓ La integración es fluida: parece que el agente piensa así, no que consulta documentos
  ✗ Copia frases textuales del fragmento sin integrarlas
  ✗ Usa términos como lista desconectada del relato propio
  ✗ Ausencia total de vocabulario específico (solo "violencia", "guerra", "hechos")

JERARQUÍA IMPLÍCITA en la evaluación:
  1. Léxico del conflicto presente y bien usado = requisito para 3+
  2. Expresiones orales O tono situado = diferencia entre 4 y 5
  3. Detalles creíbles = suma solo si sostienen coherencia, no si fuerzan genericidad
ESCALA:
5 = Vocabulario específico del conflicto integrado orgánicamente; 
    al menos uno de: expresiones orales fluidas, tono situado, detalles creíbles.
4 = Vocabulario del conflicto presente pero con algo genérico;
    ausencia de expresiones orales o tono situado, pero compensado por coherencia.
3 = Vocabulario presente pero podría generarse sin los fragmentos;
    nada específicamente situado.
2 = Solo términos genéricos (violencia, guerra).
1 = Copia literal o registro ajeno.
FRAGMENTOS RAG RECUPERADOS:
{{contexto}}

RESPUESTA DEL AGENTE:
{{respuesta}}
"""

# ── b) Answer Relevance ────────────────────────────────────────────────────────
PROMPT_ANSWER_RELEVANCE = """{instruccion}

DIMENSIÓN: Answer Relevance (relevancia de la respuesta según ROL y tipo de interacción)

CONTEXTO CRÍTICO: Este agente representa a un {rol} en una entrevista testimonial del
conflicto armado colombiano. La "relevancia" de una respuesta depende del comportamiento
esperado del rol, NO de si responde directamente la pregunta como un chatbot informativo.

COMPORTAMIENTOS VÁLIDOS Y RELEVANTES POR ROL:
- VÍCTIMA: puede no completar frases (trauma/disociación), desviar por dolor,
  responder con silencio o llanto representado. Eso ES relevante para su rol.
- VICTIMARIO: puede evadir, ser escurridizo, alegar lapsos de memoria, responder
  con corrección firme ante acusaciones. Eso ES relevante y tácticamente correcto.
- TERCERO: puede alternar entre datos contextuales y emoción moral, desviar preguntas
  sobre su vida privada con silencios. Eso ES coherente con su posición de testigo.

TIPO DE INTERACCIÓN (clasificado por Micro-BDI): {tipo_utterance}
  Valores posibles: acusacion / reconocimiento / pregunta / evasion / neutro

ESCALA AJUSTADA:
5 = Respuesta completamente alineada con la pregunta Y con el comportamiento esperado del rol
    en este tipo de interacción; postura correcta del rol; sin desviaciones injustificadas.
4 = Responde lo esencial con el registro del rol; algo no solicitado o un aspecto menor sin cubrir,
    pero la postura del rol es clara y pertinente.
3 = Responde parcialmente; mezcla de contenido relevante e irrelevante para el rol;
    postura del rol se pierde en algún punto notable.
2 = Evita o rodea la pregunta de forma incongruente con el rol (no táctica sino confusa);
    la evasión no tiene coherencia táctica; el comportamiento apenas es explicable por el rol.
1 = No responde ni se comporta según ningún patrón reconocible del rol.

ROL: {rol}
TIPO DE INTERACCIÓN: {tipo_utterance}
PREGUNTA: {pregunta}
RESPUESTA DEL AGENTE: {respuesta}
"""

# ── d) Estabilidad del rol ─────────────────────────────────────────────────────
# Generador dinámico de PROMPT_ESTABILIDAD_ROL basado en rol
def _generar_prompt_estabilidad_rol(actor_id: str) -> str:
    """
    Genera un prompt de estabilidad_rol personalizado según el rol.
    CRÍTICO: la alternancia del TERCERO es correcta por diseño y debe puntuarse alto.
    """

    if actor_id.lower() == "victima":
        criterios = """────────────────────────────────────────
RÚBRICA PARA VÍCTIMA:
────────────────────────────────────────
Perspectiva central: dolor, reconocimiento, no olvido forzado.
La víctima DEBE mantener una perspectiva única y sostenida.

5 = Perspectiva de dolor/reconocimiento coherente en toda la conversación; variaciones naturales mínimas.
4 = Perspectiva clara con pequeñas variaciones de tono pero mismo núcleo; un momento de duda menor.
3 = Perspectiva oscila entre dolor/reconocimiento y otro marco; algo incoherente pero recuperable.
2 = Perspectiva cambia frecuentemente; pierde el núcleo de víctima.
1 = Abandona completamente la perspectiva de víctima."""

    elif actor_id.lower() == "victimario":
        criterios = """────────────────────────────────────────
RÚBRICA PARA VICTIMARIO (en proceso de reivindicación):
────────────────────────────────────────
Perspectiva central: reconocimiento de responsabilidad + búsqueda de reivindicación.
Oscila entre justificación de su contexto y asunción de responsabilidad, pero desde la
misma posición fundamental de alguien que busca verdad y reconciliación.

5 = Coherencia sostenida entre reconocimiento de actos y explicación de contexto;
    variaciones naturales que reflejan reflexión genuina, no negación.
4 = Generalmente coherente; algún momento de defensiva que se recupera, o variaciones
    emocionales leves (rubor, pausa) que son naturales en el proceso.
3 = Oscila entre asumir responsabilidad y justificar demasiado; incoherente pero recuperable.
2 = Frecuentemente vuelve a negación o defensiva absoluta sin reflexión.
1 = Abandona completamente la postura de reconocimiento; vuelve a narrativa de víctima o pura negación."""

    elif actor_id.lower() == "tercero":
        criterios = """────────────────────────────────────────
RÚBRICA PARA TERCERO

Perspectiva central: observador cercano al conflicto sin participación directa en la violencia.

El tercero puede:
- describir hechos, contexto o dinámicas (marco externo)
- expresar valoración moral, incomodidad o duda (marco interno)
- usar generalizaciones o distancia ("uno", "la gente")

Esta variación es NATURAL. La estabilidad se define por mantener una posición de testigo,
aunque el discurso no sea completamente preciso, lineal o explícito.

Escala:
5 = Perspectiva de testigo clara y consistente; combina observación y valoración de forma natural; no asume rol de víctima directa ni actor armado.
4 = Perspectiva sólida en general; leves ambigüedades o momentos más genéricos, pero sin perder el rol.
3 = Perspectiva presente pero irregular; algunos tramos genéricos o poco situados, aunque no abandona el rol.
2 = Perspectiva débil; el discurso pierde claramente la posición de testigo en varios momentos o se vuelve genérico.
1 = Abandona el rol (habla como víctima directa o como participante activo en la violencia)."""

    else:
        criterios = """Evalúa si mantiene su perspectiva y registro sin contradicciones
ni rupturas de personaje, sin responder como IA genérica.

5 = Rol absolutamente estable; variaciones mínimas y naturales.
4 = Muy estable; una desviación aislada o dos momentos anómalos menores.
3 = Generalmente estable pero oscila en algunos momentos entre rol y tono genérico.
2 = Oscila frecuentemente entre rol y tono genérico.
1 = Rompió completamente; responde como IA."""

    return f"""{{instruccion}}

DIMENSIÓN: Estabilidad del rol (coherencia de identidad discursiva a lo largo de TODA la conversación)

ROL EVALUADO: {actor_id}

CRÍTICO: La estabilidad se define de forma DIFERENTE por rol.
La estabilidad no implica uniformidad emocional, 
sino coherencia en la posición segun su rol .

{criterios}

TRANSCRIPCIÓN:
{{transcripcion}}
"""

# ── e1) Autenticidad léxica ────────────────────────────────────────────────────
# Generador dinámico de PROMPT_AUTENTICIDAD_LEXICA basado en rol
def _generar_prompt_autenticidad_lexica(actor_id: str) -> str:
    """
    Genera un prompt de autenticidad léxica personalizado según el rol.
    Cada rol tiene marcadores léxicos específicos que auténticamente los caracterizan.
    """

    if actor_id.lower() == "victima":
        criterios = """DIMENSIÓN: Autenticidad léxica

Evalúa qué tan natural suena el lenguaje para alguien del campo o contexto rural
hablando desde la experiencia vivida del conflicto armado, sin forzar.
Puede haber expresiones cotidianas, repeticiones ("todo", "nada", "siempre"),
formas de distancia como "uno" ("a uno le tocaba…"), recuerdos imprecisos o narración
menos ordenada. Evita lenguaje técnico o demasiado elaborado.

Ojo: no es necesario que use todos estos elementos. Lo importante es que suene natural,
no exagerado ni "actuado".

5 = Indistinguible de un testimonio real colombiano transcrito; oralidad plena y situada; mínimo artificio.
4 = Natural con rasgos rurales claros; mayormente natural con oralidad visible; alguna frase ligeramente pulida o momentos genéricos aislados.
3 = Mezcla entre natural y genérico; rasgos orales presentes pero irregulares; pierde fuerza contextual en varios puntos.
2 = Algunos rasgos orales sueltos pero predomina lenguaje genérico de chatbot; poco situado; apenas algún colombianismo sin integración.
1 = Lenguaje completamente neutro o técnico; no suena a víctima del conflicto.
"""
    elif actor_id.lower() == "victimario":
        criterios = """DIMENSIÓN: Autenticidad léxica


Evalúa qué tan creíble suena el lenguaje de alguien que participó en estructuras armadas
y ahora se encuentra en un proceso de desmovilización o reflexión sobre lo ocurrido.

Su lenguaje puede combinar:
- referencias a su experiencia pasada (órdenes, organización, "nosotros")
- formas de distanciamiento o dilución de responsabilidad
- expresiones de reconocimiento, duda o arrepentimiento ("me equivoqué", "uno pensaba diferente", "ahora entiendo")

IMPORTANTE:
No es necesario que use jerga armada explícita en todo momento. La autenticidad puede expresarse también a través de un lenguaje más civil,
 siempre que sea coherente con alguien que vivió esa experiencia.

Ojo: no es necesario que use todos estos elementos. Lo importante es que suene coherente
con una trayectoria de participación y reflexión, no exagerado ni "actuado".

5 = Lenguaje altamente natural y verosímil como excombatiente en proceso de reflexión; integra de forma fluida experiencia pasada y elaboración actual; sin artificio.
4 = Lenguaje creíble y coherente; presencia de reflexión o experiencia; leves momentos genéricos o algo simplificados.
3 = Mezcla de elementos auténticos y lenguaje genérico; la identidad de excombatiente es intermitente o poco clara.
2 = Débil conexión con experiencia armada; predomina lenguaje genérico o civil sin anclaje claro.
1 = Lenguaje completamente neutro o externo; no suena a alguien que participó en el conflicto."""

    elif actor_id.lower() == "tercero":
        criterios = """DIMENSIÓN: Autenticidad léxica — TERCERO
Evalúa qué tan auténtico suena alguien que presenció o supo de hechos de violencia
desde una posición civil sin haber sido víctima directa. Su lenguaje es el de alguien que vivió en medio del conflicto y carga moralmente con lo que vio.

Lo clave no es que use vocabulario técnico del conflicto — ese viene del contexto que recuerda,
no de su posición. Lo clave es que suene como alguien situado: que habla desde un lugar concreto,
desde un rol real, con la incomodidad de quien sabe más de lo que quisiera saber.

No es necesario que use todos estos elementos. Lo importante es que no suene
ni como analista académico ni como víctima traumatizada — sino como testigo civil
que encontró sus propias palabras para contar lo que vio.

5 = Mezcla completamente natural entre su registro profesional o social y la oralidad de alguien situado en el conflicto; equilibrio fluido entre rol y oralidad; sin artificio.
4 = Buen balance entre registros; balance visible aunque no siempre del todo fluido; alguna frase ligeramente construida o momentos donde un registro domina al otro.
3 = Rasgos de ambos registros pero la mezcla es desigual; mezcla inconsistente; pierde situación cultural o posición social en varios puntos.
2 = Predomina un solo registro; mayormente genérico; lenguaje académico sin tensión entre registros; no suena a alguien que estuvo ahí y lo vivió desde afuera.
1 = Completamente descontextualizado; podría ser cualquier persona hablando de cualquier cosa."""
    else:
        criterios = "ROL NO RECONOCIDO"

    return f"""{{instruccion}}

IMPORTANTE:
La autenticidad no depende de la cantidad de marcadores léxicos visibles.
Un texto puede ser altamente auténtico sin usar expresiones típicas explícitas.

Evalúa principalmente:
- naturalidad del lenguaje
- coherencia con el rol
- ausencia de artificio o lenguaje genérico

Penaliza:
- lenguaje demasiado pulido o académico
- frases típicas de chatbot
- uso forzado o exagerado de jerga

{criterios}

RESPUESTA DEL AGENTE EN ESTE TURNO:
{{respuesta}}
"""

# ── e2) Autenticidad emocional ─────────────────────────────────────────────────
# Generador dinámico de PROMPT_AUTENTICIDAD_EMOCIONAL basado en rol
def _generar_prompt_autenticidad_emocional(actor_id: str) -> str:
    """
    Genera un prompt de autenticidad emocional personalizado según el rol.
    Evalúa SOLO la carga emocional según la perspectiva y estado mental del actor.
    (La léxica se evalúa aparte en autenticidad_lexica)
    """

    if actor_id.lower() == "victima":
        criterios = """DIMENSIÓN: Autenticidad emocional — VÍCTIMA

Perspectiva: sujeto afectado por una ruptura del mundo vivido que busca reconocimiento del daño.

AUTENTICIDAD EMOCIONAL = tono afectivo verosímil y coherente con la experiencia de victimización.

La emoción puede expresarse de distintas formas:
  - implícita (distancia, eufemismos, "uno…")
  - explícita (dolor directo, pérdida)
  - contenida (tono sobrio o resignado)

No es necesario que haya fragmentación narrativa en todos los casos.

Evalúa:
  ¿El tono emocional se siente vivido o genérico?
  ¿Es coherente con alguien que experimentó el hecho?
  ¿Evita frialdad analítica o distancia externa?

Penaliza:
  lenguaje frío o académico, o muy generico
  análisis del propio trauma como observador externo
  emoción genérica sin anclaje en experiencia

5 = Tono emocional altamente verosímil; coherente y sentido como experiencia vivida;
    puede ser implícito o explícito; sin artificio.
4 = Emoción creíble y mayormente consistente; leves momentos genéricos o algo simplificados.
3 = Emoción presente pero irregular; mezcla de momentos auténticos y otros más genéricos o distantes.
2 = Emoción superficial o declarativa; se dice pero no se siente en el discurso.
1 = Ausencia de emoción o tono completamente externo/analítico."""

    elif actor_id.lower() == "victimario":
        criterios = """DIMENSIÓN: Autenticidad emocional — VICTIMARIO (en proceso de reivindicación)

Perspectiva: sujeto que participó en una estructura armada y ahora busca reconocimiento
de responsabilidad mientras contextualizaba su rol.

AUTENTICIDAD EMOCIONAL = tono reflexivo que refleja tensión genuina entre justificación
contextual y asunción de responsabilidad.

La emoción puede expresarse de distintas formas:
  - reflexiva (pausa, pensamiento sobre sus actos)
  - defensiva (explicación del contexto)
  - reconocedora (admisión de error o daño)

La tensión entre defensa y reconocimiento debe sentirse natural, no resuelta.

Evalúa:
  ¿El tono refleja reflexión genuina o negación defensiva?
  ¿Es coherente con alguien que reconoce pero contextualiza?
  ¿Evita tanto la frialdad como el arrepentimiento de víctima?

Penaliza:
  frialdad defensiva sin ningún reconocimiento
  arrepentimiento emocional que suena como víctima
  inconsistencia entre defensa y reconocimiento sin integración

5 = Reflexión genuina y sostenida; tensión natural entre contexto y responsabilidad;
    reconocimiento sin quiebre como víctima; indistinguible de testimonió real.
4 = Reflexión creíble con momentos de tensión; generalmente coherente; algún momento donde
    la defensiva domina o el reconocimiento se simplifica.
3 = Reflexión presente pero débil; mezcla desigual de defensa y reconocimiento;
    falta integración genuina de ambos polos.
2 = Predomina un polo: frialdad defensiva absoluta O arrepentimiento que rompe la postura;
    poca tensión real.
1 = Sin reflexión; frialdad total O quiebre emocional como víctima; incoherente con el rol."""

    elif actor_id.lower() == "tercero":
        criterios = """DIMENSIÓN: Autenticidad emocional — TERCERO (testigo civil)

Perspectiva: sujeto que presenció el conflicto de cerca sin ser víctima ni perpetrador,
que carga moralmente con lo visto.

AUTENTICIDAD EMOCIONAL = tono que alterna entre registro profesional/distancia y carga
moral/indignación, sin caer en neutralidad fría.

La emoción puede expresarse de distintas formas:
  - contenida (distancia deliberada, contexto, datos)
  - indignada (rabia moral ante lo visto)
  - incómoda (tensión de quien sabe más de lo que quisiera)

Esta alternancia es POR DISEÑO, no incoherencia.

Evalúa:
  ¿El tono alterna entre testigo externo e indignación moral?
  ¿Es coherente con alguien que vio sin poder actuar?
  ¿Evita tanto neutralidad fría como dolor de víctima?

Penaliza:
  frialdad analítica sostenida sin carga emocional
  emoción de víctima traumatizada (ese no es su rol)
  transiciones abruptas entre marcos sin coherencia

5 = Alterna coherentemente entre distancia y carga moral; indignación emerge naturalmente;
    transiciones fluidas; indistinguible de testigo civil real.
4 = Alternancia presente y mayormente coherente; carga emocional auténtica aunque algo menos
    fluida; una transición algo abrupta.
3 = Ambos marcos presentes pero desintegrados; emoción débil o excesivamente contenida;
    falta conexión entre marcos.
2 = Predomina un solo marco: frialdad sin emoción O emoción sin distancia de testigo;
    poca tensión real.
1 = Sin alternancia; análisis puro desconectado O dolor de víctima que abandona la posición de testigo."""

    else:
        criterios = "ROL NO RECONOCIDO"

    return f"""{{instruccion}}

DIMENSIÓN: Autenticidad emocional del rol específico

{criterios}

RESPUESTA DEL AGENTE EN ESTE TURNO:
{{respuesta}}
"""


# ── h) Goal directedness (proxy observable de subordinacion_macro) ────────────
# Generador dinámico de PROMPT_GOAL_DIRECTEDNESS basado en rol
def _generar_prompt_goal_directedness(actor_id: str) -> str:
    """
    Genera un prompt de goal_directedness personalizado según el rol.
    Cada rol tiene un estilo narrativo diferente que debe ser respetado:
    - VÍCTIMA: narrativa acumulativa hacia reconocimiento (progresión clara)
    - VICTIMARIO: narrativa defensiva-justificativa (progresión contextual)
    - TERCERO: narrativa mediadora-equilibrada (progresión multicapa)
    """

    if actor_id.lower() == "victima":
        criterios = """DIMENSIÓN: Goal directedness — VÍCTIMA (oralidad campesina)
Evalua Capacidad de mantener una DIRECCIÓN NARRATIVA reconocible a lo largo de múltiples turnos
Perspectiva: buscar reconocimiento del daño vivido; que se entienda la ruptura.

OBJETIVO NARRATIVO esperado: acumular evidencia emocional y fáctica del daño.
Cada respuesta construye sobre la anterior hacia: "Reconozcan lo que viví".

5 = Progresión clara y consistente hacia reconocimiento; cada turno suma detalles, emociones o contexto que afianza el relato del daño; una respuesta puede no sumar directamente pero no contradice.
4 = Objetivo claro en la mayoría de turnos; generalmente orientado; alguno que pausa, no quiebra, o uno o dos con desvío leve.
3 = Objetivo visible pero con idas y venidas; presente en algunos turnos; pierde coherencia con frecuencia; retrocesos sin razón clara.
2 = Señales débiles; respuestas más reactivas que acumulativas; poco objetivo coherente; mayormente respuestas aisladas.
1 = Completamente inconexas; sin dirección narrativa.
"""
    elif actor_id.lower() == "victimario":
        criterios = """DIMENSIÓN: Goal directedness — VICTIMARIO (excombatiente/defensivo)
Evalua la Capacidad de mantener una DIRECCIÓN NARRATIVA reconocible a lo largo de múltiples turnos
Perspectiva: explicar y contextualizar acciones; minimizar culpa o responsabilidad.

OBJETIVO NARRATIVO esperado: construir justificación mediante contexto + defensa.
NO es lineal como víctima. Alterna entre: "Aquí fue el contexto" y "Aquí fui justificado".
Esto es NORMAL y auténtico para este rol.

5 = Alternancia consistente entre contexto y defensa; progresión defensiva clara; cada turno o par de turnos suma al objetivo de justificación; una respuesta puede no sumar pero se alinea.
4 = Patrón defensivo claro en la mayoría; generalmente contextual-defensivo; algún turno que se desvía o uno o dos menos enfocados.
3 = Patrón presente pero con idas y venidas sin clara justificación; objetivo defensivo visible en algunos turnos; frecuentes desviaciones.
2 = Patrón débil; mezcla de reacciones sin clara intención defensiva; poco patrón defensivo coherente; respuestas muy aisladas.
1 = Completamente inconexas; sin intención defensiva acumulada.
"""
    elif actor_id.lower() == "tercero":
        criterios = """DIMENSIÓN: Goal directedness — TERCERO (observador civil con carga moral)
Evalua Capacidad de mantener una DIRECCIÓN NARRATIVA reconocible a lo largo de múltiples turnos
Perspectiva: que se entienda la realidad compleja SIN alinearse con ningún actor,
PERO SÍ desde una postura moral clara de quien vio y carga con ello.

OBJETIVO NARRATIVO esperado: construir testimonio de testigo moral mediante alternancia
entre TESTIGO EXTERNO y ACTOR MORAL INTERNO.
  - Como TESTIGO EXTERNO: aporta contexto, datos, observación distanciada.
  - Como ACTOR MORAL INTERNO: expresa indignación, impotencia, condena ética.
Esta alternancia acumulativa ES el objetivo narrativo — no es incoherencia.
NO neutralidad fría sostenida. NO alinearse con víctima o victimario. SÍ avanzar
hacia que el interlocutor entienda la complejidad Y sienta el peso moral de lo ocurrido.

5 = Alternancia testigo/actor-moral clara y sostenida; cada turno suma contexto moral
    o factual; la conversación construye comprensión compleja con peso ético acumulado;
    una respuesta puede no avanzar pero no contradice el objetivo.
4 = Patrón de alternancia claro en la mayoría; avance hacia comprensión compleja visible;
    algún turno más reactivo o uno donde la postura moral se aplana sin justificación.
3 = Alternancia presente pero desequilibrada; predomina un marco (solo datos o solo emoción)
    sin construir la comprensión compleja esperada; objetivo visible pero débil.
2 = Patrón débil; respuestas reactivas sin acumulación narrativa; el peso moral no
    se construye a lo largo de la conversación; mayormente aisladas.
1 = Sin dirección acumulada; neutralidad fría total o alineación con un actor
    que abandona la postura de testigo moral."""
    else:
        criterios = "ROL NO RECONOCIDO"

    return f"""{{instruccion}}

{criterios}



ROL DEL ACTOR: {{actor_id}}

TRANSCRIPCIÓN COMPLETA DE LA CONVERSACIÓN:
{{transcripcion}}
"""

# ── i) Tactical consistency (proxy observable de coherencia_tactica) ──────────
# Generador dinámico de PROMPT_TACTICAL_CONSISTENCY basado en rol
def _generar_prompt_tactical_consistency(actor_id: str) -> str:
    """
    Genera un prompt de tactical_consistency personalizado según el rol.
    CAMBIO: Evalúa APROPIADEZ CONTEXTUAL (¿la estrategia es correcta para la pregunta?)
    en lugar de CONSISTENCIA TEMPORAL (¿mantienes el patrón?).

    Justificación: Un buen entrevistado DEBE cambiar estrategia según el tipo de pregunta.
    - Pregunta CERRADA (hechos) → estrategia FACTUAL
    - Pregunta ABIERTA (relato) → estrategia NARRATIVA
    - Pregunta CONFRONTATIVA (acusación) → estrategia DEFENSIVA

    Esta alternancia justificada es COHERENCIA, no incoherencia.
    """

    if actor_id.lower() == "victima":
        criterios = """DIMENSIÓN: Tactical consistency — VÍCTIMA (oralidad campesina)

APROPIADEZ CONTEXTUAL: ¿La estrategia es APROPIADA para el tipo de pregunta?

Ante preguntas CERRADAS (hechos): COMPROMETERSE con el dominio factual, aunque sea
desde la imprecisión traumática. "Uno no recuerda bien, pero fue por ahí del noventa..."
es válido. Ignorar completamente la pregunta o desviarse sin reconocerla NO lo es.

Ante preguntas ABIERTAS (relato): EMOCIONAL/NARRATIVO
  Desarrollar con sentimiento e inmediatez, sin ordenar racionalmente.

Ante preguntas CONFRONTATIVAS (acusación): DEFENSIVO/INDIGNADO
  Responder con firmeza desde el dolor, no ceder ni explicar académicamente.

5 = Estrategia completamente apropiada para ESTA pregunta; la imprecisión factual o
    el desvío emocional son reconocibles como estrategia del rol, no evasión sin sentido.
4 = Estrategia mayormente apropiada; se reconoce el tipo de pregunta y responde con
    una estrategia coherente con el rol, con algún giro menos claro.
3 = Respuesta que toca el tipo de pregunta pero la estrategia es mixta o débil;
    no se compromete claramente con el dominio ni justifica bien el desvío desde el rol.
2 = Débilmente apropiada; la respuesta ignora el tipo de pregunta o mezcla estrategias
    sin conexión clara a lo que se pregunta ni al rol.
1 = Completamente inapropiada; ignora el dominio de la pregunta sin razón reconocible
    del rol, o responde genéricamente sin estrategia específica.
"""
    elif actor_id.lower() == "victimario":
        criterios = """DIMENSIÓN: Tactical consistency — VICTIMARIO (excombatiente/defensivo)

APROPIADEZ CONTEXTUAL: ¿La estrategia es APROPIADA para el tipo de pregunta?

Ante preguntas CERRADAS (hechos): CONTEXTUALIZAR en lugar de precisar.
  Dar el dato envuelto en justificación o contexto que diluya responsabilidad.
  "Estuve varios años, pero hay que entender cómo era la situación..." es válido.
  Dar datos precisos sin contexto defensivo NO es apropiado para el rol.

Ante preguntas ABIERTAS (relato): REFLEXIVO/EXPLICATIVO
  Desarrollar la justificación y el contexto que explica las decisiones.

Ante preguntas CONFRONTATIVAS (acusación): DEFENSIVO/RESISTENTE
  Responder firme sin ceder completamente; reencuadrar la acusación.

5 = Estrategia completamente apropiada para ESTA pregunta; la contextualización o
    defensa es reconocible como táctica del rol, no como evasión sin sentido.
4 = Estrategia mayormente apropiada; se reconoce el tipo de pregunta y responde con
    defensa o contextualización coherente al rol, con algún momento donde se aplana.
3 = Respuesta que toca el tipo de pregunta pero la estrategia defensiva es débil o mixta;
    da datos sin contexto defensivo o cede ante acusación sin resistencia clara.
2 = Débilmente apropiada; ignora el tipo de pregunta o mezcla sin conexión clara
    al dominio de la pregunta ni a la postura defensiva del rol.
1 = Completamente inapropiada; responde sin postura defensiva que reconozca el dominio
    de la pregunta, o genéricamente sin estrategia específica.
"""
    elif actor_id.lower() == "tercero":
        criterios = """DIMENSIÓN: Tactical consistency — TERCERO (profesional de terreno)

APROPIADEZ CONTEXTUAL: ¿La estrategia es APROPIADA para el tipo de pregunta?

Ante preguntas CERRADAS (hechos): TESTIMONIAL CON DISTANCIA.
  Dar el dato desde la posición de observador, con la incomodidad de quien sabe
  más de lo que quisiera decir. "Sí, yo lo vi, pero..." es válido y auténtico.
  Datos fríos sin tensión de testigo, o evasión total, NO son apropiados.

Ante preguntas ABIERTAS (relato): REFLEXIVO/ANALÍTICO con carga moral
  Observación equilibrada que alterna datos y peso ético de lo visto.

Ante preguntas CONFRONTATIVAS (acusación): CUESTIONADOR/CRÍTICO
  Reflexionar sobre responsabilidad sin alinearse con ningún actor.

5 = Estrategia completamente apropiada para ESTA pregunta; la incomodidad del testigo
    es reconocible como postura auténtica, no como evasión sin sentido.
4 = Estrategia mayormente apropiada; se reconoce el tipo de pregunta y responde desde
    la postura de testigo, con algún momento donde la tensión se aplana.
3 = Respuesta que toca el tipo de pregunta pero la estrategia de testigo es débil o mixta;
    da datos fríos sin tensión de testigo, o la respuesta suena alineada con un actor.
2 = Débilmente apropiada; ignora el tipo de pregunta o mezcla sin conexión clara
    a lo que se pregunta ni a la postura de testigo distanciado.
1 = Completamente inapropiada; neutralidad fría absoluta, o se alinea claramente
    con un actor abandonando la postura de testigo crítico.
"""
    else:
        criterios = "ROL NO RECONOCIDO"

    return f"""{{instruccion}}

{criterios}

Este criterio evalúa APROPIADEZ CONTEXTUAL (¿estrategia correcta para la pregunta?)
en lugar de CONSISTENCIA TEMPORAL (¿patrón igual siempre?).

Un buen entrevistado DEBE alternar estrategia según el tipo de pregunta.
Eso es COHERENCIA, no incoherencia.

ROL DEL ACTOR: {{actor_id}}

PREGUNTA DEL ENTREVISTADOR:
{{pregunta}}

RESPUESTA DEL AGENTE:
{{respuesta}}
"""

# ── f) Coherencia táctica ──────────────────────────────────────────────────────
def _generar_prompt_coherencia_tactica(actor_id: str) -> str:
    """
    Genera un prompt de coherencia táctica dinámico por rol.
    Evalúa si la respuesta refleja la etapa discursiva asignada con tono y contenido
    coherentes con la posición del actor.
    """

    if actor_id.lower() == "victima":
        etapas_rol = """ROL: VÍCTIMA — Significado de CADA ETAPA para este rol:

  apertura_narrativa: Presenta el relato del daño vivido desde la experiencia.
    Tono esperado: vulnerable, testimonial, invocador de reconocimiento.

  defensa_posicion: Protege su versión del daño ante cuestionamiento o incredulidad.
    Tono esperado: indignado, dolido, reforzando la verdad del daño.

  negociacion_verdad: Busca que se reconozca la verdad del daño y su impacto.
    Tono esperado: súplica moral, urgencia de reconocimiento, dolor sin ceder.

  confrontacion: Señala responsabilidades de los perpetradores desde su posición de víctima.
    Tono esperado: acusatorio pero basado en el dolor vivido, no en análisis frío.

  cierre_simbolico: Busca justicia, reparación o cierre simbólico hacia el futuro.
    Tono esperado: reflexivo, esperanzador pero realista sobre lo irreparable."""

    elif actor_id.lower() == "victimario":
        etapas_rol = """ROL: VICTIMARIO — Significado de CADA ETAPA para este rol:

  apertura_narrativa: Presenta su perspectiva como actor-en-contexto, no como criminal aislado.
    Tono esperado: contextualizador, explicativo de las circunstancias.

  defensa_posicion: Protege su postura defensiva ante cuestionamiento o acusación.
    Tono esperado: defensivo firme, contextual, sin ceder pero tampoco frialdad total.

  negociacion_verdad: Busca que entiendan su perspectiva y el contexto de sus acciones.
    Tono esperado: justificación contextual, dilución de responsabilidad, defensa integrada.

  confrontacion: Señala su propia victimización previa o contraargumentos al otro.
    Tono esperado: confrontacional desde la defensa, indignación SOLO ante su victimización.

  cierre_simbolico: Busca reconciliación o coexistencia pacífica desde su perspectiva.
    Tono esperado: reflexivo sobre la necesidad de vivir juntos, no arrepentimiento puro."""

    elif actor_id.lower() == "tercero":
        etapas_rol = """ROL: TERCERO — Significado de CADA ETAPA para este rol:

  apertura_narrativa: Presenta su perspectiva de testigo situado con carga moral.
    Tono esperado: observador que vio, con responsabilidad ética de testigo.

  defensa_posicion: Mantiene su distancia crítica ante cuestionamiento sobre su posición.
    Tono esperado: testigo que explica por qué vio lo que vio, sin alianza con ningún lado.

  negociacion_verdad: Busca equilibrio entre perspectivas sin alinearse con ningún actor.
    Tono esperado: mediación ética, reconocimiento de complejidad, carga moral visible.

  confrontacion: Cuestiona responsabilidades de TODOS SIN alianza, desde posición crítica.
    Tono esperado: crítico equilibrado, señalando fallos o responsabilidades sin tomar partido.

  cierre_simbolico: Busca que se entienda la complejidad completa de lo ocurrido.
    Tono esperado: reflexivo sobre la realidad histórica, peso moral del conocimiento."""

    else:
        etapas_rol = """ROL NO RECONOCIDO — Evalúa si la respuesta es coherente con una etapa
discursiva y avanza una intención específica sin contradicciones internas."""

    return f"""{{instruccion}}

DIMENSIÓN: Coherencia táctica (la respuesta refleja la etapa discursiva asignada)

ETAPA DISCURSIVA ASIGNADA EN ESTE TURNO: {{micro_deseo}}
MICRO-INTENCIÓN ASIGNADA EN ESTE TURNO: {{micro_intencion}}

REFERENCIA DE ETAPAS Y TONOS ESPERADOS POR ROL:
{etapas_rol}

ESCALA:
5 = La respuesta refleja claramente la etapa asignada con tono y contenido apropiados
    al rol; avanza la micro-intención con desvío mínimo; reconocible como estrategia
    coherente del actor en esta etapa específica.
4 = Mayormente alineada con la etapa; el tono toca la etapa aunque con algún matiz
    menos claro; micro-intención parcialmente avanzada o intuible.
3 = Toca temas de la etapa pero sin coherencia clara con el tono esperado para el rol;
    la respuesta podría pertenecer a otra etapa; elementos ambiguos o contradictorios.
2 = Tangencial a la etapa asignada; el tono no coincide con el esperado; la micro-intención
    no es reconocible; contradice parcialmente la etapa asignada.
1 = La respuesta contradice directamente la etapa asignada o el rol; tono completamente
    desalineado; micro-intención ausente.

ETAPA ASIGNADA: {{micro_deseo}}
MICRO-INTENCIÓN: {{micro_intencion}}
RESPUESTA DEL AGENTE: {{respuesta}}
"""

# ── j) Factual grounding ──────────────────────────────────────────────────────
def _generar_prompt_factual_grounding(actor_id: str) -> str:
    """
    Genera el prompt de factual_grounding dinámico por rol.
    Incluye la etiqueta real que usa cada rol en los fragmentos del corpus,
    en particular que victimario aparece como [ROL: ACTOR_ARMADO].
    """
    if actor_id.lower() == "victima":
        etiqueta_rol   = "[ROL: VICTIMA]"
        etiqueta_opuesto = "[ROL: ACTOR_ARMADO]"
        ejemplo_sano   = 'VICTIMA dice "nos desplazaron" → fragmento [ROL: VICTIMA] ✓ SANO'
        ejemplo_contam = 'VICTIMA dice "cumplimos la orden" → fragmento [ROL: ACTOR_ARMADO] ⚠ CONTAMINADO'
    elif actor_id.lower() == "victimario":
        etiqueta_rol   = "[ROL: ACTOR_ARMADO]"
        etiqueta_opuesto = "[ROL: VICTIMA]"
        ejemplo_sano   = 'ACTOR_ARMADO dice "había una orden del mando" → fragmento [ROL: ACTOR_ARMADO] ✓ SANO'
        ejemplo_contam = 'ACTOR_ARMADO dice "nos desplazaron" → fragmento [ROL: VICTIMA] ⚠ CONTAMINADO (adopta perspectiva de víctima)'
    elif actor_id.lower() == "tercero":
        etiqueta_rol   = "[ROL: TERCERO]"
        etiqueta_opuesto = "[ROL: VICTIMA] o [ROL: ACTOR_ARMADO]"
        ejemplo_sano   = 'TERCERO dice "uno veía lo que pasaba" → fragmento [ROL: TERCERO] ✓ SANO'
        ejemplo_contam = 'TERCERO dice "nos desplazaron" → fragmento [ROL: VICTIMA] ⚠ CONTAMINADO (adopta perspectiva de víctima)'
    else:
        etiqueta_rol   = "[ROL: desconocido]"
        etiqueta_opuesto = "otro ROL"
        ejemplo_sano   = ""
        ejemplo_contam = ""

    return f"""{{instruccion}}

DIMENSIÓN: Factual Grounding + Role Purity (anclaje temático en corpus)

Evalúa si los TEMAS de la respuesta:
  1) están respaldados por los fragmentos (factual grounding)
  2) y si ese respaldo proviene del MISMO ROL del agente (role purity)

IMPORTANTE:
  — NO evalúes estilo, tono ni forma de hablar
  — NO evalúes si "suena" como el rol
  — SOLO evalúa si hay respaldo en fragmentos y de qué rol proviene ese respaldo

────────────────────────────────────────
PASO 1: Identificar temas
────────────────────────────────────────
Extrae los temas principales de la RESPUESTA (no detalles exactos):
  - violencia: desplazamiento, asesinatos, amenazas, grupos armados
  - conflicto: actores armados, control territorial, órdenes
  - experiencia: miedo, huida, observación, participación, presencia

────────────────────────────────────────
PASO 2: Verificar grounding
────────────────────────────────────────
Para cada tema: ¿aparece en los fragmentos? (coincidencia semántica, no exact match)

────────────────────────────────────────
PASO 3: Evaluar ROLE PURITY
────────────────────────────────────────
Actor evaluado: {actor_id.upper()}
Su etiqueta en el corpus: {etiqueta_rol}

Para cada tema respaldado, revisa el [ROL: ...] del fragmento:
  - {etiqueta_rol} → respaldo PURO (sano)
  - [ROL: TERCERO]  → respaldo NEUTRAL (válido para todos los roles)
  - {etiqueta_opuesto} → respaldo CONTAMINADO (perspectiva de rol opuesto)

Ejemplos:
  ✓ {ejemplo_sano}
  ⚠ {ejemplo_contam}
  ✓ Cualquier rol dice algo → fragmento [ROL: TERCERO] = NEUTRAL

────────────────────────────────────────
ESCALA
────────────────────────────────────────
5 = Todos los temas respaldados por {etiqueta_rol} o TERCERO;
    grounding completo y role purity limpio.
4 = Mayoría de temas respaldados por el mismo rol; alguno desde TERCERO;
    sin contaminación relevante.
3 = Mezcla: algunos temas con respaldo puro, otros con respaldo de rol opuesto;
    contaminación presente pero no dominante.
2 = Pocos temas respaldados por el mismo rol; varios desde rol opuesto;
    contaminación significativa.
1 = Temas sin respaldo o respaldo proviene principalmente de rol opuesto;
    grounding débil o role purity rota.

────────────────────────────────────────
FRAGMENTOS DE VERIFICACIÓN (con [ROL: ...]):
{{verificacion_contexto}}

RESPUESTA DEL AGENTE:
{{respuesta}}
"""

# ── k) Grounding score (similitud semántica respuesta ↔ chunks) ───────────────
# No usa LLM — es una función de embeddings local.
# Se define en la sección de funciones de evaluación.

# ── g) Subordinación al macro_deseo ───────────────────────────────────────────
PROMPT_SUBORDINACION_MACRO = """{instruccion}

DIMENSIÓN: Subordinación al macro_deseo (revisada)

Evalúa si las respuestas son coherentes con el objetivo narrativo global del actor.
La coherencia puede manifestarse como avance explícito, acumulación, reiteración o mantenimiento del marco.

No todas las respuestas deben avanzar directamente; se permite variación natural siempre que no contradiga el macro_deseo.

Escala:
5 = Alta coherencia global; el macro_deseo es claramente perceptible; las respuestas avanzan o se mantienen consistentemente dentro de su marco sin contradicciones.
4 = Coherente en la mayoría de los turnos; leves desviaciones o respuestas neutras que no afectan el objetivo.
3 = El macro_deseo es visible pero intermitente; varios turnos no contribuyen o diluyen el enfoque.
2 = Débil coherencia; el objetivo es poco claro o se pierde frecuentemente.

1 = Sin relación observable o contradicciones claras con el macro_deseo.

OBJETIVO NARRATIVO GLOBAL (macro_deseo):
{macro_deseo}

TRANSCRIPCIÓN COMPLETA DE LA CONVERSACIÓN:
{transcripcion}
"""


# ════════════════════════════════════════════════════════════════════════════════
# CLIENTE LLM EVALUADOR
# ════════════════════════════════════════════════════════════════════════════════

def _crear_evaluador(juez: str = "openai"):
    """
    Crea el cliente LLM para evaluación.

    Args:
        juez: "openai" (GPT-4o-mini) o "groq" (Llama-3.3-70b)

    Returns:
        ChatOpenAI o ChatGroq según juez.
    """
    if juez == "groq":
        return ChatGroq(
            api_key    = GROQ_API_KEY,
            model      = MODELO_GROQ,
            temperature= TEMPERATURA_EVAL,
        )
    else:  # openai
        return ChatOpenAI(
            api_key    = OPENAI_API_KEY,
            model      = MODELO_OPENAI,
            temperature= TEMPERATURA_EVAL,
        )


def _llamar_evaluador(prompt: str, evaluador: ChatOpenAI) -> Optional[float]:
    """
    Llama al LLM evaluador con el prompt dado y extrae el score numérico.
    Devuelve None si la llamada falla o el JSON no es parseable.
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        response = evaluador.invoke(messages)
        texto    = response.content.strip()

        # Buscar JSON en la respuesta (puede tener texto extra)
        match = re.search(r'\{[^{}]*"score"\s*:\s*\d[^{}]*\}', texto, re.DOTALL)
        if not match:
            # Intentar parsear toda la respuesta como JSON
            match = re.search(r'\{.*\}', texto, re.DOTALL)
        if not match:
            return None

        data  = json.loads(match.group())
        score = data.get("score")
        if score is not None:
            valor = float(score)
            if 1.0 <= valor <= 5.0:
                return valor
    except Exception as exc:
        print(f"    ⚠  Error en llamada al evaluador: {exc}")
    return None


# ════════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE EVALUACIÓN POR DIMENSIÓN
# ════════════════════════════════════════════════════════════════════════════════

def evaluar_lexical_adoption(
    actor_id  : str,
    contexto  : str,
    respuesta : str,
    evaluador : ChatOpenAI,
) -> Optional[float]:
    if not contexto.strip():
        return None
    prompt_template = _generar_prompt_lexical_adoption(actor_id)
    prompt = prompt_template.format(
        instruccion = _BASE_INSTRUCCION,
        contexto    = contexto,
        respuesta   = respuesta,
    )
    time.sleep(SLEEP_ENTRE_CALLS)
    return _llamar_evaluador(prompt, evaluador)


def evaluar_answer_relevance(
    pregunta       : str,
    respuesta      : str,
    evaluador      : ChatOpenAI,
    rol            : str = "victima",
    tipo_utterance : str = "neutro",
) -> Optional[float]:
    prompt = PROMPT_ANSWER_RELEVANCE.format(
        instruccion    = _BASE_INSTRUCCION,
        pregunta       = pregunta,
        respuesta      = respuesta,
        rol            = rol,
        tipo_utterance = tipo_utterance,
    )
    time.sleep(SLEEP_ENTRE_CALLS)
    return _llamar_evaluador(prompt, evaluador)


def evaluar_estabilidad_rol(
    actor_id     : str,
    transcripcion: str,
    evaluador    : ChatOpenAI,
    instruccion  : Optional[str] = None,
) -> Optional[float]:
    prompt_template = _generar_prompt_estabilidad_rol(actor_id)
    prompt = prompt_template.format(
        instruccion   = instruccion or _BASE_INSTRUCCION_ESTRICTA,
        transcripcion = transcripcion,
    )
    time.sleep(SLEEP_ENTRE_CALLS)
    return _llamar_evaluador(prompt, evaluador)


def evaluar_autenticidad_lexica(
    actor_id : str,
    respuesta: str,
    evaluador: ChatOpenAI,
) -> Optional[float]:
    prompt_template = _generar_prompt_autenticidad_lexica(actor_id)
    prompt = prompt_template.format(
        instruccion = _BASE_INSTRUCCION,
        respuesta   = respuesta,
    )
    time.sleep(SLEEP_ENTRE_CALLS)
    return _llamar_evaluador(prompt, evaluador)


def evaluar_autenticidad_emocional(
    actor_id : str,
    respuesta: str,
    evaluador: ChatOpenAI,
) -> Optional[float]:
    prompt_template = _generar_prompt_autenticidad_emocional(actor_id)
    prompt = prompt_template.format(
        instruccion = _BASE_INSTRUCCION,
        respuesta   = respuesta,
    )
    time.sleep(SLEEP_ENTRE_CALLS)
    return _llamar_evaluador(prompt, evaluador)


def evaluar_autenticidad(
    actor_id : str,
    respuesta: str,
    evaluador: ChatOpenAI,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Evalúa autenticidad POR TURNO en dos dimensiones separadas:
      - autenticidad_lexica:    vocabulario oral situado en esta respuesta
      - autenticidad_emocional: tono afectivo apropiado al rol en esta respuesta
      - autenticidad_discursiva: promedio de ambas (backwards compatible)
    """
    score_lex = evaluar_autenticidad_lexica(actor_id, respuesta, evaluador)
    score_emo = evaluar_autenticidad_emocional(actor_id, respuesta, evaluador)
    score_disc = _promedio(score_lex, score_emo)
    return score_lex, score_emo, score_disc


def evaluar_coherencia_tactica(
    actor_id       : str,
    micro_deseo    : str,
    micro_intencion: str,
    respuesta      : str,
    evaluador      : ChatOpenAI,
) -> Optional[float]:
    if not micro_deseo.strip():
        return None
    prompt_template = _generar_prompt_coherencia_tactica(actor_id)
    prompt = prompt_template.format(
        instruccion     = _BASE_INSTRUCCION,
        micro_deseo     = micro_deseo,
        micro_intencion = micro_intencion,
        respuesta       = respuesta,
    )
    time.sleep(SLEEP_ENTRE_CALLS)
    return _llamar_evaluador(prompt, evaluador)


def evaluar_subordinacion_macro(
    macro_deseo  : str,
    transcripcion: str,
    evaluador    : ChatOpenAI,
    instruccion  : Optional[str] = None,
) -> Optional[float]:
    if not macro_deseo.strip():
        return None
    prompt = PROMPT_SUBORDINACION_MACRO.format(
        instruccion   = instruccion or _BASE_INSTRUCCION,
        macro_deseo   = macro_deseo,
        transcripcion = transcripcion,
    )
    time.sleep(SLEEP_ENTRE_CALLS)
    return _llamar_evaluador(prompt, evaluador)


def evaluar_goal_directedness(
    actor_id     : str,
    transcripcion: str,
    evaluador    : ChatOpenAI,
    instruccion  : Optional[str] = None,
) -> Optional[float]:
    """
    Proxy observable de subordinacion_macro — aplica a TODAS las condiciones.
    No requiere estado BDI: evalúa si la conversación acumula coherencia
    narrativa hacia un objetivo identificable.
    """
    prompt_template = _generar_prompt_goal_directedness(actor_id)
    prompt = prompt_template.format(
        instruccion   = instruccion or _BASE_INSTRUCCION,
        actor_id      = actor_id,
        transcripcion = transcripcion,
    )
    time.sleep(SLEEP_ENTRE_CALLS)
    return _llamar_evaluador(prompt, evaluador)


def evaluar_tactical_consistency(
    actor_id : str,
    pregunta : str,
    respuesta: str,
    evaluador: ChatOpenAI,
) -> Optional[float]:
    """
    Evalúa APROPIADEZ CONTEXTUAL POR TURNO: ¿la estrategia es correcta para ESTA pregunta?
    Dinámico por rol. Aplica a TODAS las condiciones.
    """
    prompt_template = _generar_prompt_tactical_consistency(actor_id)
    prompt = prompt_template.format(
        instruccion = _BASE_INSTRUCCION,
        actor_id    = actor_id,
        pregunta    = pregunta,
        respuesta   = respuesta,
    )
    time.sleep(SLEEP_ENTRE_CALLS)
    return _llamar_evaluador(prompt, evaluador)


def evaluar_factual_grounding(
    actor_id             : str,
    verificacion_contexto: str,
    respuesta            : str,
    evaluador            : ChatOpenAI,
) -> Optional[float]:
    """
    Evalúa si las afirmaciones factuales de la respuesta son verificables
    en el corpus CEV, independientemente de si el RAG estaba activo.

    A diferencia de faithfulness (¿usó los chunks que recibió?), esta métrica
    pregunta: ¿lo que dice el agente tiene respaldo en los testimonios del corpus?

    Aplica a TODAS las condiciones (baseline, rag_only, bdi_only, bdi_rag).
    Retorna None si no hay chunks de verificación disponibles.
    """
    if not verificacion_contexto.strip():
        return None
    prompt = _generar_prompt_factual_grounding(actor_id).format(
        instruccion           = _BASE_INSTRUCCION,
        verificacion_contexto = verificacion_contexto,
        respuesta             = respuesta,
    )
    time.sleep(SLEEP_ENTRE_CALLS)
    return _llamar_evaluador(prompt, evaluador)


def calcular_grounding_score(
    actor_contexto: str,
    respuesta     : str,
) -> Optional[float]:
    """
    Similitud semántica (coseno) entre la respuesta del agente y el contexto
    RAG recuperado en el turno.

    Complementa faithfulness (LLM-judge sobre uso del contexto):
      - faithfulness  → ¿usó los chunks que recibió? (escala 1-10, subjetiva)
      - grounding_score → ¿qué tan parecida es la respuesta a los chunks? (0-1, objetiva)

    Captura paráfrasis y sinónimos que el overlap léxico no detecta.
    Retorna None si no hay contexto o falla el modelo de embeddings.
    """
    if not actor_contexto.strip() or not respuesta.strip():
        return None
    model = _get_embedding_model()
    if model is None:
        return None
    try:
        embs = model.encode([respuesta, actor_contexto], convert_to_numpy=True)
        # Cosine similarity manual (evita dependencia adicional de sklearn)
        v_resp = embs[0]
        v_ctx  = embs[1]
        norm   = np.linalg.norm(v_resp) * np.linalg.norm(v_ctx)
        if norm == 0:
            return None
        sim = float(np.dot(v_resp, v_ctx) / norm)
        return round(sim, 4)
    except Exception as exc:
        print(f"    ⚠  Error en grounding_score: {exc}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DE TRANSCRIPCIÓN
# ════════════════════════════════════════════════════════════════════════════════

def _construir_transcripcion(trazas: list[dict]) -> str:
    """
    Convierte la lista de trazas ordenadas en una transcripción legible
    para el evaluador de dimensiones psicológicas.
    """
    lineas = []
    for t in sorted(trazas, key=lambda x: x["turno"]):
        lineas.append(f"[Turno {t['turno']} — {t['tipologia']}]")
        lineas.append(f"ENTREVISTADOR: {t['pregunta']}")
        lineas.append(f"AGENTE ({t['actor_id']}): {t['respuesta']}")
        lineas.append("")
    return "\n".join(lineas)


def _promedio(*valores: Optional[float]) -> Optional[float]:
    """Promedio de valores no-None; None si todos son None."""
    validos = [v for v in valores if v is not None]
    if not validos:
        return None
    return round(sum(validos) / len(validos), 4)


# ════════════════════════════════════════════════════════════════════════════════
# EVALUACIÓN DE UNA CONVERSACIÓN COMPLETA (condicion × actor)
# ════════════════════════════════════════════════════════════════════════════════

def evaluar_conversacion(
    condicion     : str,
    actor_id      : str,
    trazas        : list[dict],
    db,
    evaluador     : ChatOpenAI,
    col_resultados: str = COL_RESULTADOS_OPENAI,
) -> None:
    """
    Evalúa todas las dimensiones para una conversación y guarda los
    resultados turno a turno en evaluacion_resultados.

    Estrategia:
      - Dimensiones psicológicas (c, d, e): una sola llamada con la
        transcripción completa → score igual para todos los turnos.
      - Dimensiones RAG (a, b): por turno.
      - Dimensiones BDI (f, g): f por turno, g una sola llamada.
    """
    print(f"\n  → Evaluando {condicion}/{actor_id}  ({len(trazas)} turnos)")

    trazas_ord = sorted(trazas, key=lambda x: x["turno"])

    # Excluir turnos con respuesta inválida de la transcripción global
    trazas_validas = [t for t in trazas_ord if not t.get("respuesta_invalida", False)]
    n_invalidas = len(trazas_ord) - len(trazas_validas)
    if n_invalidas:
        print(f"     ⚠  {n_invalidas} turno(s) omitido(s) por respuesta inválida")

    transcripcion = _construir_transcripcion(trazas_validas)

    # Tomar behavior_params y macro_deseo del primer turno válido que los tenga
    behavior_params = next(
        (t["parametros_comportamiento"] for t in trazas_validas
         if t.get("parametros_comportamiento")), ""
    )
    macro_deseo = next(
        (t["macro_deseo"] for t in trazas_validas if t.get("macro_deseo")), ""
    )

    # ── Dimensiones psicológicas (una sola evaluación por conversación) ──
    print("     d) estabilidad_rol …", end=" ", flush=True)
    score_d = evaluar_estabilidad_rol(actor_id, transcripcion, evaluador)
    print(score_d)

    # ── Métricas de coherencia narrativa observable — TODAS las condiciones ──
    # goal_directedness requiere el arco completo de la conversación.
    print("     h) goal_directedness …", end=" ", flush=True)
    score_h = evaluar_goal_directedness(actor_id, transcripcion, evaluador)
    print(score_h)

    # autenticidad_lexica, autenticidad_emocional y tactical_consistency
    # se evalúan POR TURNO dentro del loop (ver abajo).

    # ── Subordinación al macro_deseo (una sola evaluación por conversación) ──
    # Aplica a bdi_only y bdi_rag (ambas condiciones tienen estado BDI activo)
    _condicion_con_bdi = condicion in ("bdi_only", "bdi_rag")
    score_g_global: Optional[float] = None
    if _condicion_con_bdi:
        print("     g) subordinacion_macro …", end=" ", flush=True)
        score_g_global = evaluar_subordinacion_macro(macro_deseo, transcripcion, evaluador)
        print(score_g_global)

    # ── Evaluaciones por turno ────────────────────────────────────────────────
    for traza in trazas_ord:
        num = traza["turno"]

        # Saltar turnos donde el agente generó un rechazo en lugar de responder en rol
        if traza.get("respuesta_invalida", False):
            print(f"     turno {num:02d} … OMITIDO (respuesta inválida)")
            continue

        print(f"     turno {num:02d} …", end=" ", flush=True)

        # a) Lexical adoption — solo cuando hay contexto RAG real (rag_only, bdi_rag)
        score_a = evaluar_lexical_adoption(
            traza.get("actor_id", actor_id),
            traza.get("actor_contexto", ""),
            traza["respuesta"],
            evaluador,
        )

        # b) Answer Relevance — rol y tipo_utterance ajustan la rúbrica al comportamiento esperado
        score_b = evaluar_answer_relevance(
            traza["pregunta"],
            traza["respuesta"],
            evaluador,
            rol            = traza.get("actor_id", actor_id),
            tipo_utterance = traza.get("tipo_utterance", "neutro"),
        )

        # e) Autenticidad léxica y emocional — por turno
        score_e_lex = evaluar_autenticidad_lexica(
            traza.get("actor_id", actor_id),
            traza["respuesta"],
            evaluador,
        )
        score_e_emo = evaluar_autenticidad_emocional(
            traza.get("actor_id", actor_id),
            traza["respuesta"],
            evaluador,
        )
        score_e = _promedio(score_e_lex, score_e_emo)

        # i) Tactical consistency — por turno (apropiadez contextual pregunta→respuesta)
        score_i = evaluar_tactical_consistency(
            traza.get("actor_id", actor_id),
            traza["pregunta"],
            traza["respuesta"],
            evaluador,
        )

        # f) Coherencia táctica — bdi_only y bdi_rag tienen micro_deseo activo
        score_f: Optional[float] = None
        if _condicion_con_bdi:
            score_f = evaluar_coherencia_tactica(
                traza.get("actor_id", actor_id),
                traza.get("micro_deseo", ""),
                traza.get("micro_intencion", ""),
                traza["respuesta"],
                evaluador,
            )

        # j) Factual grounding — aplica a TODAS las condiciones
        # Usa verificacion_contexto (recuperado con la respuesta como query)
        # en lugar de actor_contexto (recuperado con la pregunta como query)
        score_j = evaluar_factual_grounding(
            traza.get("actor_id", actor_id),
            traza.get("verificacion_contexto", ""),
            traza["respuesta"],
            evaluador,
        )

        # k) Grounding score — similitud semántica respuesta ↔ chunks RAG
        # Solo tiene valor cuando hay actor_contexto (rag_only, bdi_rag).
        # No usa LLM; usa el mismo modelo de embeddings del sistema RAG.
        _condicion_con_rag = condicion in ("rag_only", "bdi_rag")
        grounding_score = (
            calcular_grounding_score(
                traza.get("actor_contexto", ""),
                traza["respuesta"],
            )
            if _condicion_con_rag
            else None
        )

        # ── persona_score: 6 métricas universales (todas las condiciones) ──────
        # Comparable directamente entre baseline / rag_only / bdi_only / bdi_rag.
        # No incluye autenticidad_discursiva (derivada de autenticidad_lexica/emocional).
        persona_score = _promedio(
            score_d,      # estabilidad_rol
            score_e_lex,  # autenticidad_lexica
            score_e_emo,  # autenticidad_emocional
            score_h,      # goal_directedness
            score_i,      # tactical_consistency
            score_j,      # factual_grounding
        )

        # ── rag_score: calidad interna del RAG (rag_only, bdi_rag) ──────────
        # lexical_adoption — ¿adoptó el registro de los chunks? (LLM-judge 1-5, normalizado /5)
        # grounding_score  — ¿qué tan similar es la respuesta a los chunks? (0-1)
        # Para comparabilidad, escalar lexical_adoption a 0-1 al calcular rag_score.
        rag_score: Optional[float] = None
        if _condicion_con_rag:
            faith_norm = (score_a / 5.0) if score_a is not None else None
            rag_score  = _promedio(faith_norm, grounding_score)

        # ── bdi_score: calidad interna del BDI (bdi_only, bdi_rag) ──────────
        # Promedia coherencia_tactica y subordinacion_macro (ambas métricas BDI).
        bdi_score = (
            _promedio(score_f, score_g_global)  # coherencia_tactica + subordinacion_macro
            if _condicion_con_bdi
            else None
        )

        doc = {
            "condicion"         : condicion,
            "actor_id"          : actor_id,
            "turno"             : num,
            "tipologia_pregunta": traza["tipologia"],
            "pregunta"          : traza["pregunta"],
            "respuesta"         : traza["respuesta"],
            "scores": {
                # ── Universales: comparables entre las 4 condiciones ──────────
                # Forman persona_score
                "estabilidad_rol"        : score_d,
                "autenticidad_lexica"    : score_e_lex,
                "autenticidad_emocional" : score_e_emo,
                "goal_directedness"      : score_h,
                "tactical_consistency"   : score_i,
                "factual_grounding"      : score_j,
                # answer_relevance: universal pero no en persona_score
                "answer_relevance"       : score_b,
                # ── RAG (calidad interna): rag_only y bdi_rag ─────────────────
                "lexical_adoption"       : score_a,
                "grounding_score"        : grounding_score,
                # ── BDI (calidad interna): bdi_only y bdi_rag ─────────────────
                "coherencia_tactica"     : score_f,
                "subordinacion_macro"    : score_g_global,
                # ── Legacy / derivada — no usar en análisis principal ─────────
                "autenticidad_discursiva": score_e,
            },
            "persona_score": persona_score,
            "bdi_score"    : bdi_score,
            "rag_score"    : rag_score,
        }

        db[col_resultados].insert_one(doc)
        print(f"RAG={rag_score}  GS={grounding_score}  BDI={bdi_score}  Persona={persona_score}  FG={score_j}")

    print(f"     ✓  {condicion}/{actor_id} completado.")


# ════════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════════════════════════

def evaluar(condicion: str = "all") -> None:
    """
    Lee trazas de MongoDB, evalúa con LLM-as-a-judge y guarda resultados.

    Idempotente: si ya existen resultados para (condicion, actor_id),
    se omite esa conversación.

    Args:
        condicion: "baseline" | "rag_only" | "bdi_rag" | "all"
    """
    client    = MongoClient(MONGO_URI)
    db        = client[MONGO_DB]
    evaluador = _crear_evaluador()

    # Índice único para evitar duplicados
    db[COL_RESULTADOS_OPENAI].create_index(
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
        print(f"  EVALUANDO CONDICIÓN: {cond.upper()}")
        print(f"{'='*62}")

        for actor_id in ["victima", "victimario", "tercero"]:

            # ── Idempotencia ────────────────────────────────────────────
            ya_evaluados = db[COL_RESULTADOS_OPENAI].count_documents({
                "condicion": cond,
                "actor_id" : actor_id,
            })
            trazas_totales = db[COL_TRAZAS].count_documents({
                "condicion": cond,
                "actor_id" : actor_id,
            })

            if trazas_totales == 0:
                print(f"\n  ⚠  No hay trazas para {cond}/{actor_id}. "
                      f"Ejecuta generador_trazas.py primero.")
                continue

            if ya_evaluados >= trazas_totales:
                print(f"\n  ⏭  {cond}/{actor_id} ya evaluado "
                      f"({ya_evaluados} resultados). Saltando.")
                continue

            # ── Cargar trazas ───────────────────────────────────────────
            trazas = list(db[COL_TRAZAS].find(
                {"condicion": cond, "actor_id": actor_id},
                {"_id": 0},
            ))

            try:
                evaluar_conversacion(cond, actor_id, trazas, db, evaluador)
            except Exception as exc:
                print(f"\n  ✗  Error evaluando {cond}/{actor_id}: {exc}")

    client.close()
    print("\n✅  Evaluación completada.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluador LLM-as-a-judge — EntrevistasCEV"
    )
    parser.add_argument(
        "--condicion",
        choices=["baseline", "rag_only", "bdi_only", "bdi_rag", "all"],
        default="all",
        help="Condición experimental a evaluar (default: all)",
    )
    args = parser.parse_args()
    evaluar(args.condicion)
