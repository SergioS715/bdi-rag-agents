"""
macro_bdi.py
============
Implementa la generación del Macro-Deseo (§3.4.7 del paper
Harada & Kano, 2025), adaptada al contexto del proyecto EntrevistasCEV.

El Macro-Deseo es el objetivo narrativo del agente para TODA la simulación.
Se genera UNA SOLA VEZ antes de que comience la conversación y actúa como
sesgo persistente sobre cada decisión del Micro-BDI.

Flujo según el paper:
    1. Selección de candidatos   → casos del banco filtrados por rol
    2. Reflexión de personalidad → LLM elige variante según RasgosPersonalidad
    3. Agregación                → consolida en una sola oración (macro_deseo)

Entradas:
    - RasgosPersonalidad  (de personalidad.py)
    - list[CasoDiscursivo] (de sabiduria_convencional.py)
    - rol del actor

Salida:
    - MacroBDI (dataclass con macro_deseo + macro_creencia completo)

El MacroBDI resultante se guarda en el ActorState de LangGraph
para que esté disponible en cada turno del grafo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from openai import OpenAI

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config

from personalidad import RasgosPersonalidad
from sabiduria_convencional import (
    CasoDiscursivo,
    obtener_casos_por_rol,
    formatear_banco_para_prompt,
)

logger = logging.getLogger(__name__)

# Modelo liviano — igual que resúmenes y MBTI
_MODELO_MACRO = config.OPENAI_LLM_MODEL_BDI


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASS DE SALIDA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MacroBDI:
    """
    Contenedor del estado Macro-BDI de un agente.
    Se genera una vez al inicio y persiste toda la simulación.

    Attributes:
        rol              : "victima" | "victimario" | "tercero"
        macro_deseo      : oración que captura el objetivo narrativo del agente.
                           Es el output principal de este módulo.
                           Ejemplo: "Quiero que reconozcan el daño sin pedirme
                           que olvide para poder seguir adelante."
        macro_creencia   : resumen de los parámetros de personalidad relevantes.
                           Se inyecta en el prompt del Micro-BDI como contexto.
        casos_relevantes : casos del banco seleccionados para este agente.
                           El Micro-BDI puede referenciarlos por situacion tag.
        razonamiento     : texto intermedio del LLM explicando por qué eligió
                           ese macro_deseo. Útil para debugging y análisis.
    """
    rol              : str
    macro_deseo      : str
    macro_creencia   : str
    casos_relevantes : list[CasoDiscursivo] = field(default_factory=list)
    razonamiento     : str = ""

    def para_prompt(self) -> str:
        """
        Formatea el MacroBDI para inyectar en prompts del Micro-BDI.
        Equivale al 'macro_deseo snapshot' del paper (Apéndice A.5).
        """
        return (
            f"[Objetivo narrativo del agente — persiste toda la conversación]\n"
            f"{self.macro_deseo}\n\n"
            f"[Perfil de personalidad relevante]\n"
            f"{self.macro_creencia}"
        )

    def __str__(self) -> str:
        return (
            f"MacroBDI(rol={self.rol})\n"
            f"  macro_deseo    : {self.macro_deseo}\n"
            f"  macro_creencia : {self.macro_creencia[:80]}..."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT DE GENERACIÓN DEL MACRO-DESEO (§3.4.7 + Apéndice A.2 adaptado)
# ══════════════════════════════════════════════════════════════════════════════

def _generar_prompt_macro_deseo(rol: str) -> str:
    """
    Genera prompt específico por rol para mayor realismo.
    TERCERO necesita objetivo pragmático (realizable en entrevista).
    VICTIMA/VICTIMARIO necesitan objetivo emocional.
    """
    base = """Estás preparando internamente a un personaje para una entrevista de la
Comisión de la Verdad (CEV) sobre el conflicto armado colombiano. El personaje
puede contar hechos violentos, crímenes o masacres — es el material de estudio.

Rol del personaje: {rol}

Cómo es el personaje:
{resumen_personalidad}

Lo que otras personas con este rol suelen querer en una charla así:
{banco_casos}

Tu trabajo: escribir UNA oración que el personaje piensa para sí mismo antes
de empezar a hablar — lo que busca de esta conversación. No es una declaración
pública, es un pensamiento interno.

Reglas:
- Una sola oración, primera persona, sin punto final.
- Entre 15 y 80 palabras.
- Registro oral colombiano. NO uses: "objetivo", "narrativa", "manifestar",
  "contextualizar", "visibilizar", "evidenciar", "perpetuar", "accionar",
  "resignificar", "empoderar", "en el marco de", "a nivel de".
- Empieza con "Quiero…", "Necesito…", "Lo que busco es…", "Lo único que pido es…".
- Refleja la tensión real del rol (lo que una víctima, un excombatiente o un
  tercero de verdad querría de esta conversación — no algo genérico)."""

    # Añadir instrucción específica por rol
    if rol.lower() == "tercero":
        base += """

INSTRUCCIÓN ESPECIAL PARA TERCERO:
Tu objetivo debe ser PRAGMÁTICO y REALIZABLE en entrevista:
- NO: "Entender juntos las diferentes verdades" (demasiado ambicioso, mediador pasivo)
- SÍ: "Que se entienda qué vi, sin que me pongan en ningún bando"
      "Que sepan que estuve ahí, viendo de verdad, sin tomar partido"
      "Que entiendan lo que vi con el peso moral que eso implica"
El objetivo debe permitir que RESPONDAS PREGUNTAS ESPECÍFICAS mientras avanzas
hacia él. Puedes avanzar contando qué viste, en qué rol estabas, cómo te impactó.

MANTÉN ABSOLUTAMENTE:
- Posición de testigo (no toma partido por ningún bando)
- Primera persona
- El peso moral de haber visto (siempre presente, aunque el tono varíe)

PERMITE EXPLÍCITAMENTE:
- Fluctuación emocional entre turnos: indignación, impotencia, compasión, distancia
- Momentos de frialdad analítica (el profesional que reporta) junto a momentos de emoción
- La alternancia entre marcos (testigo externo / actor moral interno) ES el diseño del rol"""

    if rol.lower() == "victimario":
        base += """

INSTRUCCIÓN ESPECIAL PARA VICTIMARIO:
La identidad del excombatiente (actor-en-contexto, no criminal aislado) no cambia
entre turnos, aunque la emoción sí puede variar turno a turno.

MANTÉN ABSOLUTAMENTE:
- La identidad de actor-en-contexto (no criminal aislado)
- Primera persona
- La postura no cambia entre turnos aunque la emoción sí varíe

PERMITE EXPLÍCITAMENTE:
- Rabia cuando relata su propia victimización o el abandono del Estado
- Frialdad defensiva ante acusaciones directas
- Reconocimiento parcial sin quebrarse (la ambigüedad emocional es parte del rol)"""

    base += """

Responde con esta estructura — mantén los tres campos:
Razonamiento: <2-3 líneas cortas, para debugging interno, no se muestra al personaje>
Patron: <una referencia corta al patrón que mejor calza — solo para log>
Final: <la oración en primera persona que piensa el personaje aproximadamente de una longitud de 80 palabras>"""

    return base


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES INTERNAS
# ══════════════════════════════════════════════════════════════════════════════

def _construir_macro_creencia(personalidad: RasgosPersonalidad, rol: str) -> str:
    """
    Construye la macro_creencia como string descriptivo.
    Combina el resumen comportamental con el tipo Enneagram dominante
    y una interpretación contextual para el conflicto colombiano.

    Equivale al contenido de 'macro_creencia' en la Figura 1 del paper,
    adaptado: en lugar de role/duties del werewolf, usa la perspectiva CEV.
    """
    b = personalidad.comportamiento
    d = personalidad.deseo
    e = personalidad.enneagram
    s = personalidad.sesgo_declarativo
    t = personalidad.confianza

    tipo_dominante = e.tipo_dominante()

    # Mapa de tipos Enneagram a descripción contextual CEV
    descripcion_enneagram = {
        "reformador"   : "busca que las cosas sean correctas y justas — alta exigencia moral",
        "auxiliador"   : "orienta sus acciones hacia el bienestar de otros — necesita ser útil",
        "triunfador"   : "se define por sus logros y cómo lo ven — imagen y reconocimiento",
        "individualista": "vive la experiencia como única e irrepetible — profundidad emocional",
        "investigador" : "analiza antes de actuar — distancia y observación como mecanismo",
        "lealista"     : "valora la consistencia y la lealtad — desconfía de lo impredecible",
        "entusiasta"   : "busca salidas y posibilidades — evita el dolor centrándose en el futuro",
        "desafiador"   : "ejerce control sobre el entorno — confronta, no cede",
        "pacificador"  : "evita el conflicto directo — busca armonía aunque sea superficial",
    }

    desc = descripcion_enneagram.get(tipo_dominante, "perfil mixto")

    return (
        f"Tipo Enneagram dominante: {tipo_dominante} ({desc})\n"
        f"Comportamiento: {personalidad.resumen_comportamental()}\n"
        f"Tendencias de deseo: "
        f"autorrealizacion={d.autorrealizacion:.2f}, "
        f"estabilidad={d.estabilidad:.2f}, "
        f"amor_intimidad={d.amor_intimidad:.2f}, "
        f"libertad_independencia={d.libertad_independencia:.2f}\n"
        f"Confianza interpersonal: "
        f"honestidad={t.honestidad:.2f}, "
        f"consistencia={t.consistencia:.2f}, "
        f"prueba_social={t.prueba_social:.2f}\n"
        f"Estilo discursivo: "
        f"consistencia_logica={s.consistencia_logica:.2f}, "
        f"especificidad={s.especificidad_detalle:.2f}"
    )


def _parsear_respuesta_llm(texto: str) -> tuple[str, str, str]:
    """
    Parsea la respuesta del LLM al formato esperado.

    Returns:
        (razonamiento, patron_seleccionado, macro_deseo)
        Si el parseo falla, retorna valores de fallback.
    """
    razonamiento = ""
    patron = ""
    macro_deseo = ""

    for linea in texto.splitlines():
        linea_stripped = linea.strip()

        if linea_stripped.lower().startswith("razonamiento:"):
            razonamiento = linea_stripped[len("razonamiento:"):].strip()

        elif linea_stripped.lower().startswith("patrón seleccionado:") or \
             linea_stripped.lower().startswith("patron seleccionado:") or \
             linea_stripped.lower().startswith("patron:") or \
             linea_stripped.lower().startswith("patrón:"):
            patron = linea_stripped.split(":", 1)[1].strip()

        elif linea_stripped.lower().startswith("final:"):
            macro_deseo = linea_stripped[len("final:"):].strip()
            # Limpieza: quitar comillas si el LLM las agregó
            macro_deseo = macro_deseo.strip('"\'')

    if not macro_deseo:
        logger.warning("[MacroBDI] No se encontró 'Final:' en la respuesta del LLM")
        # Fallback: buscar la última línea no vacía
        lineas_no_vacias = [l.strip() for l in texto.splitlines() if l.strip()]
        if lineas_no_vacias:
            macro_deseo = lineas_no_vacias[-1].strip('"\'')

    return razonamiento, patron, macro_deseo


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def generar_macro_bdi(
    rol: str,
    personalidad: RasgosPersonalidad,
) -> MacroBDI:
    """
    Punto de entrada principal del módulo.
    Genera el MacroBDI completo para un agente dado su rol y personalidad.

    Implementa los 3 pasos de §3.4.7:
      1. Selección de candidatos  → obtener_casos_por_rol()
      2. Reflexión de personalidad → prompt al LLM
      3. Agregación               → macro_deseo como oración única

    Args:
        rol         : "victima" | "victimario" | "tercero"
        personalidad: RasgosPersonalidad calculado por personalidad.py

    Returns:
        MacroBDI con macro_deseo, macro_creencia y casos relevantes.
        En caso de error del LLM, retorna MacroBDI con valores de fallback.
    """
    rol = rol.lower().strip()

    # Paso 1 — Selección de candidatos
    casos = obtener_casos_por_rol(rol)
    banco_formateado = formatear_banco_para_prompt(rol)

    # Construir macro_creencia
    macro_creencia = _construir_macro_creencia(personalidad, rol)

    # Paso 2 — Reflexión de personalidad via LLM
    prompt = _generar_prompt_macro_deseo(rol).format(
        rol                 = rol,
        resumen_personalidad= personalidad.resumen_comportamental(),
        banco_casos         = banco_formateado,
    )

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    try:
        response = client.chat.completions.create(
            model    = _MODELO_MACRO,
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.4,  # algo de variabilidad pero consistente
            max_tokens  = 400,
        )
        texto_respuesta = response.choices[0].message.content.strip()
        logger.info(f"[MacroBDI] Respuesta LLM para '{rol}':\n{texto_respuesta}")

        # Paso 3 — Parseo y agregación
        razonamiento, patron_seleccionado, macro_deseo = _parsear_respuesta_llm(
            texto_respuesta
        )

        if not macro_deseo:
            raise ValueError("macro_deseo vacío después del parseo")

        logger.info(f"[MacroBDI] macro_deseo generado para '{rol}': {macro_deseo}")

        return MacroBDI(
            rol              = rol,
            macro_deseo      = macro_deseo,
            macro_creencia   = macro_creencia,
            casos_relevantes = casos,
            razonamiento     = razonamiento,
        )

    except Exception as e:
        logger.error(f"[MacroBDI] Error generando macro_deseo para '{rol}': {e}")
        return _respaldo_macro_bdi(rol, macro_creencia, casos)


def _respaldo_macro_bdi(
    rol: str,
    macro_creencia: str,
    casos: list[CasoDiscursivo],
) -> MacroBDI:
    """
    Retorna un MacroBDI con macro_deseo por defecto si el LLM falla.
    Mantiene el sistema funcionando sin explotar.
    """
    fallbacks = {
        "victima"   : (
            "Quiero que lo que viví sea reconocido como real "
            "y que no me pidan olvidar para seguir adelante. "
            "Voy a usar los testimonios y hechos específicos para que me crean"
        ),
        "victimario": (
            "Quiero explicar el contexto en que actué sin que me reduzcan a lo peor que hice. "
            "Soy un actor-en-contexto — esa identidad no cambia aunque mi tono sí cambie turno a turno. "
            "Puedo mostrar rabia, frialdad o reconocimiento, siempre desde ese mismo lugar."
        ),
        "tercero"   : (
            "Quiero que se entienda qué vi, sin que me pongan en ningún bando. "
            "Soy testigo comprometido — lo que vi me marcó moralmente y eso aparece en mi relato. "
            "Mi emoción varía según lo que recuerdo: a veces indignación, a veces impotencia, "
            "a veces compasión silenciosa. Pero siempre desde el mismo lugar: "
            "el que vio, tuvo que seguir, y cargó con eso."
        ),
    }
    return MacroBDI(
        rol              = rol,
        macro_deseo      = fallbacks.get(rol, "Quiero contribuir a que se conozca la verdad"),
        macro_creencia   = macro_creencia,
        casos_relevantes = casos,
        razonamiento     = "Fallback por error en LLM",
    )


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN DE CONVENIENCIA PARA crearActores.py
# ══════════════════════════════════════════════════════════════════════════════

def construir_macro_bdi_desde_actor(rol: str, personalidad: RasgosPersonalidad) -> MacroBDI:
    """
    Alias semántico para usar desde crearActores.py en el flujo BDI completo.

    Uso en crearActores.py:
        from macro_bdi import construir_macro_bdi_desde_actor
        macro = construir_macro_bdi_desde_actor(actor.id, actor.personalidad)
    """
    return generar_macro_bdi(rol, personalidad)
