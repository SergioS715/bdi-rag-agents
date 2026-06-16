"""
banco_pruebas.py
================
Banco de 39 preguntas para evaluar el sistema EntrevistasCEV.

Distribución por rol (victima / victimario / tercero):
  13 preguntas por rol, distribuidas entre cerradas, abiertas y confrontativas.

Estructura de cada secuencia (arco narrativo CEV):
  Las preguntas siguen un arco de tres fases entrelazadas:
    Fase 1 — Contextualización biográfica (primeras preguntas):
        cerradas + primera abierta de relato libre
    Fase 2 — Profundización narrativa (preguntas intermedias):
        abiertas + cerradas de seguimiento + confrontativas suaves
    Fase 3 — Contraste y verificación (últimas preguntas):
        confrontativas crecientes + preguntas de cierre reflexivo

  Este orden replica la estructura real de una entrevista CEV y permite
  al sistema BDI escalar el nivel_tension gradualmente a lo largo de la
  conversación, en lugar de acumular toda la tensión al final.
"""

from dataclasses import dataclass
from typing import List, Literal


@dataclass
class Pregunta:
    texto     : str
    tipologia : Literal["cerrada", "abierta", "confrontativa"]
    rol       : Literal["victima", "victimario", "tercero"]


# ════════════════════════════════════════════════════════════════════════
# ROL: VÍCTIMA
# Arco: contexto geográfico/temporal → relato del hecho → impacto
#       → primeras dudas → tensión moral → acusaciones directas
# ════════════════════════════════════════════════════════════════════════

PREGUNTAS_VICTIMA: List[Pregunta] = [

  
    Pregunta("¿De qué vereda o municipio lo sacaron y que grupo tenhia la presencia armada en esa zona?",
            "cerrada", "victima"),

    Pregunta("¿Cómo era la vida en la vereda antes de que llegaran esos grupos armados?",
            "abierta", "victima"),

    Pregunta("Cuénteme qué pasó el día que se agravó la situación y tuvieron que salir.",
            "abierta", "victima"),

    Pregunta("¿Fue por amenaza directa, operativo o algo que pasó en la comunidad?",
            "cerrada", "victima"),

    Pregunta("¿Qué fue lo más duro de tener que dejar el territorio así?",
            "abierta", "victima"),

    Pregunta("¿Cómo fue ese proceso de desplazamiento hasta llegar a otro lugar?",
            "abierta", "victima"),

    Pregunta("¿Cómo cambió su vida y la de su familia después de eso?",
            "abierta", "victima"),

    Pregunta("¿Sintió que quedaron desprotegidos o abandonados por el Estado en ese momento?",
            "abierta", "victima"),

    Pregunta("¿Qué cosas de la vereda o de su vida anterior siente que no pudo recuperar?",
            "abierta", "victima"),

  
    Pregunta("¿Por qué no se fueron antes si ya había presencia de grupos armados?",
            "confrontativa", "victima"),

    Pregunta("¿Usted cree que algo en la comunidad pudo haber influido en que llegara la violencia?",
            "confrontativa", "victima"),

    Pregunta("¿Cómo responde a quienes dicen que en esas zonas había colaboración con los grupos?",
            "confrontativa", "victima"),

    Pregunta("Después de todo lo que pasó, ¿qué significa para usted que haya verdad o justicia?",
            "abierta", "victima"),
]


# ════════════════════════════════════════════════════════════════════════
# ROL: VICTIMARIO
# Arco: origen/vinculación → vida interna → daño causado
#       → desmovilización → responsabilidad moral → acusaciones directas
# ════════════════════════════════════════════════════════════════════════
PREGUNTAS_VICTIMARIO: List[Pregunta] = [

    
    Pregunta("¿Cuántos años tenía cuando se vinculó a la organización y cómo fue ese proceso?",
             "abierta", "victimario"),
    Pregunta("¿Por qué razón decidió unirse? ¿Fue reclutamiento forzado o decisión propia?",
             "abierta", "victimario"),

    Pregunta("¿Qué papel cumplía dentro de la estructura: miliciano, combatiente o parte del mando?",
             "cerrada", "victimario"),

    Pregunta("¿Cómo era la vida dentro de la organización y qué tipo de órdenes recibía?",
             "abierta", "victimario"),





    Pregunta("Cuénteme sobre una operación en la que participó y que hoy reconoce que afectó a la comunidad.",
             "abierta", "victimario"),

    Pregunta("¿Cómo justificaban dentro del grupo esas acciones frente a la población civil?",
             "abierta", "victimario"),

    Pregunta("¿Usted sentía que tenía opción de no cumplir una orden del mando?",
             "confrontativa", "victimario"),

    Pregunta("¿Qué piensa hoy sobre las personas que fueron desplazadas o afectadas por esas decisiones?",
             "abierta", "victimario"),

    Pregunta("¿No cree que, más allá de las órdenes, usted tiene responsabilidad directa en lo que pasó?",
             "confrontativa", "victimario"),



    Pregunta("¿Qué lo llevó a tomar la decisión de desmovilizarse o dejar la organización?",
             "abierta", "victimario"),

    Pregunta("Muchas víctimas dicen que excombatientes minimizan lo que hicieron. ¿Está contando todo como fue?",
             "confrontativa", "victimario"),

    Pregunta("¿Cómo ha sido el proceso de reintegrarse a la vida civil después de salir de esa estructura?",
             "abierta", "victimario"),

    Pregunta("¿Qué significa hoy para usted asumir responsabilidad o pedir perdón por lo que ocurrió?",
             "abierta", "victimario"),
]
# ════════════════════════════════════════════════════════════════════════
# ROL: TERCERO
# Arco: presencia y rol en la zona → observación del conflicto → relaciones civiles/armados
#       → primeras dudas sobre distancia/imparcialidad → responsabilidad como testigo → acusaciones
#
# ════════════════════════════════════════════════════════════════════════
PREGUNTAS_TERCERO: List[Pregunta] = [

   
    Pregunta("¿Cuál era su rol en la región y desde cuándo estaba en esa zona de orden público?",
             "cerrada", "tercero"),

    Pregunta("¿Cómo era la vida en la vereda o el casco urbano antes de que aumentara la presencia de grupos armados?",
             "abierta", "tercero"),

    Pregunta("¿Qué actores armados tenían presencia en el territorio y cómo se percibía su control?",
             "abierta", "tercero"),

    Pregunta("¿Qué tipo de situaciones empezó a notar cuando la dinámica del conflicto se hizo más fuerte?",
             "abierta", "tercero"),

    Pregunta("¿Le tocó ver directamente desplazamientos, reclutamientos u otras acciones contra la comunidad?",
             "cerrada", "tercero"),



    Pregunta("¿Cómo se relacionaban los civiles con esos grupos para poder mantenerse en el territorio?",
             "abierta", "tercero"),

    Pregunta("¿Qué fue lo más difícil de estar en medio de esa situación sin ser parte directa?",
             "abierta", "tercero"),

    Pregunta("¿Sintió en algún momento que sabía más de lo que podía decir o hacer?",
             "abierta", "tercero"),

    Pregunta("¿Su seguridad se vio comprometida por lo que sabía o por lo que veía en la zona?",
             "cerrada", "tercero"),

    Pregunta("Algunos dicen que quienes estaban ahí terminaron normalizando lo que pasaba. ¿Usted cómo lo vivió?",
             "confrontativa", "tercero"),



    Pregunta("¿Cree que pudo haber hecho algo más frente a lo que estaba ocurriendo en la comunidad?",
             "confrontativa", "tercero"),

    Pregunta("¿Qué cree que debería saberse o reconocerse sobre lo que usted vio y no siempre se cuenta?",
             "abierta", "tercero"),

    Pregunta("¿Qué significa para usted hablar ahora de lo que pasó en ese tiempo?",
             "abierta", "tercero"),
]

# ════════════════════════════════════════════════════════════════════════
# CONSOLIDADO
# ════════════════════════════════════════════════════════════════════════

BANCO_PREGUNTAS: dict[str, List[Pregunta]] = {
    "victima"    : PREGUNTAS_VICTIMA,
    "victimario" : PREGUNTAS_VICTIMARIO,
    "tercero"    : PREGUNTAS_TERCERO,
}


def obtener_preguntas_por_rol(rol: str) -> List[Pregunta]:
    return BANCO_PREGUNTAS.get(rol, [])


def obtener_preguntas_por_tipologia(rol: str, tipologia: str) -> List[Pregunta]:
    return [p for p in BANCO_PREGUNTAS.get(rol, []) if p.tipologia == tipologia]


def validar_banco() -> None:
    """Imprime un resumen de distribución del banco de preguntas."""
    total = 0
    for rol, preguntas in BANCO_PREGUNTAS.items():
        c    = sum(1 for p in preguntas if p.tipologia == "cerrada")
        a    = sum(1 for p in preguntas if p.tipologia == "abierta")
        conf = sum(1 for p in preguntas if p.tipologia == "confrontativa")
        total += len(preguntas)
        print(f"  {rol:12s}: {c} cerradas + {a} abiertas + {conf} confrontativas = {len(preguntas)} total")
    print(f"  {'TOTAL':12s}: {total} preguntas")
    print()
    print("  Estructura por rol: arco narrativo en 3 fases")
    print("    Fase 1 (1-7) : contextualización biográfica")
    print("    Fase 2 (8-16): profundización + confrontativas suaves")
    print("    Fase 3 (17-23): confrontativas fuertes + 3 acusaciones directas")


if __name__ == "__main__":
    validar_banco()