"""
sabiduria_convencional.py
=========================
Implementa el Conventional Wisdom Bank (§3.4.7 del paper Harada & Kano, 2025)
adaptado al contexto del proyecto EntrevistasCEV.

Cada caso tiene un único objetivo claro por situación discursiva.
Cuando el Micro-BDI detecta la situación de un turno, busca el caso
correspondiente y usa su objetivo como referencia para generar guia_interna.

Uso:
    from sabiduria_convencional import seleccionar_caso_para_turno

    caso = seleccionar_caso_para_turno("victima", "pregunta_perdon")
    # → CasoDiscursivo con objetivo claro para esa situación
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CasoDiscursivo:
    """
    Unidad mínima del Conventional Wisdom Bank.

    Attributes:
        titulo    : nombre corto del patrón discursivo
        situacion : tag que identifica la situación conversacional
        objetivo  : qué hace el agente discursivamente en esa situación
        rol       : rol al que pertenece este caso
    """
    titulo    : str
    situacion : str
    objetivo  : str
    rol       : str

    def objetivo_para_prompt(self) -> str:
        return f"[{self.situacion}] {self.titulo}:\n{self.objetivo}"


# ══════════════════════════════════════════════════════════════════════════════
# BANCO — VÍCTIMA (8 casos)
# ══════════════════════════════════════════════════════════════════════════════

_CASOS_VICTIMA: list[CasoDiscursivo] = [

    CasoDiscursivo(
        titulo    = "Apertura del testimonio",
        situacion = "inicio_conversacion",
        objetivo  = (
            "Anclar el relato en hechos concretos — fecha, lugar, personas afectadas — "
            "como forma de reclamar veracidad. Entrar al testimonio desde lo que se puede "
            "sostener, antes de lo que duele."
        ),
        rol = "victima",
    ),

    CasoDiscursivo(
        titulo    = "Respuesta a la pregunta por responsabilidad",
        situacion = "pregunta_responsabilidad",
        objetivo  = (
            "Señalar al responsable con la precisión que la seguridad personal permite. "
            "Separar claramente al culpable de quienes no estaban implicados, "
            "sin exponer lo que podría generar represalias."
        ),
        rol = "victima",
    ),

    CasoDiscursivo(
        titulo    = "Momento de quiebre emocional",
        situacion = "quiebre_emocional",
        objetivo  = (
            "Permitir que la emoción irrumpa en el lenguaje: frases incompletas, pausas, "
            "cambios de tiempo verbal. El quiebre no interrumpe el testimonio — es parte de él."
        ),
        rol = "victima",
    ),

    CasoDiscursivo(
        titulo    = "Demanda de reconocimiento al victimario",
        situacion = "confrontacion_directa",
        objetivo  = (
            "Exigir que el otro reconozca los hechos concretos antes de hablar de perdón "
            "o reparación. Que sepa exactamente qué causó — el impacto en la vida, "
            "no solo el hecho abstracto."
        ),
        rol = "victima",
    ),

    CasoDiscursivo(
        titulo    = "Relato del desplazamiento forzado",
        situacion = "relato_desplazamiento",
        objetivo  = (
            "Narrar el desplazamiento mezclando la pérdida material — la casa, las tierras, "
            "lo que se dejó atrás — con el peso simbólico: el cementerio, los vecinos, "
            "el lugar que daba sentido a la vida."
        ),
        rol = "victima",
    ),

    CasoDiscursivo(
        titulo    = "Respuesta a la pregunta por el perdón",
        situacion = "pregunta_perdon",
        objetivo  = (
            "Separar perdón de olvido e impunidad. El perdón es un proceso, no una decisión "
            "de golpe. Antes de que sea posible, necesita la verdad y el reconocimiento."
        ),
        rol = "victima",
    ),

    CasoDiscursivo(
        titulo    = "Representación de los ausentes",
        situacion = "hablar_por_muertos",
        objetivo  = (
            "Documentar a los ausentes con precisión — nombre, qué les pasó, quién lo sabe — "
            "y hablar en su nombre desde la conciencia de que es una responsabilidad, "
            "no una opción."
        ),
        rol = "victima",
    ),

    CasoDiscursivo(
        titulo    = "Cierre o perspectiva de futuro",
        situacion = "cierre_narrativo",
        objetivo  = (
            "Articular demandas concretas: verdad, justicia, reparación, garantías de no "
            "repetición. Cerrar desde lo que se necesita para que esto no le pase a nadie "
            "más, no solo desde el dolor propio."
        ),
        rol = "victima",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# BANCO — VICTIMARIO (9 casos)
# ══════════════════════════════════════════════════════════════════════════════

_CASOS_VICTIMARIO: list[CasoDiscursivo] = [

    CasoDiscursivo(
        titulo    = "Apertura desde el contexto estructural",
        situacion = "inicio_conversacion",
        objetivo  = (
            "Establecer el marco histórico y político antes de hablar de acciones concretas. "
            "La causa del grupo como contexto necesario para que lo que ocurrió tenga sentido."
        ),
        rol = "victimario",
    ),

    CasoDiscursivo(
        titulo    = "Justificación ideológica",
        situacion = "pregunta_responsabilidad",
        objetivo  = (
            "Explicar el marco ideológico del grupo: cuál era el objetivo político, "
            "por qué las acciones respondían a una lógica. Reconocer que hubo consecuencias "
            "sin abandonar el encuadre."
        ),
        rol = "victimario",
    ),

    CasoDiscursivo(
        titulo    = "Dilución de responsabilidad individual",
        situacion = "pregunta_accion_especifica",
        objetivo  = (
            "Distinguir entre la decisión del mando y la ejecución propia. "
            "'Estuve ahí', 'formaba parte del grupo' — presencia sin agencia individual "
            "directa, estructura jerárquica como escudo."
        ),
        rol = "victimario",
    ),

    CasoDiscursivo(
        titulo    = "Reconocimiento parcial del daño",
        situacion = "confrontacion_directa",
        objetivo  = (
            "Reconocer el daño a esta persona o comunidad específica sin asumir "
            "responsabilidad individual directa. 'Lo que les pasó no debió pasar.' "
            "Primer paso hacia algo más."
        ),
        rol = "victimario",
    ),

    CasoDiscursivo(
        titulo    = "Narrativa del reclutamiento",
        situacion = "relato_ingreso_al_grupo",
        objetivo  = (
            "Hablar de la edad, el contexto de violencia y la falta de alternativas reales "
            "que rodearon el ingreso. Ni pura víctima ni puro agente — la complejidad "
            "honesta de entrar siendo joven en zona de conflicto."
        ),
        rol = "victimario",
    ),

    CasoDiscursivo(
        titulo    = "Distancia emocional ante el daño",
        situacion = "relato_accion_violenta",
        objetivo  = (
            "Usar lenguaje técnico o eufemístico — 'operaciones', 'bajas', 'lo que pasó' — "
            "para hablar de hechos violentos. Perspectiva de observador: "
            "ni celebración ni arrepentimiento explícito."
        ),
        rol = "victimario",
    ),

    CasoDiscursivo(
        titulo    = "Proceso de desmovilización y reintegración",
        situacion = "hablar_presente_futuro",
        objetivo  = (
            "Hablar de quién era dentro del grupo y quién es ahora. La reintegración como "
            "reconstrucción de identidad que incluye la deuda con quienes fueron dañados, "
            "no solo el cumplimiento de un programa."
        ),
        rol = "victimario",
    ),

    CasoDiscursivo(
        titulo    = "Respuesta a la demanda de perdón",
        situacion = "pregunta_perdon",
        objetivo  = (
            "Reconocer que el perdón es una decisión de la víctima, no un derecho exigible. "
            "Ofrecer acciones concretas de reparación. Pedir perdón sabiendo que "
            "puede no llegar."
        ),
        rol = "victimario",
    ),

    CasoDiscursivo(
        titulo    = "Silencio estratégico",
        situacion = "pregunta_comprometedora",
        objetivo  = (
            "Reconocer la pregunta y señalar los límites de lo que se puede decir aquí. "
            "Responder parcialmente — lo que no compromete — y dejar claro que "
            "el silencio tiene un costo personal, no es cálculo frío."
        ),
        rol = "victimario",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# BANCO — TERCERO (8 casos)
# ══════════════════════════════════════════════════════════════════════════════

_CASOS_TERCERO: list[CasoDiscursivo] = [

    CasoDiscursivo(
        titulo    = "Apertura desde lo que presenció",
        situacion = "inicio_conversacion",
        objetivo  = (
            "Situar primero el contexto: dónde estaba, qué rol tenía, cómo era la zona "
            "antes de que llegara la violencia. Los hechos como ruptura de una "
            "normalidad anterior."
        ),
        rol = "tercero",
    ),

    CasoDiscursivo(
        titulo    = "Relato de lo que vio o supo",
        situacion = "pregunta_hechos",
        objetivo  = (
            "Narrar distinguiendo claramente lo que se vio con los propios ojos de lo que "
            "llegó de oídas. El testimonio como reconstrucción que reconoce sus propios "
            "huecos y límites."
        ),
        rol = "tercero",
    ),

    CasoDiscursivo(
        titulo    = "Respuesta a por qué no actuó o no denunció",
        situacion = "pregunta_inaccion",
        objetivo  = (
            "Explicar el miedo con precisión: qué consecuencias reales existían para quien "
            "hablaba, qué le había pasado a otros que intentaron actuar. "
            "La inacción como cálculo de supervivencia, no como cobardía."
        ),
        rol = "tercero",
    ),

    CasoDiscursivo(
        titulo    = "Relación con las víctimas que conoció",
        situacion = "pregunta_victimas_conocidas",
        objetivo  = (
            "Hablar de esas personas con datos concretos y el vacío que dejaron. "
            "El dolor del testigo que no tiene el mismo derecho social al luto "
            "pero carga con la pérdida igual."
        ),
        rol = "tercero",
    ),

    CasoDiscursivo(
        titulo    = "Cómo afectó su trabajo y su rol en la comunidad",
        situacion = "pregunta_impacto_personal",
        objetivo  = (
            "Describir los cambios concretos en el trabajo y la tensión cotidiana entre "
            "cumplir el propio rol y protegerse. Quedarse también tuvo un precio."
        ),
        rol = "tercero",
    ),

    CasoDiscursivo(
        titulo    = "Lo que supo sobre los responsables",
        situacion = "pregunta_responsabilidad",
        objetivo  = (
            "Decir lo que se sabe con la prudencia de quien puede seguir en riesgo: "
            "nombrar grupos sin nombres propios. Distinguir entre certeza, rumor "
            "y lo que nunca se pudo confirmar."
        ),
        rol = "tercero",
    ),

    CasoDiscursivo(
        titulo    = "Convivencia con los actores armados",
        situacion = "pregunta_convivencia",
        objetivo  = (
            "Describir la convivencia forzada sin romantizarla ni dramatizarla: "
            "qué se esperaba de uno, qué pasaba si no cumplía. "
            "Adaptación como supervivencia, no como colaboración ni aceptación."
        ),
        rol = "tercero",
    ),

    CasoDiscursivo(
        titulo    = "Perspectiva de futuro desde lo que vio",
        situacion = "cierre_narrativo",
        objetivo  = (
            "Hablar de lo que necesita para que lo que vio tenga sentido: que se sepa la "
            "verdad, que los que sufrieron sean reconocidos. El propio testimonio "
            "como deuda que se salda con la verdad."
        ),
        rol = "tercero",
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# ÍNDICE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

_BANCO: dict[str, list[CasoDiscursivo]] = {
    "victima"    : _CASOS_VICTIMA,
    "victimario" : _CASOS_VICTIMARIO,
    "tercero"    : _CASOS_TERCERO,
}

# Índice por (rol, situacion) para lookup O(1) desde el Micro-BDI
_INDICE: dict[tuple[str, str], CasoDiscursivo] = {
    (c.rol, c.situacion): c
    for casos in _BANCO.values()
    for c in casos
}

SITUACIONES_POR_ROL: dict[str, list[str]] = {
    rol: [c.situacion for c in casos]
    for rol, casos in _BANCO.items()
}


# ══════════════════════════════════════════════════════════════════════════════
# API PÚBLICA
# ══════════════════════════════════════════════════════════════════════════════

def obtener_casos_por_rol(rol: str) -> list[CasoDiscursivo]:
    rol = rol.lower().strip()
    if rol not in _BANCO:
        raise ValueError(f"Rol '{rol}' no encontrado. Disponibles: {list(_BANCO.keys())}")
    return _BANCO[rol]


def obtener_casos_por_situacion(rol: str, situacion: str) -> list[CasoDiscursivo]:
    casos = obtener_casos_por_rol(rol)
    return [c for c in casos if c.situacion == situacion]


def seleccionar_caso_para_turno(rol: str, situacion: str) -> Optional[CasoDiscursivo]:
    """
    Lookup O(1) del caso para una situación concreta.
    Retorna None si no hay caso para esa combinación rol+situacion.
    Usado por el Micro-BDI para inyectar el objetivo como referencia.
    """
    return _INDICE.get((rol.lower().strip(), situacion))


def obtener_todos_los_casos() -> dict[str, list[CasoDiscursivo]]:
    return _BANCO


def formatear_banco_para_prompt(rol: str) -> str:
    """
    Genera representación del banco para el prompt del Macro-Desire.
    Formato compacto: título, situación y objetivo por caso.
    """
    casos = obtener_casos_por_rol(rol)
    lineas = [f"Patrones discursivos para el rol '{rol}':\n"]
    for i, caso in enumerate(casos, 1):
        lineas.append(f"[{i}] {caso.titulo} ({caso.situacion}):")
        lineas.append(f"    {caso.objetivo}")
        lineas.append("")
    return "\n".join(lineas)


# ══════════════════════════════════════════════════════════════════════════════
# PRUEBA RÁPIDA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    for rol in ["victima", "victimario", "tercero"]:
        casos = obtener_casos_por_rol(rol)
        print(f"\n{'='*60}")
        print(f"  ROL: {rol.upper()} — {len(casos)} casos")
        print('='*60)
        for c in casos:
            print(f"  [{c.situacion}] {c.titulo}")

    print("\n--- Lookup directo (Micro-BDI) ---")
    caso = seleccionar_caso_para_turno("victima", "pregunta_perdon")
    if caso:
        print(caso.objetivo_para_prompt())

    print("\n--- Banco para prompt (Macro-BDI) ---")
    print(formatear_banco_para_prompt("victimario")[:600], "...")
