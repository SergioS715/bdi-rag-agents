"""
actor.py
========
Dataclass del actor CEV, extendida con el campo `personalidad`
para soportar el sistema BDI del paper Harada & Kano (2025).

Cambio respecto a la versión anterior:
  - Se agrega `personalidad: Optional[RasgosPersonalidad] = None`
  - El campo es OPCIONAL: el chatbot actual sigue funcionando sin cambios.
  - Solo se llena cuando se activa el modo simulación BDI.
  - Se mantiene Pydantic BaseModel para compatibilidad con LangGraph.
"""

from typing import Optional
from pydantic import BaseModel, Field

from personalidad import RasgosPersonalidad


class Actor(BaseModel):
    """
    Representa un actor del conflicto armado colombiano.

    Campos originales (sin cambios):
        id          : identificador del rol ("victima", "victimario", "tercero")
        nombre      : nombre genérico para presentarse en la conversación
        perspectiva : cómo interpreta y vivió el conflicto
        estilo      : cómo se comunica (registro oral, tono, vocabulario)

    Campo nuevo (BDI):
        personalidad : parámetros MBTI + Enneagram + 24 rasgos derivados.
                       None si aún no se ha calculado o si no se usa BDI.
    """
    id          : str = Field(description="Identificador único del rol")
    nombre      : str = Field(description="Nombre genérico del actor")
    perspectiva : str = Field(description="Cómo interpreta el conflicto")
    estilo      : str = Field(description="Cómo se comunica")

    # Campo nuevo — opcional para no romper código existente
    personalidad: Optional[RasgosPersonalidad] = Field(
        default=None,
        description=(
            "Parámetros de personalidad BDI: MBTI, Enneagram y 24 rasgos derivados. "
            "Se llena al iniciar una simulación multi-agente. "
            "None durante el modo chatbot estándar."
        )
    )

    # Pydantic v2: permite tipos arbitrarios (dataclasses anidadas)
    model_config = {"arbitrary_types_allowed": True}

    def tiene_personalidad(self) -> bool:
        """Retorna True si el actor tiene personalidad BDI calculada."""
        return self.personalidad is not None

    def resumen_para_prompt(self) -> str:
        """
        Genera el bloque de personalidad para inyectar en prompts.
        Si no hay personalidad calculada, retorna string vacío.

        Uso en prompts.py:
            {actor_personalidad} → actor.resumen_para_prompt()
        """
        if self.personalidad is None:
            return ""
        return (
            f"[Parámetros de personalidad]\n"
            f"{self.personalidad.resumen_comportamental()}"
        )

    def __str__(self) -> str:
        tiene = "con personalidad BDI" if self.tiene_personalidad() else "sin personalidad BDI"
        return f"Actor(id={self.id}, nombre={self.nombre}, {tiene})"