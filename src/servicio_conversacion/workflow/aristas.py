from typing_extensions import Literal

from langgraph.graph import END

from estado import EstadoActor
from config import TOTAL_MESSAGES_SUMMARY_TRIGGER


def debe_resumir_conversacion(
    estado: EstadoActor,
) -> Literal["nodo_resumir_conversacion", "__end__"]:
    messages = estado["messages"]

    if len(messages) > TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "nodo_resumir_conversacion"

    return END
