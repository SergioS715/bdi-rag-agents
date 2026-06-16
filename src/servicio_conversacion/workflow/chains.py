"""
chains.py
=========
Cadenas LangChain — EntrevistasCEV.

Cambios respecto a la versión anterior:
  - obtener_cadena_actor() ahora pasa micro_intencion, micro_deseo,
    macro_deseo, parametros_comportamiento y numero_turno al prompt.
  - Guardrail VICTIMARIO eliminado de aquí — ahora vive en el template
    Jinja2 de prompts.py, antes del bloque RAG.
  - Función de escape de curly braces más robusta (cubre tildes y espacios).
"""

import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, OPENAI_LLM_MODEL, OPENAI_LLM_MODEL_CONTEXT_SUMMARY
from jinja2 import Template
from tools import herramientas
from prompts import (
    PROMPT_RESUMEN_CONTEXTO,
    PROMPT_EXTENDER_RESUMEN,
    TARJETA_PERSONAJE_ACTOR,
    PROMPT_RESUMEN,
)


def _escapar_curly_braces_rag(texto: str) -> str:
    """
    Escapa cualquier {variable} en el texto RAG para evitar que LangChain
    los interprete como template variables.

    Cubre: {PALABRA}, {PALABRA_CON_GUION}, { PALABRA }, {Palabra_Con_Mayúsculas}
    No toca: {{ ya_escapado }} (ya tiene doble llave).
    """
    # Proteger los ya-escapados para no procesarlos dos veces
    texto = texto.replace('{{', '\x00OPEN\x00').replace('}}', '\x00CLOSE\x00')
    # Escapar cualquier {contenido} simple
    texto = re.sub(r'\{([^{}]+)\}', r'{{\1}}', texto)
    # Restaurar los ya-escapados
    texto = texto.replace('\x00OPEN\x00', '{{').replace('\x00CLOSE\x00', '}}')
    return texto


def obtener_modelo_chat(
    temperature: float = 0.7,
    model_name : str   = OPENAI_LLM_MODEL,
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key    = OPENAI_API_KEY,
        model_name = model_name,
        temperature= temperature,
    )


def obtener_cadena_actor(
    actor_nombre              : str = "",
    actor_perspectiva         : str = "",
    actor_estilo              : str = "",
    actor_contexto            : str = "",
    id_actor                  : str = "",
    macro_deseo               : str = "",
    parametros_comportamiento : str = "",
    micro_intencion           : str = "",
    resumen_conversacion      : str = "",
    numero_turno              : int = 1,
    usar_herramientas         : bool = True,
    tool_choice               : str = "auto",
):
    model = obtener_modelo_chat()
    if usar_herramientas:
        # "auto" = el modelo decide, "required" = fuerza la llamada (evaluación)
        model = model.bind_tools(herramientas, tool_choice=tool_choice)
    else:
        model = model.bind_tools(herramientas, tool_choice="none")

    # Escapar curly braces en actor_contexto para evitar que LangChain los interprete
    # como variables de template cuando se pase system_rendered a ChatPromptTemplate
    actor_contexto_escaped = _escapar_curly_braces_rag(actor_contexto) if actor_contexto else ""

    # Renderizar el prompt con jinja2 ANTES de pasarlo a LangChain
    system_rendered = Template(TARJETA_PERSONAJE_ACTOR).render(
        actor_nombre              = actor_nombre,
        actor_perspectiva         = actor_perspectiva,
        actor_estilo              = actor_estilo,
        actor_contexto            = actor_contexto_escaped,
        id_actor                  = id_actor,
        macro_deseo               = macro_deseo,
        parametros_comportamiento = parametros_comportamiento,
        micro_intencion           = micro_intencion,
        resumen_conversacion      = resumen_conversacion,
        numero_turno              = numero_turno,
    )

    # Sin template_format — el string ya está renderizado
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_rendered),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt | model

def obtener_cadena_resumen_conversacion(
    resumen_conversacion : str = "",
    actor_nombre         : str = "",
):
    model = obtener_modelo_chat(model_name=OPENAI_LLM_MODEL_CONTEXT_SUMMARY)
    template = PROMPT_EXTENDER_RESUMEN if resumen_conversacion else PROMPT_RESUMEN
    mensaje_renderizado = Template(template).render(
        resumen_conversacion = resumen_conversacion,
        actor_nombre         = actor_nombre,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("human", mensaje_renderizado),
        ]
    )
    return prompt | model


def obtener_cadena_resumen_contexto(id_actor: str = ""):
    model = obtener_modelo_chat(model_name=OPENAI_LLM_MODEL_CONTEXT_SUMMARY)
    system_rendered = Template(PROMPT_RESUMEN_CONTEXTO).render(id_actor=id_actor)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_rendered),
        ("human", "{context}"),
    ])
    return prompt | model
