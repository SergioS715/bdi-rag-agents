from langchain_huggingface import HuggingFaceEmbeddings

ModeloEmbedding = HuggingFaceEmbeddings


def obtener_modelo_embedding(
    id_modelo: str,
    dispositivo: str = "cpu",
) -> ModeloEmbedding:
    """Obtiene una instancia de un modelo de embedding de HuggingFace.

    Args:
        id_modelo (str): El ID/nombre del modelo de embedding de HuggingFace a usar.
        dispositivo (str): El dispositivo de cómputo (ej. "cpu", "cuda"). Por defecto "cpu".

    Returns:
        ModeloEmbedding: Una instancia configurada del modelo de embedding.
    """
    return _obtener_modelo_huggingface(id_modelo, dispositivo)


def _obtener_modelo_huggingface(
    id_modelo: str, dispositivo: str
) -> HuggingFaceEmbeddings:
    """Obtiene una instancia del modelo de embedding de HuggingFace.

    Args:
        id_modelo (str): El ID/nombre del modelo de embedding de HuggingFace a usar.
        dispositivo (str): El dispositivo de cómputo (ej. "cpu", "cuda").

    Returns:
        HuggingFaceEmbeddings: Una instancia configurada del modelo de embedding
            con confianza en código remoto habilitada y normalización desactivada.
    """
    return HuggingFaceEmbeddings(
        model_name=id_modelo,
        model_kwargs={"device": dispositivo, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": False},
    )
