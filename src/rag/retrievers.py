from langchain_huggingface import HuggingFaceEmbeddings
import langchain_mongodb
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import (
    MongoDBAtlasHybridSearchRetriever,
)
from loguru import logger
from config import ATLAS_COLLECTION, ATLAS_BD_NAME, ATLAS_URI

from .embeddings import ModeloEmbedding, obtener_modelo_embedding

TipoRecuperador = MongoDBAtlasHybridSearchRetriever


def obtener_recuperador(
    id_modelo_embedding: str,
    k: int = 3,
    dispositivo: str = "cpu",
) -> TipoRecuperador:
    """Crea y retorna un recuperador de búsqueda híbrida con el modelo de embedding especificado.

    Args:
        id_modelo_embedding (str): El identificador del modelo de embedding a usar.
        k (int, optional): Número de documentos a recuperar. Por defecto 3.
        dispositivo (str, optional): Dispositivo en el que ejecutar el modelo de embedding. Por defecto "cpu".

    Returns:
        TipoRecuperador: Un recuperador de búsqueda híbrida configurado.
    """
    logger.info(
        f"Initializing retriever | model: {id_modelo_embedding} | device: {dispositivo} | top_k: {k}"
    )

    modelo_embedding = obtener_modelo_embedding(id_modelo_embedding, dispositivo)

    return obtener_recuperador_hibrido(modelo_embedding, k)


def obtener_recuperador_hibrido(
    modelo_embedding: HuggingFaceEmbeddings, k: int
) -> MongoDBAtlasHybridSearchRetriever:
    """Crea un recuperador de búsqueda híbrida de MongoDB Atlas con el modelo de embedding dado.

    Args:
        modelo_embedding (HuggingFaceEmbeddings): El modelo de embedding a usar para la búsqueda vectorial.
        k (int): Número de documentos a recuperar.

    Returns:
        MongoDBAtlasHybridSearchRetriever: Un recuperador de búsqueda híbrida configurado que usa tanto
            búsqueda vectorial como búsqueda de texto completo.
    """
    vectorstore = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=ATLAS_URI,
        embedding=modelo_embedding,
        namespace=f"{ATLAS_BD_NAME}.{ATLAS_COLLECTION}",
        text_key="texto",
        embedding_key="embedding",
        relevance_score_fn="dotProduct",
    )

    retriever = MongoDBAtlasHybridSearchRetriever(
        vectorstore=vectorstore,
        search_index_name="hybrid_search_index",
        top_k=k,
        vector_penalty=50,
        fulltext_penalty=50,
    )

    return retriever
