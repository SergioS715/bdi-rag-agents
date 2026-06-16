from pathlib import Path
from dotenv import load_dotenv
import os

# Carga explícita desde la raíz del proyecto (junto a config.py),
# independiente del directorio de trabajo desde donde se arranque el servidor.
load_dotenv(Path(__file__).parent / ".env")

# ── LLMs ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY                   = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL                 = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
OPENAI_LLM_MODEL_CONTEXT_SUMMARY = os.getenv("OPENAI_LLM_MODEL_CONTEXT_SUMMARY", "gpt-4o-mini")
OPENAI_LLM_MODEL_BDI             = os.getenv("OPENAI_LLM_MODEL_BDI", "gpt-4o-mini")

GROQ_API_KEY                   = os.getenv("GROQ_API_KEY")
GROQ_LLM_MODEL                 = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_LLM_MODEL_CONTEXT_SUMMARY = os.getenv("GROQ_LLM_MODEL_CONTEXT_SUMMARY", "llama-3.1-8b-instant")

# ── MongoDB Atlas (RAG) ────────────────────────────────────────────────────────
ATLAS_URI        = os.getenv("ATLAS_URI")
ATLAS_BD_NAME    = os.getenv("ATLAS_BD_NAME", "ChunksProyecto")
ATLAS_COLLECTION = os.getenv("ATLAS_COLLECTION", "Chunks")

# ── MongoDB local ──────────────────────────────────────────────────────────────
MONGO_URI                         = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME                           = os.getenv("DB_NAME", "EntrevistasCEV")
COL_CHUNKS                        = os.getenv("COL_CHUNKS", "chunks")
COL_ENTREVISTAS                   = os.getenv("COL_ENTREVISTAS", "Entrevistas")
COL_CLASIFICACIONES               = os.getenv("COL_CLASIFICACIONES", "Clasificaciones")
COL_INTERVENCIONES                = os.getenv("COL_INTERVENCIONES", "Intervenciones")

# ── MongoDB (Actores / LangGraph) ──────────────────────────────────────────────
MONGO_DB_NAME                     = os.getenv("MONGO_DB_NAME", "Actores")
MONGO_STATE_CHECKPOINT_COLLECTION = os.getenv("MONGO_STATE_CHECKPOINT_COLLECTION", "langgraph_checkpoints")
MONGO_STATE_WRITES_COLLECTION     = os.getenv("MONGO_STATE_WRITES_COLLECTION", "langgraph_writes")
MONGO_LONG_TERM_MEMORY_COLLECTION = os.getenv("MONGO_LONG_TERM_MEMORY_COLLECTION", "Actores_long_term_memory")

# ── Embeddings ────────────────────────────────────────────────────────────────
MODELO_EMBEDDINGS        = os.getenv("MODELO_EMBEDDINGS", "paraphrase-multilingual-mpnet-base-v2")
RAG_TEXT_EMBEDDING_MODEL_ID  = os.getenv("RAG_TEXT_EMBEDDING_MODEL_ID", "paraphrase-multilingual-mpnet-base-v2")
RAG_TEXT_EMBEDDING_MODEL_DIM = int(os.getenv("RAG_TEXT_EMBEDDING_MODEL_DIM", "768"))
RAG_TOP_K      = int(os.getenv("RAG_TOP_K", "5"))
RAG_DEVICE     = os.getenv("RAG_DEVICE", "cpu")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "256"))

# ── Conversación ──────────────────────────────────────────────────────────────
TOTAL_MESSAGES_SUMMARY_TRIGGER = int(os.getenv("TOTAL_MESSAGES_SUMMARY_TRIGGER", "30"))
TOTAL_MESSAGES_AFTER_SUMMARY   = int(os.getenv("TOTAL_MESSAGES_AFTER_SUMMARY", "5"))
