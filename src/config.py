import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CHROMA_PATH = os.environ.get("CHROMA_PATH", os.path.join(BASE_DIR, "chroma_db"))
COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION", "nepali_legal_docs")
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
EXTRACT_CACHE_DIR = os.path.join(MODELS_DIR, "extract_cache")

# ---- Retrieval models ------------------------------------------------
# UPGRADE: BAAI/bge-m3 has vastly better Devanagari (Nepali) coverage than
# paraphrase-multilingual-MiniLM-L12-v2, and a much larger context window.
# NOTE: switching models changes vector dimensionality — you MUST reindex
#       with `python -m src.ingest --rebuild` after changing this value.
# Set EMBEDDING_MODEL_PATH to a local directory to keep the old model.
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
EMBEDDING_MODEL_PATH = os.environ.get(
    "EMBEDDING_MODEL_PATH",
    EMBEDDING_MODEL_NAME,
)

# UPGRADE: multilingual reranker. The previous cross-encoder/ms-marco-MiniLM-L-6-v2
# is trained almost exclusively on English and distorts rankings on Devanagari.
CROSS_ENCODER_MODEL = os.environ.get(
    "CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3"
)
CROSS_ENCODER_FALLBACK = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BM25_INDEX_PATH = os.path.join(MODELS_DIR, "bm25_index.pkl")
CASE_INDEX_PATH = os.path.join(MODELS_DIR, "case_index.json")
INGEST_METADATA_PATH = os.path.join(MODELS_DIR, "ingest_metadata.json")
CASE_SUMMARIES_PATH = os.path.join(MODELS_DIR, "case_summaries.json")

# Voice & Live Model Settings
VOICE_LIVE = os.environ.get("VOICE_LIVE", "on").lower() in ("on", "true", "1")
VOICE_LANGUAGE = os.environ.get("VOICE_LANGUAGE", "ne-NP")
GEMINI_VOICE = os.environ.get("GEMINI_VOICE", "Kore")
GEMINI_LIVE_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview").replace("models/", "")
GEMINI_STT_MODEL = os.environ.get("GEMINI_STT_MODEL", "gemini-3.5-transcribe").replace("models/", "")

EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "32"))  # bge-m3 is heavy
CHROMA_UPSERT_BATCH = int(os.environ.get("CHROMA_UPSERT_BATCH", "500"))
BM25_CANDIDATE_MULTIPLIER = int(os.environ.get("BM25_CANDIDATE_MULTIPLIER", "8"))
VECTOR_CANDIDATE_CAP = int(os.environ.get("VECTOR_CANDIDATE_CAP", "500"))

# Chunking: keep legal arguments intact.  Prakaran (section) based chunking
# gets a larger budget so a full issue + reasoning stays in one chunk.
CHUNK_MAX_CHARS = int(os.environ.get("CHUNK_MAX_CHARS", "1000"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("CHUNK_OVERLAP_CHARS", "150"))
PRAKARAN_MAX_CHARS = int(os.environ.get("PRAKARAN_MAX_CHARS", "1800"))
PRAKARAN_OVERLAP_CHARS = int(os.environ.get("PRAKARAN_OVERLAP_CHARS", "200"))

LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
METADATA_LLM_MODEL = os.environ.get("METADATA_LLM_MODEL", "openai/gpt-oss-20b")
CHUNK_DOCS_PATH = os.path.join(MODELS_DIR, "chunk_docs.jsonl")
INGEST_WORKERS = int(os.environ.get("INGEST_WORKERS", "4"))
QUERY_VEC_CACHE_SIZE = int(os.environ.get("QUERY_VEC_CACHE_SIZE", "256"))

# Hybrid search weights
BM25_WEIGHT = 0.3
VECTOR_WEIGHT = 0.3
RRF_K = 60.0


def ensure_models_dir() -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EXTRACT_CACHE_DIR, exist_ok=True)
    return MODELS_DIR