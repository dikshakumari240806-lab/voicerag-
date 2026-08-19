import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY","")
    openai_model: str = os.getenv("OPENAI_MODEL","gpt-4o-mini")
    embed_model: str = os.getenv("EMBED_MODEL","sentence-transformers/all-MiniLM-L6-v2")
    vector_backend: str = os.getenv("VECTOR_BACKEND","faiss").lower()
    top_k: int = int(os.getenv("TOP_K","5"))
    min_retrieval_score: float = float(os.getenv("MIN_RETRIEVAL_SCORE","0.30"))
    whisper_model: str = os.getenv("WHISPER_MODEL","base")
    qdrant_path: str = os.getenv("QDRANT_PATH","storage/qdrant")
    faiss_dir: str = "storage/faiss"
settings = Settings()
