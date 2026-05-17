from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # App
    app_name: str = "GeoGuessr AI"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://geo:geo@localhost/geoguessr"

    # Models
    clip_model: str = "ViT-B/32"
    yolo_model: str = "yolov8m.pt"
    embed_model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"

    # RAG / retrieval
    faiss_index_path: Path = Path("data/faiss.index")
    kb_path: Path = Path("knowledge_base/clues.json")
    retrieval_top_k: int = 15
    retrieval_threshold: float = 0.60

    # Redis
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600

    # Image processing
    max_image_dim: int = 1024

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()