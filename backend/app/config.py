import logging
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/products"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "products"

    ollama_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    embedding_dimension: int = 1024
    ollama_model: str = "qwen2.5:3b-instruct"
    ollama_timeout_seconds: float = 600.0
    ollama_max_tokens: int = 300
    mock_mode: bool = False

    products_json_path: str = "data/products.json"
    search_default_limit: int = 10
    search_max_limit: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
