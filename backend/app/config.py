import logging
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # PostgreSQL
    database_url: str = "postgresql://postgres:postgres@localhost:5432/products"
    pg_pool_min_size: int = 2
    pg_pool_max_size: int = 20
    pg_command_timeout: float = 10.0

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_grpc_port: int = 6334
    qdrant_http_port: int = 6333
    qdrant_collection: str = "products"
    qdrant_prefer_grpc: bool = True
    qdrant_hnsw_m: int = 32
    qdrant_hnsw_ef_construct: int = 200

    # Embeddings
    embeddings_url: str = "http://localhost:8001"
    embedding_dimension: int = 384

    # Search
    search_hnsw_ef: int = 128
    search_default_limit: int = 10
    search_max_limit: int = 100

    # Cache
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300

    # Network
    network_timeout_seconds: float = 10.0

    # Bulk load
    bulk_batch_size: int = 500
    bulk_max_retries: int = 5
    bulk_retry_base_delay: float = 0.5
    bulk_parallel_workers: int = 4

    # Demo tenant
    demo_tenant_id: str = "00000000-0000-0000-0000-000000000001"

    # Ollama — опциональная генерация человеческого ответа поверх результатов поиска.
    # Бесплатная лёгкая локальная модель. Если ollama_enabled=False или сервис недоступен,
    # поиск всё равно работает и просто не возвращает поле answer.
    ollama_enabled: bool = True
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:0.5b"
    ollama_timeout_seconds: float = 8.0
    ollama_max_tokens: int = 200


@lru_cache
def get_settings() -> Settings:
    return Settings()


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
