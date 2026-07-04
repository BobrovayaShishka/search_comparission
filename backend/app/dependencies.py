from app.cache.search_cache import SearchCache
from app.clients.embeddings_client import EmbeddingsClient
from app.clients.ollama_client import OllamaClient
from app.clients.postgres_client import PostgresManager
from app.clients.qdrant_client import QdrantManager
from app.config import Settings, get_settings
from app.services.answer_service import AnswerService
from app.services.bulk_loader import BulkLoader
from app.services.collection_service import CollectionService
from app.services.product_service import ProductService
from app.services.search_service import SearchService

settings = get_settings()
postgres = PostgresManager(settings)
qdrant = QdrantManager(settings)
embeddings = EmbeddingsClient(settings)
cache = SearchCache(settings)
ollama = OllamaClient(settings)
collection_service = CollectionService(qdrant, settings)
bulk_loader = BulkLoader(qdrant, collection_service, settings)
search_service = SearchService(qdrant, postgres, cache, settings)
answer_service = AnswerService(ollama, settings)
product_service = ProductService(
    qdrant, postgres, embeddings, search_service, bulk_loader, answer_service, settings
)
