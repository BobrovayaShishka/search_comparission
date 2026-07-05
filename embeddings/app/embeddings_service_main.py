import os
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

model: SentenceTransformer | None = None
# Больше не хардкодим EMBEDDING_DIM=384. Размерность определяется реальной
# моделью после загрузки — если кто-то поменяет EMBEDDING_MODEL на модель с
# другой размерностью, /health немедленно отразит это, а не будет врать
# основному приложению константой из кода.
embedding_dim: int | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, embedding_dim
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    embedding_dim = model.get_sentence_embedding_dimension()
    yield
    model = None
    embedding_dim = None


app = FastAPI(title="Embeddings Service", version="1.1.0", lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: Annotated[list[str], Field(min_length=1, max_length=256)]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    model: str
    dimension: int


class HealthResponse(BaseModel):
    status: str
    model: str
    dimension: int


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if model is None or embedding_dim is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="ok", model=MODEL_NAME, dimension=embedding_dim)


@app.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest) -> EmbedResponse:
    if model is None or embedding_dim is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    vectors = model.encode(
        payload.texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return EmbedResponse(
        embeddings=vectors.tolist(),
        model=MODEL_NAME,
        dimension=embedding_dim,
    )
