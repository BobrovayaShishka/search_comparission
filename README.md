# Product Search — векторный поиск товаров

Универсальный шаблон проекта: PostgreSQL + pgvector, собственный сервис эмбеддингов и лёгкий FastAPI backend в отдельных Docker-контейнерах.

## Архитектура

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Backend   │────▶│  Embeddings API  │     │  PostgreSQL         │
│  (FastAPI)  │     │ sentence-transf. │     │  + pgvector (HNSW)  │
│  :8000      │────▶│  :8001           │────▶│  :5432              │
└─────────────┘     └──────────────────┘     └─────────────────────┘
```

| Контейнер    | Назначение                                      |
|--------------|-------------------------------------------------|
| `postgres`   | Хранение товаров и векторов, HNSW-индекс        |
| `embeddings` | Локальная модель эмбеддингов (384 dim)          |
| `backend`    | REST API: CRUD товаров + семантический поиск    |

## Быстрый старт

```bash
cp .env.example .env
docker compose up --build -d
```

Первый запуск embeddings-сервиса занимает 1–3 минуты (скачивание модели).

Проверка:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
```

Swagger UI: http://localhost:8000/docs

## Структура проекта

```
.
├── docker-compose.yml
├── .env.example
├── db/init/01_schema.sql      # схема БД + HNSW-индекс
├── embeddings/                # сервис эмбеддингов
│   ├── Dockerfile
│   └── app/main.py
├── backend/                   # REST API
│   ├── Dockerfile
│   └── app/
│       ├── main.py
│       ├── routes/
│       └── services/
└── scripts/seed_demo.py       # демо-данные
```

## API

### Добавить товар

```bash
curl -X POST http://localhost:8000/products \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Беспроводные наушники Pro X",
    "description": "Шумоподавление, 30 часов работы",
    "category": "Электроника",
    "price": 8990,
    "sku": "HP-PRO-X"
  }'
```

### Семантический поиск

```bash
curl -X POST http://localhost:8000/products/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "наушники с шумоподавлением",
    "limit": 5,
    "category": "Электроника",
    "min_score": 0.3
  }'
```

Ответ содержит `score` (0–1, чем выше — тем релевантнее).

### Демо-данные

```bash
python scripts/seed_demo.py
```

## Эффективность поиска

- **pgvector HNSW** — approximate nearest neighbor, O(log n) вместо полного сканирования
- **Cosine distance** (`<=>`) с нормализованными эмбеддингами
- **`hnsw.ef_search = 100`** — баланс скорости и точности (настраивается в `product_service.py`)
- **GIN-индекс** на `metadata` для фильтрации по JSON-полям

## Настройка модели

По умолчанию: `paraphrase-multilingual-MiniLM-L12-v2` (384 dim, поддержка русского).

Смена модели в `.env`:

```env
EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_DEVICE=cpu
```

> При смене модели измените размерность `vector(384)` в `db/init/01_schema.sql` и пересоздайте volume PostgreSQL.

## GPU (опционально)

В `docker-compose.yml` для сервиса `embeddings` добавьте:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

И установите `EMBEDDING_DEVICE=cuda` в `.env`.

## Остановка

```bash
docker compose down        # сохранить данные
docker compose down -v     # удалить volume с БД
```
