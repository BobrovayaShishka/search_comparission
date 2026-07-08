# Сравнение полнотекстового (Postgres) и векторного (Qdrant) поиска + RAG-ответ

Исследовательский проект: один FastAPI-бэкенд поверх синтетического каталога (~65 товаров в `backend/data/products.json`).
Показывает разницу между **лексическим** поиском (Postgres `tsvector`, русский стеммер) и **семантическим**
(эмбеддинги **bge-m3** через Ollama → cosine-поиск в Qdrant), плюс endpoint `/ask` — RAG-ответ LLM на вопрос покупателя.

Дополнительно: метрики **токенов** и **latency**, сравнение скорости FTS vs vector, автоматические отчёты.

---

## Задача проекта

| Что сравниваем | Как |
|---|---|
| **Релевантность** | Один запрос → `/compare` → результаты FTS и vector рядом |
| **Скорость** | `latency_ms` в каждом ответе; `/compare` → блок `speed`; скрипт `benchmark_fts_vs_vector` |
| **Стоимость LLM** | Поля `tokens` во всех endpoint'ах, где есть embed или генерация |
| **RAG** | `/ask` — vector-поиск + ответ qwen2.5:3b-instruct только по найденным товарам |

---

## Архитектура

```
                         ┌─────────────────────┐
                         │   Клиент            │
                         │ curl / Swagger UI   │
                         └──────────┬──────────┘
                                    │ HTTP :8000
                         ┌──────────▼──────────┐
                         │  FastAPI backend     │
                         │  (search-comparison) │
                         └───┬─────────┬───┬───┘
                             │         │   │
              ┌──────────────┘         │   └────────────────┐
              │                        │                    │
   ┌──────────▼──────────┐  ┌─────────▼────────┐  ┌────────▼────────┐
   │     PostgreSQL       │  │      Qdrant       │  │     Ollama       │
   │  catalog_products    │  │  коллекция        │  │  bge-m3          │
   │  + tsvector/GIN      │  │  products         │  │  qwen2.5:3b      │
   │  (полнотекст)        │  │  (векторный)      │  │  embed + /ask    │
   └──────────────────────┘  └───────────────────┘  └─────────────────┘
```

При старте backend **идемпотентно** заливает каталог из `products.json` в Postgres и Qdrant (если таблица пуста).

---

## Стек

| Компонент | Технология | Версия |
|---|---|---|
| Backend | Python + FastAPI | 3.12 (Docker) |
| PostgreSQL | postgres:16-alpine | FTS `tsvector` |
| Qdrant | qdrant/qdrant | v1.12.5 |
| Ollama | ollama/ollama | 0.3.12 |
| Эмбеддинги | **bge-m3** | 1024 dim, мультиязычная |
| LLM | **qwen2.5:3b-instruct** | RAG `/ask` |
| Оркестрация | docker-compose | 4 сервиса |

---

## Структура репозитория

```
praktikum/
├── docker-compose.yml          # postgres + qdrant + ollama + backend
├── .env.example                # все настройки
├── db/init/01_schema.sql       # таблица catalog_products + tsvector-триггер
└── backend/
    ├── Dockerfile
    ├── requirements.txt
    ├── data/
    │   ├── products.json
    │   ├── comparison.md
    │   └── benchmark_fts_vs_vector.md
    ├── static/
    │   └── index.html          # веб-UI на /
    └── app/
        ├── main.py
        ├── llm.py
        ├── db.py
        ├── vector_store.py
        ├── config.py
        ├── gen_products.py
        ├── compare_report.py
        └── benchmark_fts_vs_vector.py
```

---

## Модели (русский язык)

| Роль | Модель | Зачем |
|---|---|---|
| Эмбеддинги | `bge-m3` | Одна из лучших мультиязычных моделей для русского; 1024 dim |
| LLM | `qwen2.5:3b-instruct` | Компактная instruct-модель, хорошо отвечает на русском в RAG |

Обе модели крутятся **локально через Ollama** — без внешних API и ключей.

---

## Требования

- **Docker** и **Docker Compose** v2
- ~8 GB RAM (qwen2.5:3b + bge-m3 в Ollama)
- ~5 GB диска под модели Ollama
- Windows / Linux / macOS

> Локальный запуск без Docker на Windows с Python 3.14 может не сработать (ошибка сборки `asyncpg`).
> Рекомендуется использовать Docker — в образе Python 3.12-slim.

---

## Настройка

```powershell
Copy-Item .env.example .env
```

Основные переменные (см. `.env.example`):

| Переменная | По умолчанию | Описание |
|---|---|---|
| `EMBED_MODEL` | `bge-m3` | Модель эмбеддингов в Ollama |
| `EMBEDDING_DIMENSION` | `1024` | Размерность bge-m3 (не менять без смены модели) |
| `OLLAMA_MODEL` | `qwen2.5:3b-instruct` | LLM для `/ask` |
| `OLLAMA_TIMEOUT_SECONDS` | `600` | Таймаут генерации (холодный старт на CPU) |
| `OLLAMA_MAX_TOKENS` | `300` | Лимит токенов ответа LLM |
| `MOCK_MODE` | `false` | `true` — hash-эмбеддинги + заглушка LLM без Ollama |
| `BACKEND_PORT` | `8000` | Порт API |

---

## Запуск

### 1. Первый запуск (с реальными моделями)

```powershell
# Поднять Ollama и скачать модели (один раз, ~3–5 GB)
docker compose up -d ollama
docker compose exec ollama ollama pull bge-m3
docker compose exec ollama ollama pull qwen2.5:3b-instruct

# Проверить, что модели на месте
docker compose exec ollama ollama list

# Поднять всё
docker compose up --build -d

# Логи backend (seed каталога + старт API)
docker compose logs -f backend
```

Первый запуск backend занимает 1–3 минуты: эмбеддинги всех ~65 товаров через bge-m3.

### 2. Быстрый демо-режим (без моделей)

В `.env`:

```env
MOCK_MODE=true
```

```powershell
docker compose up --build -d
```

API работает, но эмбеддинги — детерминированный hash, LLM — заглушка. Подходит для проверки структуры ответов, не для оценки качества поиска.

### 3. Ollama на хосте (альтернатива)

Если Ollama уже установлена локально и модели скачаны:

1. Не поднимайте контейнер `ollama` (закомментируйте в compose или `docker compose up postgres qdrant backend`).
2. В `.env` для backend укажите `OLLAMA_URL=http://host.docker.internal:11434` (в compose уже есть `extra_hosts` при необходимости).

---

## Проверка работоспособности

**Веб-UI (главное для демо):** http://localhost:8000/

Кнопки: Сравнить · FTS · Vector · RAG /ask — плюс готовые примеры запросов, latency и токены.

**Swagger UI:** http://localhost:8000/docs

```powershell
# Статус всех сервисов + модели
curl "http://localhost:8000/health"

# Сравнение обоих движков
curl "http://localhost:8000/compare?q=кроссовки+для+бега"

# Только полнотекст
curl "http://localhost:8000/search/fulltext?q=Gel-Nimbus"

# Только векторный
curl "http://localhost:8000/search/vector?q=что-нибудь+для+приготовления+кофе"

# RAG-ответ
curl "http://localhost:8000/ask?q=посоветуй+недорогие+беговые+кроссовки"
```

Ожидаемый `/health`:

```json
{
  "status": "ok",
  "postgres": { "status": "ok", "catalog_count": 65 },
  "qdrant": { "status": "ok", "points_count": 65 },
  "ollama": { "status": "ok", "embed_available": true, "llm_available": true },
  "mock_mode": false,
  "embed_model": "bge-m3",
  "llm_model": "qwen2.5:3b-instruct"
}
```

---

## API

### `GET /search/fulltext?q=&limit=10`

Лексический поиск: Postgres `websearch_to_tsquery('russian', …)` → при пустом результате fallback `plainto_tsquery`.

**Пример ответа:**

```json
{
  "query": "Gel-Nimbus",
  "hits": [{ "id": "...", "score": 0.82, "name": "ASICS Gel-Nimbus 26", "price": 14990, "sku": "SH-NIMBUS" }],
  "total": 1,
  "mode": "strict",
  "latency_ms": 4.2,
  "tokens": null
}
```

### `GET /search/vector?q=&limit=10`

Семантический поиск: запрос → bge-m3 → Qdrant cosine.

```json
{
  "query": "гаджет для уборки",
  "hits": [{ "id": "...", "score": 0.87, "name": "Dyson V15 Detect", "category": "Пылесосы" }],
  "latency_ms": 312.5,
  "latency_embed_ms": 298.1,
  "latency_qdrant_ms": 14.4,
  "tokens": { "embed_tokens": 8, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 8 }
}
```

### `GET /compare?q=&limit=5`

Оба движка + сравнение скорости + embed-токены.

```json
{
  "query": "...",
  "fulltext": { "...": "..." },
  "vector": { "...": "..." },
  "speed": {
    "winner": "fulltext",
    "fulltext_ms": 3.1,
    "vector_ms": 285.0,
    "delta_ms": 281.9,
    "vector_embed_ms": 270.0,
    "vector_qdrant_ms": 15.0
  },
  "tokens": {
    "fulltext": null,
    "vector_embed": { "embed_tokens": 6, "total_tokens": 6 }
  }
}
```

### `GET /ask?q=&limit=5`

RAG: vector top-5 → сортировка по цене → промпт → qwen2.5:3b-instruct.

```json
{
  "query": "недорогие беговые кроссовки",
  "answer": "Из найденных товаров самые доступные беговые кроссовки — ...",
  "sources": [{ "name": "Reebok Floatride Energy 5", "price": 8990, "score": 0.84 }],
  "latency_ms": 4520.0,
  "latency_embed_ms": 280.0,
  "latency_qdrant_ms": 12.0,
  "latency_llm_ms": 4228.0,
  "tokens": {
    "embed_tokens": 12,
    "prompt_tokens": 420,
    "completion_tokens": 95,
    "total_tokens": 527
  },
  "tokens_breakdown": {
    "embed": { "embed_tokens": 12 },
    "llm_prompt": 420,
    "llm_completion": 95
  }
}
```

---

## Метрики токенов

| Endpoint | Что считается |
|---|---|
| `/search/fulltext` | `tokens: null` (LLM не вызывается) |
| `/search/vector` | `embed_tokens` — оценка по длине текста запроса (Ollama embed не всегда отдаёт счётчик) |
| `/compare` | `tokens.vector_embed` — embed-токены vector-части |
| `/ask` | `embed_tokens` + `prompt_tokens` + `completion_tokens` от Ollama chat (`prompt_eval_count`, `eval_count`) |

**Latency:**

- FTS — только время SQL-запроса
- Vector — `latency_embed_ms` (bge-m3) + `latency_qdrant_ms` (поиск в Qdrant)
- `/ask` — vector + `latency_llm_ms` (генерация)

---

## Отчёты (для защиты / сравнения с коллегами)

Запускать **при работающем backend**:

```powershell
# 1. Релевантность + скорость + токены + примеры RAG (10 запросов)
docker compose exec backend python -m app.compare_report
# → backend/data/comparison.md

# 2. Бенчмарк скорости FTS vs vector (p50/p95, 12 запросов × 5 раундов)
docker compose exec backend python -m app.benchmark_fts_vs_vector
# → backend/data/benchmark_fts_vs_vector.md
```

---

## Показательные запросы

| Запрос | FTS | Vector |
|---|---|---|
| «что-нибудь для приготовления кофе» | часто пусто | кофемашины |
| «гаджет для уборки квартиры» | часто пусто | пылесосы |
| «устройство чтобы слушать музыку без проводов» | часто пусто | наушники |
| «кросовки» (опечатка) | часто пусто | кроссовки |
| «Gel-Nimbus», «ASUS VivoBook» | точное совпадение | тоже находит |
| «чайник», «кофе» | находит | находит |
| «айфон» | может найти iPhone | может увести к «похожим гаджетам» — нужен гибрид |

---

## Перегенерация каталога

```powershell
# Сгенерировать products.json (~65 товаров)
py -3 backend/app/gen_products.py

# Сбросить данные и перезалить
docker compose down -v
docker compose up --build -d
```

---

## Реализация (кратко)

- **Postgres:** `tsvector` с весами (название A, категория B, описание C), GIN-индекс, `websearch_to_tsquery('russian', …)`; если строгий AND ничего не дал — мягкий OR через `plainto_tsquery`.
- **Qdrant:** коллекция под 1024 dim (bge-m3), cosine distance; создаётся автоматически при старте.
- **Seed:** один JSON → Postgres (FTS) + Qdrant (векторы); повторный старт не дублирует данные.
- **`/ask`:** контекст сортируется по цене; промпт запрещает выдумывать товары.

---

## Устранение неполадок

| Проблема | Решение |
|---|---|
| Backend не стартует, `ollama` degraded | `docker compose exec ollama ollama pull bge-m3` и `qwen2.5:3b-instruct` |
| `/ask` долго отвечает | Первый вызов LLM на CPU — до 600 с (`OLLAMA_TIMEOUT_SECONDS`); повторные быстрее |
| Пустой vector-поиск после смены модели | `docker compose down -v` — старые векторы несовместимы с новой размерностью |
| `catalog_count: 0` | Проверьте наличие `backend/data/products.json`; смотрите логи `docker compose logs backend` |
| Порт 8000 занят | В `.env`: `BACKEND_PORT=8001` |

---

## Хостинг

### Раздел «Приложения» на хостинге (сборка из Git)

В корне репозитория есть **`Dockerfile`** — хостинг собирает **только backend** (FastAPI + UI).

**Настройки сборки в панели:**

| Параметр | Значение |
|----------|----------|
| Dockerfile | `Dockerfile` (корень репо) |
| Контекст сборки | `.` (корень репо) |
| Порт приложения | `8000` |

Отдельно в разделе «Приложения» подними **3 сервиса** (или managed):

| Сервис | Версия | Переменная для backend |
|--------|--------|------------------------|
| PostgreSQL | **16** | `DATABASE_URL=postgresql://user:pass@host:5432/db` |
| Qdrant | **1.12.x** | `QDRANT_URL=http://host:6333` |
| Ollama | **0.3.x** | `OLLAMA_URL=http://host:11434` |

**Переменные окружения backend** (обязательные):

```env
DATABASE_URL=postgresql://...
QDRANT_URL=http://...
OLLAMA_URL=http://...
EMBED_MODEL=bge-m3
EMBEDDING_DIMENSION=1024
OLLAMA_MODEL=qwen2.5:3b-instruct
MOCK_MODE=false
```

После деплоя Postgres выполни SQL из `db/init/01_schema.sql` (таблица `catalog_products`). Backend при старте сам зальёт товары в Postgres и Qdrant.

В Ollama на хосте: `ollama pull bge-m3` и `ollama pull qwen2.5:3b-instruct`.

> **Ошибка «Dockerfile not found»** — хостинг ищет файл в **корне** репозитория, не в `backend/`. Используй корневой `Dockerfile` из репо (не `backend/Dockerfile` в настройках, если контекст — корень).

### VPS с Docker Compose (рекомендуется для всего стека)

```bash
docker compose up --build -d
```

Снаружи открыт только **порт 8000** (UI + API).

1. VPS **8 GB RAM** / 4 vCPU
2. `git clone`, `.env`, `ollama pull` для моделей
3. `docker compose up --build -d`

### Что не выставлять наружу

| Порт | Сервис |
|------|--------|
| 5432 | Postgres |
| 6333 | Qdrant |
| 11434 | Ollama |

---

## Остановка

```powershell
docker compose down          # сохранить данные (Postgres, Qdrant, модели Ollama)
docker compose down -v         # удалить все volume — полный сброс
```
