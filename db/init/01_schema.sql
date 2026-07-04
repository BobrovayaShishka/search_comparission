-- PostgreSQL: операционные данные (НЕ контент для поиска — он в Qdrant payload)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tenants (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        TEXT NOT NULL,
    slug        TEXT NOT NULL UNIQUE,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    plan        TEXT NOT NULL DEFAULT 'free',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    email       TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'member',
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, email)
);

-- Связь tenant ↔ Qdrant point (без дублирования контента)
CREATE TABLE IF NOT EXISTS product_refs (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id        UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    qdrant_point_id  UUID NOT NULL UNIQUE,
    sku              TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, sku)
);

CREATE TABLE IF NOT EXISTS search_logs (
    id             BIGSERIAL PRIMARY KEY,
    tenant_id      UUID REFERENCES tenants(id) ON DELETE SET NULL,
    query_hash     TEXT NOT NULL,
    latency_ms     DOUBLE PRECISION NOT NULL,
    results_count  INT NOT NULL DEFAULT 0,
    cache_hit      BOOLEAN NOT NULL DEFAULT FALSE,
    filters        JSONB NOT NULL DEFAULT '{}',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS billing_events (
    id          BIGSERIAL PRIMARY KEY,
    tenant_id   UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    event_type  TEXT NOT NULL,
    quantity    INT NOT NULL DEFAULT 1,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_product_refs_tenant ON product_refs (tenant_id);
CREATE INDEX IF NOT EXISTS idx_search_logs_tenant_created ON search_logs (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_search_logs_created ON search_logs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_billing_events_tenant ON billing_events (tenant_id, created_at DESC);

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_tenants_updated_at ON tenants;
CREATE TRIGGER trg_tenants_updated_at
    BEFORE UPDATE ON tenants FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS trg_product_refs_updated_at ON product_refs;
CREATE TRIGGER trg_product_refs_updated_at
    BEFORE UPDATE ON product_refs FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- Демо-арендатор
INSERT INTO tenants (id, name, slug, plan)
VALUES ('00000000-0000-0000-0000-000000000001', 'Demo Tenant', 'demo', 'pro')
ON CONFLICT (slug) DO NOTHING;
