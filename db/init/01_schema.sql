-- Каталог для полнотекстового поиска (tsvector, русский стеммер)
CREATE TABLE IF NOT EXISTS catalog_products (
    id            UUID PRIMARY KEY,
    name          TEXT NOT NULL,
    description   TEXT NOT NULL DEFAULT '',
    category      TEXT NOT NULL DEFAULT '',
    price         NUMERIC(12, 2),
    sku           TEXT UNIQUE,
    search_vector TSVECTOR,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_catalog_search ON catalog_products USING GIN (search_vector);
CREATE INDEX IF NOT EXISTS idx_catalog_category ON catalog_products (category);

CREATE OR REPLACE FUNCTION catalog_products_search_vector_update() RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('russian', coalesce(NEW.name, '')), 'A') ||
        setweight(to_tsvector('russian', coalesce(NEW.category, '')), 'B') ||
        setweight(to_tsvector('russian', coalesce(NEW.description, '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_catalog_search_vector ON catalog_products;
CREATE TRIGGER trg_catalog_search_vector
    BEFORE INSERT OR UPDATE ON catalog_products
    FOR EACH ROW EXECUTE FUNCTION catalog_products_search_vector_update();
