#!/usr/bin/env python3
"""Загрузка демо-товаров для проверки поиска."""

import json
import sys
import urllib.error
import urllib.request

API_URL = "http://localhost:8000"

DEMO_PRODUCTS = [
    {
        "name": "Беспроводные наушники Pro X",
        "description": "Шумоподавление, 30 часов работы, Bluetooth 5.3",
        "category": "Электроника",
        "price": 8990,
        "sku": "HP-PRO-X",
    },
    {
        "name": "Кофемашина Espresso Home",
        "description": "Автоматическая, 19 бар, капучинатор",
        "category": "Бытовая техника",
        "price": 24990,
        "sku": "CM-ESP-01",
    },
    {
        "name": "Беговые кроссовки AirRun",
        "description": "Лёгкие, амортизация, для асфальта и беговой дорожки",
        "category": "Спорт",
        "price": 7490,
        "sku": "SH-AIR-RUN",
    },
    {
        "name": "Рюкзак городской Urban 25L",
        "description": "Отделение для ноутбука 15.6, водоотталкивающая ткань",
        "category": "Аксессуары",
        "price": 3990,
        "sku": "BP-URB-25",
    },
    {
        "name": "Умная лампа SmartLight",
        "description": "RGB, управление через приложение, голосовой ассистент",
        "category": "Электроника",
        "price": 1990,
        "sku": "LT-SMART-01",
    },
]


def request(method: str, path: str, data: dict | None = None) -> dict:
    url = f"{API_URL}{path}"
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    print("Checking API health...")
    try:
        health = request("GET", "/health")
    except urllib.error.URLError as exc:
        print(f"API unavailable: {exc}")
        return 1

    print(f"API status: {health['status']}")

    print("Seeding demo products...")
    for product in DEMO_PRODUCTS:
        created = request("POST", "/products", product)
        print(f"  + [{created['id']}] {created['name']}")

    print("\nSample search: 'наушники с шумоподавлением'")
    search = request(
        "POST",
        "/products/search",
        {"query": "наушники с шумоподавлением", "limit": 3},
    )
    for item in search["results"]:
        print(f"  {item['score']:.3f} — {item['product']['name']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
