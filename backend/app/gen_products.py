#!/usr/bin/env python3
"""Генератор синтетического каталога (~64 товара) для демо сравнения поиска."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "products.json"

# SKU -> детерминированный UUID для синхронизации Postgres и Qdrant
_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _pid(sku: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, sku))


PRODUCTS: list[dict] = [
    # Электроника — смартфоны и гаджеты
    {"sku": "PH-IP15", "name": "Apple iPhone 15 128GB", "category": "Смартфоны", "price": 79990,
     "description": "6.1\" OLED, камера 48 Мп, Face ID, флагманский смартфон Apple"},
    {"sku": "PH-S24", "name": "Samsung Galaxy S24", "category": "Смартфоны", "price": 74990,
     "description": "Android 14, AMOLED 120 Гц, тройная камера, Galaxy AI"},
    {"sku": "PH-PX8", "name": "Google Pixel 8", "category": "Смартфоны", "price": 59990,
     "description": "Чистый Android, отличная камера, Tensor G3"},
    {"sku": "PH-XM14", "name": "Xiaomi 14", "category": "Смартфоны", "price": 54990,
     "description": "Leica камера, Snapdragon 8 Gen 3, быстрая зарядка 90W"},
    {"sku": "HP-SONY-WH", "name": "Sony WH-1000XM5", "category": "Наушники", "price": 29990,
     "description": "Беспроводные наушники с активным шумоподавлением, 30 часов автономности"},
    {"sku": "HP-AIRPODS", "name": "Apple AirPods Pro 2", "category": "Наушники", "price": 24990,
     "description": "TWS наушники, прозрачный режим, пространственное аудио"},
    {"sku": "HP-JBL-710", "name": "JBL Tune 710BT", "category": "Наушники", "price": 4990,
     "description": "Накладные Bluetooth наушники, до 50 часов работы"},
    {"sku": "HP-SENN-MTW", "name": "Sennheiser Momentum True Wireless 4", "category": "Наушники", "price": 27990,
     "description": "Премиальные TWS, aptX Adaptive, шумоподавление"},
    {"sku": "NB-ASUS-VB", "name": "ASUS VivoBook 15 OLED", "category": "Ноутбуки", "price": 64990,
     "description": "15.6\" OLED, Intel Core i5, 16 ГБ RAM, SSD 512 ГБ"},
    {"sku": "NB-MBA-M2", "name": "Apple MacBook Air M2", "category": "Ноутбуки", "price": 109990,
     "description": "13.6\" Retina, чип M2, 8 ГБ, SSD 256 ГБ, до 18 часов работы"},
    {"sku": "NB-LEN-IDEA", "name": "Lenovo IdeaPad Slim 5", "category": "Ноутбуки", "price": 54990,
     "description": "14\" IPS, Ryzen 5, 16 ГБ RAM, лёгкий офисный ноутбук"},
    {"sku": "TB-IPAD-AIR", "name": "Apple iPad Air M1", "category": "Планшеты", "price": 59990,
     "description": "10.9\" Liquid Retina, Apple Pencil, для рисования и работы"},
    {"sku": "WT-APPLE-S9", "name": "Apple Watch Series 9", "category": "Умные часы", "price": 39990,
     "description": "GPS 45 мм, пульс, ECG, Always-On Retina"},
    {"sku": "WT-GAL-W6", "name": "Samsung Galaxy Watch 6", "category": "Умные часы", "price": 24990,
     "description": "Wear OS, мониторинг сна, пульс, NFC"},
    {"sku": "SP-JBL-FLIP", "name": "JBL Flip 6", "category": "Колонки", "price": 8990,
     "description": "Портативная Bluetooth колонка, водозащита IP67, 12 часов"},
    {"sku": "DR-DJI-MINI", "name": "DJI Mini 4 Pro", "category": "Дроны", "price": 89990,
     "description": "Компактный дрон с 4K камерой, обход препятствий, 34 мин полёта"},
    # Бытовая техника
    {"sku": "CM-DELONG", "name": "De'Longhi Magnifica S", "category": "Кофемашины", "price": 34990,
     "description": "Автоматическая кофемашина, эспрессо и капучино, 15 бар, зерновой кофе"},
    {"sku": "CM-NESP-V", "name": "Nespresso Vertuo Next", "category": "Кофемашины", "price": 12990,
     "description": "Капсульная кофемашина, центрифужное заваривание, компактная"},
    {"sku": "CM-BREV", "name": "Breville Barista Express", "category": "Кофемашины", "price": 54990,
     "description": "Ручная эспрессо-машина, встроенная кофемолка, паровая трубка"},
    {"sku": "KT-TEFAL", "name": "Tefal Safe Tea Kettle", "category": "Чайники", "price": 3490,
     "description": "Электрический чайник 1.7 л, автоматическое отключение, 2400 Вт"},
    {"sku": "KT-BOSCH", "name": "Bosch Styline TWK8611", "category": "Чайники", "price": 7990,
     "description": "Чайник с регулировкой температуры, 1.5 л, стекло и нержавейка"},
    {"sku": "VC-DYSON-V15", "name": "Dyson V15 Detect", "category": "Пылесосы", "price": 64990,
     "description": "Беспроводной вертикальный пылесос, лазерная подсветка пыли, HEPA"},
    {"sku": "VC-ROBOROCK", "name": "Roborock S8 Pro Ultra", "category": "Пылесосы", "price": 89990,
     "description": "Робот-пылесос с станцией самоочистки, мытьё полов, LiDAR навигация"},
    {"sku": "VC-XIAOMI-G10", "name": "Xiaomi G10 Plus", "category": "Пылесосы", "price": 19990,
     "description": "Беспроводной пылесос для уборки квартиры, 60 мин работы, HEPA фильтр"},
    {"sku": "DR-DYSON-SU", "name": "Dyson Supersonic", "category": "Фены", "price": 39990,
     "description": "Профессиональный фен, ионизация, 4 насадки, быстрая сушка без перегрева"},
    {"sku": "DR-PHILIPS", "name": "Philips BHD500", "category": "Фены", "price": 4990,
     "description": "Компактный фен 2100 Вт, 3 режима температуры, складная ручка"},
    {"sku": "MW-PANASONIC", "name": "Panasonic NN-CD87", "category": "Микроволновки", "price": 14990,
     "description": "Микроволновая печь 27 л, инверторная технология, гриль"},
    {"sku": "BL-VITAMIX", "name": "Vitamix E310", "category": "Блендеры", "price": 44990,
     "description": "Мощный блендер для смузи, супов и соусов, 1.4 л контейнер"},
    {"sku": "RF-LG-INST", "name": "LG InstaView Door-in-Door", "category": "Холодильники", "price": 149990,
     "description": "Двухкамерный холодильник, умный экран, No Frost, 617 л"},
    {"sku": "WM-BOSCH-6", "name": "Bosch Serie 6 WAN28270", "category": "Стиральные машины", "price": 54990,
     "description": "Стиральная машина 9 кг, 1400 об/мин, EcoSilence Drive"},
    # Спорт
    {"sku": "SH-NIMBUS", "name": "ASICS Gel-Nimbus 26", "category": "Кроссовки", "price": 14990,
     "description": "Беговые кроссовки с максимальной амортизацией, для длинных дистанций"},
    {"sku": "SH-NIKE-PEG", "name": "Nike Air Zoom Pegasus 41", "category": "Кроссовки", "price": 12990,
     "description": "Универсальные беговые кроссовки, React foam, для асфальта"},
    {"sku": "SH-ADID-UB", "name": "Adidas Ultraboost Light", "category": "Кроссовки", "price": 15990,
     "description": "Беговые кроссовки Boost, комфорт на каждый день и пробежки"},
    {"sku": "SH-NB-880", "name": "New Balance 880v14", "category": "Кроссовки", "price": 13990,
     "description": "Нейтральные беговые кроссовки, Fresh Foam X, стабильность"},
    {"sku": "SH-SAL-SPD", "name": "Salomon Speedcross 6", "category": "Кроссовки", "price": 11990,
     "description": "Трейловые кроссовки, агрессивный протектор, для бездорожья"},
    {"sku": "SH-REEB-FLT", "name": "Reebok Floatride Energy 5", "category": "Кроссовки", "price": 8990,
     "description": "Недорогие беговые кроссовки, лёгкие, для начинающих бегунов"},
    {"sku": "BI-TREK-MAR", "name": "Trek Marlin 7", "category": "Велосипеды", "price": 89990,
     "description": "Горный велосипед 29\", алюминиевая рама, 1x12 Shimano Deore"},
    {"sku": "YG-MANDUKA", "name": "Manduka PRO Yoga Mat", "category": "Йога", "price": 9990,
     "description": "Профессиональный коврик для йоги 6 мм, нескользящая поверхность"},
    {"sku": "FT-BOWFLEX", "name": "Bowflex SelectTech 552", "category": "Фитнес", "price": 49990,
     "description": "Регулируемые гантели 2-24 кг, замена 15 пар гантелей"},
    {"sku": "SW-GARMIN", "name": "Garmin Forerunner 255", "category": "Спортивные часы", "price": 29990,
     "description": "GPS часы для бега, пульсометр, VO2 max, тренировочные планы"},
    # Одежда и аксессуары
    {"sku": "JK-NORTH-F", "name": "The North Face Thermoball", "category": "Куртки", "price": 18990,
     "description": "Утеплённая куртка синтетический наполнитель, ветрозащита, лёгкая"},
    {"sku": "BP-OSPREY", "name": "Osprey Daylite Plus 20L", "category": "Рюкзаки", "price": 7990,
     "description": "Городской рюкзак 20 л, отделение для ноутбука 13\", ventilated back"},
    {"sku": "BP-HERSCHEL", "name": "Herschel Little America", "category": "Рюкзаки", "price": 9990,
     "description": "Рюкзак 25 л в стиле mountaineering, отделение для ноутбука 15\""},
    {"sku": "GL-RAY-BAN", "name": "Ray-Ban Aviator Classic", "category": "Очки", "price": 12990,
     "description": "Классические солнцезащитные очки, металлическая оправа, UV400"},
    # Дом и сад
    {"sku": "LT-PHILIPS-HUE", "name": "Philips Hue White and Color", "category": "Освещение", "price": 4990,
     "description": "Умная RGB лампа E27, управление через приложение, голосовой ассистент"},
    {"sku": "LT-XIAOMI", "name": "Xiaomi Smart LED Bulb", "category": "Освещение", "price": 990,
     "description": "Умная лампа, регулировка яркости и цветовой температуры, Wi-Fi"},
    {"sku": "AR-DYSON-PUR", "name": "Dyson Purifier Cool TP07", "category": "Климат", "price": 54990,
     "description": "Очиститель и вентилятор воздуха HEPA H13, мониторинг качества воздуха"},
    {"sku": "HM-NEST", "name": "Google Nest Thermostat", "category": "Умный дом", "price": 14990,
     "description": "Умный термостат, экономия энергии, управление с телефона"},
    {"sku": "GR-WEBER", "name": "Weber Spirit II E-310", "category": "Грили", "price": 69990,
     "description": "Газовый гриль 3 конфорки, GS4 система зажигания, для дачи"},
    # Кухня и еда
    {"sku": "CF-LAVAZZA", "name": "Lavazza Qualità Oro", "category": "Кофе", "price": 890,
     "description": "Молотый кофе 250 г, средняя обжарка, арабика, для турки и кофемашины"},
    {"sku": "CF-ILLY", "name": "illy Classico зерно 1 кг", "category": "Кофе", "price": 3490,
     "description": "Кофе в зёрнах, 100% арабика, для эспрессо-машины"},
    {"sku": "CF-STARB", "name": "Starbucks Pike Place", "category": "Кофе", "price": 1290,
     "description": "Молотый кофе 200 г, средняя обжарка, для фильтра и френч-пресса"},
    {"sku": "TE-TWININGS", "name": "Twinings English Breakfast", "category": "Чай", "price": 590,
     "description": "Чёрный чай 100 пакетиков, классический английский завтрак"},
    {"sku": "KN-ZWILLING", "name": "Zwilling Pro Chef Knife 20 см", "category": "Кухонные принадлежности", "price": 8990,
     "description": "Профессиональный поварской нож, немецкая сталь, балансировка"},
    {"sku": "PN-CASTIRON", "name": "Lodge Cast Iron Skillet 26 см", "category": "Посуда", "price": 4990,
     "description": "Чугунная сковорода, для жарки и запекания, предварительно seasoned"},
    # Детские и прочее
    {"sku": "TO-LEGO-TECH", "name": "LEGO Technic Porsche 911", "category": "Конструкторы", "price": 14990,
     "description": "Конструктор 1458 деталей, модель спортивного автомобиля, 18+"},
    {"sku": "BK-HP-1", "name": "Гарри Поттер и философский камень", "category": "Книги", "price": 690,
     "description": "Классический роман Дж. К. Роулинг, мягкая обложка, на русском"},
    {"sku": "GM-PS5-DUAL", "name": "Sony DualSense Edge", "category": "Игровые аксессуары", "price": 17990,
     "description": "Профессиональный геймпад для PlayStation 5, сменные стики и триггеры"},
    {"sku": "GM-LOGITECH", "name": "Logitech G Pro X Superlight 2", "category": "Игровые аксессуары", "price": 14990,
     "description": "Беспроводная игровая мышь 60 г, HERO 2 сенсор, для киберспорта"},
    {"sku": "CAM-CANON-R50", "name": "Canon EOS R50", "category": "Фото", "price": 64990,
     "description": "Беззеркальная камера APS-C, 4K видео, vlog-friendly экран"},
    {"sku": "PEN-IPAD", "name": "Apple Pencil Pro", "category": "Аксессуары", "price": 12990,
     "description": "Стилус для iPad Pro и Air, жесты squeeze, поиск через Find My"},
    {"sku": "CH-ANERGIA", "name": "Anker PowerCore 20000", "category": "Powerbank", "price": 4990,
     "description": "Внешний аккумулятор 20000 mAh, USB-C PD 30W, зарядка ноутбука и телефона"},
    {"sku": "MON-LG-27", "name": "LG UltraGear 27GP850", "category": "Мониторы", "price": 34990,
     "description": "Игровой монитор 27\" QHD 165 Гц, Nano IPS, 1 мс, G-Sync Compatible"},
    {"sku": "KEY-KEYCHRON", "name": "Keychron K2 V2", "category": "Клавиатуры", "price": 8990,
     "description": "Механическая Bluetooth клавиатура 75%, hot-swap, Mac/Windows"},
    {"sku": "SSD-SAMSUNG", "name": "Samsung 990 Pro 2TB", "category": "Накопители", "price": 14990,
     "description": "NVMe SSD M.2 PCIe 4.0, до 7450 МБ/с чтение, для игр и монтажа"},
]


def build_catalog() -> list[dict]:
    items = []
    for p in PRODUCTS:
        items.append({
            "id": _pid(p["sku"]),
            "name": p["name"],
            "description": p["description"],
            "category": p["category"],
            "price": p["price"],
            "sku": p["sku"],
        })
    return items


def main() -> None:
    catalog = build_catalog()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(catalog)} products -> {OUTPUT}")


if __name__ == "__main__":
    main()
