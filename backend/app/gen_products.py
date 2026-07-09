#!/usr/bin/env python3
"""Генератор синтетического каталога (~250 товаров) для демо сравнения поиска."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "products.json"
_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _pid(sku: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, sku))


def _item(sku: str, name: str, category: str, price: int, description: str) -> dict:
    return {"sku": sku, "name": name, "category": category, "price": price, "description": description}


# --- Шаблонные генераторы (богатые описания для векторного поиска) ---

def _running_shoes() -> list[dict]:
    items = [
        ("SH-NIKE-PEG", "Nike Air Zoom Pegasus 41", 12990, "Универсальные беговые кроссовки, React foam, для асфальта"),
        ("SH-NIMBUS", "ASICS Gel-Nimbus 26", 14990, "Беговые кроссовки с максимальной амортизацией, для длинных дистанций"),
        ("SH-REEB-FLT", "Reebok Floatride Energy 5", 8990, "Недорогие лёгкие беговые кроссовки для начинающих бегунов"),
        ("SH-NIKE-VAPR", "Nike Vaporfly 3", 19990, "Лёгкие соревновательные беговые кроссовки с карбоновой пластиной"),
        ("SH-NIKE-REV", "Nike Revolution 7", 5990, "Бюджетные лёгкие кроссовки для бега и ходьбы, сетчатый верх"),
        ("SH-NIKE-DWN", "Nike Downshifter 13", 4990, "Доступные беговые кроссовки для тренировок на дорожке"),
        ("SH-NIKE-INV", "Nike Invincible 3", 16990, "Максимальная амортизация ZoomX для длительных беговых тренировок"),
        ("SH-NIKE-TEM", "Nike Tempo Next%", 15990, "Лёгкие беговые кроссовки для темповых пробежек"),
        ("SH-ADID-UB", "Adidas Ultraboost Light", 15990, "Беговые кроссовки Boost, комфорт на пробежки по городу"),
        ("SH-ADID-BOS", "Adidas Boston 12", 13990, "Лёгкие беговые кроссовки для темпового бега и марафона"),
        ("SH-ADID-SL", "Adidas Supernova Stride", 9990, "Ежедневные беговые кроссовки с поддержкой стопы"),
        ("SH-ADID-RUN", "Adidas Runfalcon 5", 5490, "Недорогие беговые кроссовки для новичков, лёгкий вес"),
        ("SH-NB-880", "New Balance Fresh Foam 880v14", 13990, "Нейтральные беговые кроссовки, Fresh Foam X"),
        ("SH-NB-1080", "New Balance Fresh Foam X 1080v13", 16990, "Премиальные беговые кроссовки для длинных дистанций"),
        ("SH-NB-574", "New Balance 574 Core", 8990, "Классические лёгкие кроссовки для бега трусцой"),
        ("SH-NB-FUEL", "New Balance FuelCell Rebel v4", 14990, "Лёгкие быстрые беговые кроссовки для темпа"),
        ("SH-ASICS-GT", "ASICS GT-2000 12", 12990, "Стабильные беговые кроссовки с поддержкой перронирования"),
        ("SH-ASICS-NOV", "ASICS Novablast 4", 13990, "Лёгкие упругие беговые кроссовки для скоростных тренировок"),
        ("SH-ASICS-DYN", "ASICS Dynablast 4", 11990, "Лёгкие беговые кроссовки для ежедневных пробежек"),
        ("SH-SAL-SPD", "Salomon Speedcross 6", 11990, "Трейловые беговые кроссовки для грунта и бездорожья"),
        ("SH-SAL-SEN", "Salomon Sense Ride 5", 10990, "Универсальные лёгкие кроссовки для трейлраннинга"),
        ("SH-HOKA-CL", "Hoka Clifton 9", 14990, "Лёгкие мягкие беговые кроссовки для восстановительных пробежек"),
        ("SH-HOKA-BON", "Hoka Bondi 8", 16990, "Максимальная амортизация для длинного бега по асфальту"),
        ("SH-HOKA-RIN", "Hoka Rincon 3", 12990, "Сверхлёгкие беговые кроссовки для быстрого бега"),
        ("SH-BROKS-GHO", "Brooks Ghost 16", 13990, "Универсальные нейтральные беговые кроссовки на каждый день"),
        ("SH-BROKS-LA", "Brooks Launch 10", 9990, "Лёгкие беговые кроссовки для скоростных тренировок"),
        ("SH-MIZ-WAVE", "Mizuno Wave Rider 27", 12990, "Беговые кроссовки с волновой пластиной, лёгкий бег"),
        ("SH-PUMA-DEV", "Puma Deviate Nitro 2", 11990, "Лёгкие беговые кроссовки с подошвой Nitro"),
        ("SH-SKETCH-GO", "Skechers Go Run Consistent 2", 6990, "Лёгкие недорогие беговые кроссовки для фитнеса"),
        ("SH-XIAOMI-RN", "Xiaomi Mijia Running Shoes", 3990, "Бюджетные лёгкие кроссовки для бега и зала"),
    ]
    return [_item(sku, name, "Кроссовки", price, desc) for sku, name, price, desc in items]


def _kitchen_appliances() -> list[dict]:
    items = [
        ("KA-MW-PAN", "Panasonic NN-CD87", 14990, "Микроволновая печь для кухни, инвертор, гриль, 27 л"),
        ("KA-MW-SAM", "Samsung MS23K3515AS", 8990, "Компактная микроволновка для кухни, 23 л, 800 Вт"),
        ("KA-MW-LG", "LG NeoChef 25 л", 10990, "Микроволновка для кухни с равномерным разогревом"),
        ("KA-BL-VITA", "Vitamix E310", 44990, "Мощный кухонный блендер для смузи, супов и соусов"),
        ("KA-BL-PHIL", "Philips HR3655", 8990, "Кухонный блендер ProMix, измельчение овощей и фруктов"),
        ("KA-BL-XIAOMI", "Xiaomi Blender Pro", 4990, "Недорогой кухонный блендер для коктейлей и пюре"),
        ("KA-MX-KIT", "KitchenAid Artisan 5KSM", 59990, "Планетарный кухонный миксер для теста и выпечки"),
        ("KA-MX-BOSCH", "Bosch MUM5", 24990, "Кухонная машина-миксер с насадками для домашней кухни"),
        ("KA-TO-TEF", "Tefal Toast Lite", 2990, "Тостер для кухни, 2 слота, хрустящий хлеб"),
        ("KA-TO-BOS", "Bosch TAT3A", 3490, "Компактный тостер для кухни, 8 степеней поджаривания"),
        ("KA-MC-RED", "Redmond RMC-M90", 6990, "Мультиварка для кухни, 5 л, 19 программ готовки"),
        ("KA-MC-POL", "Polaris PMC 0517", 5490, "Мультиварка-скороварка для кухни, 5 литров"),
        ("KA-AG-NIN", "Ninja Foodi MAX", 19990, "Аэрогриль для кухни, жарка без масла, 7.5 л"),
        ("KA-AG-PHIL", "Philips Airfryer XXL", 14990, "Фритюрница-аэрогриль для кухни, Rapid Air"),
        ("KA-AG-XIA", "Xiaomi Smart Air Fryer", 7990, "Умный аэрогриль для кухни, приложение Mi Home"),
        ("KA-KT-TEF", "Tefal Safe Tea Kettle", 3490, "Электрический чайник для кухни, 1.7 л, 2400 Вт"),
        ("KA-KT-BOS", "Bosch Styline TWK8611", 7990, "Чайник для кухни с регулировкой температуры"),
        ("KA-KT-SME", "Smeg KLF03", 12990, "Дизайнерский чайник для кухни, ретро-стиль"),
        ("KA-CM-DEL", "De'Longhi Magnifica S", 34990, "Кофемашина для кухни, эспрессо и капучино, 15 бар"),
        ("KA-CM-NES", "Nespresso Vertuo Next", 12990, "Капсульная кофемашина для кухни, компактная"),
        ("KA-CM-BREV", "Breville Barista Express", 54990, "Эспрессо-машина для кухни с кофемолкой"),
        ("KA-CM-DOL", "Dolce Gusto Genio S", 6990, "Капсульная кофемашина для кухни, латте и капучино"),
        ("KA-GR-BOS", "Bosch TKA6A", 4990, "Капельная кофеварка для кухни, 1.2 л"),
        ("KA-JR-BRA", "Braun MultiQuick 9", 9990, "Погружной кухонный блендер с насадками"),
        ("KA-SC-TEF", "Tefal Secure 5", 3990, "Пароварка для кухни, 3 корзины, таймер"),
        ("KA-WM-BOS", "Bosch Serie 6 WAN28270", 54990, "Стиральная машина для кухни и ванной, 9 кг"),
        ("KA-DW-BOS", "Bosch SMS46", 44990, "Посудомоечная машина для кухни, 60 см, 13 комплектов"),
        ("KA-RF-LG", "LG InstaView 617 л", 149990, "Холодильник для кухни, No Frost, двухкамерный"),
        ("KA-RF-SAM", "Samsung RB37", 69990, "Двухкамерный холодильник для кухни, 367 л"),
        ("KA-OV-ELE", "Electrolux OEF5E50X", 39990, "Встраиваемый духовой шкаф для кухни, 72 л"),
        ("KA-CK-RED", "Redmond RMC-M452", 4490, "Электрическая сковорода для кухни, антипригарное покрытие"),
        ("KA-YM-KIT", "KitchenAid 5KSM150", 39990, "Йогуртница и миксер для кухни, набор насадок"),
        ("KA-WP-KIT", "KitchenAid 5KFP", 19990, "Кухонный процессор для нарезки и шинковки"),
        ("KA-GR-PHIL", "Philips Daily Collection", 2990, "Электрический гриль для кухни, контактный"),
        ("KA-EG-TEF", "Tefal OptiGrill XL", 12990, "Электрический гриль для кухни, автоопределение толщины"),
    ]
    return [_item(sku, name, "Кухонная техника", price, desc) for sku, name, price, desc in items]


def _vacuums_cleaning() -> list[dict]:
    items = [
        ("VC-DYSON-V15", "Dyson V15 Detect", 64990, "Беспроводной пылесос для уборки квартиры, лазерная подсветка"),
        ("VC-ROBOROCK", "Roborock S8 Pro Ultra", 89990, "Робот-пылесос для уборки, станция самоочистки, мытьё полов"),
        ("VC-XIAOMI-G10", "Xiaomi G10 Plus", 19990, "Беспроводной пылесос для уборки дома, 60 мин, HEPA"),
        ("VC-SAM-JET", "Samsung Jet 75", 34990, "Вертикальный пылесос для уборки, мощное всасывание"),
        ("VC-PHIL-SPD", "Philips SpeedPro Max", 24990, "Беспроводной пылесос для уборки квартиры"),
        ("VC-BOSCH-UN", "Bosch Unlimited 7", 29990, "Пылесос для уборки, сменные аккумуляторы"),
        ("VC-KARCHER", "Kärcher VC 4", 14990, "Лёгкий пылесос для уборки мебели и пола"),
        ("VC-DREAME", "Dreame L10 Ultra", 59990, "Робот-пылесос с влажной уборкой для квартиры"),
    ]
    return [_item(sku, name, "Пылесосы", price, desc) for sku, name, price, desc in items]


def _headphones_audio() -> list[dict]:
    items = [
        ("HP-SONY-WH", "Sony WH-1000XM5", 29990, "Беспроводные наушники с шумоподавлением, слушать музыку без проводов"),
        ("HP-AIRPODS", "Apple AirPods Pro 2", 24990, "TWS наушники, прозрачный режим, музыка без проводов"),
        ("HP-JBL-710", "JBL Tune 710BT", 4990, "Накладные Bluetooth наушники, 50 часов, музыка без проводов"),
        ("HP-SENN-MTW", "Sennheiser Momentum TW 4", 27990, "Премиальные TWS наушники, aptX, шумоподавление"),
        ("HP-JBL-FLIP", "JBL Flip 6", 8990, "Портативная колонка Bluetooth, слушать музыку, IP67"),
        ("HP-MARSH", "Marshall Emberton II", 12990, "Компактная Bluetooth колонка, рок-стиль"),
        ("HP-BOSE-QC", "Bose QuietComfort Ultra", 34990, "Наушники с активным шумоподавлением, премиум звук"),
        ("HP-SONY-LINK", "Sony LinkBuds S", 12990, "Лёгкие TWS наушники для музыки и звонков"),
    ]
    return [_item(sku, name, "Наушники", price, desc) for sku, name, price, desc in items]


def _phones_laptops() -> list[dict]:
    items = [
        ("PH-IP15", "Apple iPhone 15 128GB", 79990, "Флагманский смартфон Apple, OLED, камера 48 Мп"),
        ("PH-S24", "Samsung Galaxy S24", 74990, "Смартфон Android, AMOLED 120 Гц, Galaxy AI"),
        ("PH-PX8", "Google Pixel 8", 59990, "Смартфон с чистым Android, отличная камера"),
        ("PH-XM14", "Xiaomi 14", 54990, "Смартфон Leica камера, Snapdragon 8 Gen 3"),
        ("PH-IP14", "Apple iPhone 14", 64990, "Смартфон Apple, A15 Bionic, двойная камера"),
        ("PH-S23", "Samsung Galaxy S23", 54990, "Компактный флагманский смартфон Samsung"),
        ("PH-REDMI", "Redmi Note 13 Pro", 24990, "Недорогой смартфон с хорошей камерой"),
        ("PH-HONOR", "Honor 200", 39990, "Смартфон с портретной камерой Studio"),
        ("NB-ASUS-VB", "ASUS VivoBook 15 OLED", 64990, "Ноутбук 15.6 OLED, Core i5, 16 ГБ RAM"),
        ("NB-MBA-M2", "Apple MacBook Air M2", 109990, "Лёгкий ноутбук M2, 18 часов автономности"),
        ("NB-LEN-IDEA", "Lenovo IdeaPad Slim 5", 54990, "Офисный ноутбук Ryzen 5, 16 ГБ"),
        ("NB-HP-PAV", "HP Pavilion 15", 49990, "Универсальный ноутбук для учёбы и работы"),
        ("NB-ACER-SW", "Acer Swift Go 14", 59990, "Лёгкий ультрабук OLED для мобильной работы"),
        ("TB-IPAD-AIR", "Apple iPad Air M1", 59990, "Планшет для работы и творчества"),
        ("TB-SAM-S9", "Samsung Galaxy Tab S9", 69990, "Android планшет с S Pen, AMOLED"),
    ]
    cats = {
        "PH": "Смартфоны", "NB": "Ноутбуки", "TB": "Планшеты",
    }
    result = []
    for sku, name, price, desc in items:
        prefix = sku.split("-")[0]
        cat = cats.get(prefix, "Электроника")
        result.append(_item(sku, name, cat, price, desc))
    return result


def _hair_dryers() -> list[dict]:
    """Отдельная узкая категория — чтобы не путались с кроссовками."""
    items = [
        ("DR-DYSON-SU", "Dyson Supersonic", 39990, "Профессиональный фен для волос, ионизация, быстрая сушка"),
        ("DR-PHILIPS", "Philips BHD500", 4990, "Компактный фен для волос, 2100 Вт"),
        ("DR-BABYLS", "BaByliss AS6700", 6990, "Фен-щётка для укладки волос"),
        ("DR-REMING", "Remington D3198", 3490, "Недорогой фен для сушки волос дома"),
    ]
    return [_item(sku, name, "Фены", price, desc) for sku, name, price, desc in items]


def _sport_other() -> list[dict]:
    items = [
        ("BI-TREK-MAR", "Trek Marlin 7", 89990, "Горный велосипед 29\", для спорта на природе"),
        ("YG-MANDUKA", "Manduka PRO Yoga Mat", 9990, "Коврик для йоги и фитнеса"),
        ("FT-BOWFLEX", "Bowflex SelectTech 552", 49990, "Гантели для домашних тренировок"),
        ("SW-GARMIN", "Garmin Forerunner 255", 29990, "GPS часы для бега и тренировок"),
        ("SP-FIT-RES", "Resistance Bands Set", 1990, "Набор резиновых эспандеров для фитнеса"),
        ("SP-JUMP-R", "Скакалка скоростная", 890, "Скакалка для кардио и бокса"),
        ("SP-FOAM-R", "Ролик массажный Grid", 4990, "Массажный ролик для восстановления мышц после бега"),
        ("SP-BALL-F", "Мяч фитнес 65 см", 1490, "Фитнес-мяч для тренировок дома"),
    ]
    cats = {"BI": "Велосипеды", "YG": "Йога", "FT": "Фитнес", "SW": "Спортивные часы", "SP": "Фитнес"}
    result = []
    for sku, name, price, desc in items:
        prefix = sku.split("-")[0]
        cat = cats.get(prefix, "Спорт")
        result.append(_item(sku, name, cat, price, desc))
    return result


def _more_running_shoes() -> list[dict]:
    items = [
        ("SH-ON-CLOUD", "On Cloud 6", 15990, "Швейцарские лёгкие беговые кроссовки, CloudTec подошва"),
        ("SH-ON-CLOUDMON", "On Cloudmonster", 17990, "Лёгкие беговые кроссовки с максимальной амортизацией"),
        ("SH-SAU-RI", "Saucony Ride 17", 12990, "Нейтральные беговые кроссовки для ежедневных пробежек"),
        ("SH-SAU-END", "Saucony Endorphin Speed 3", 14990, "Лёгкие быстрые беговые кроссовки с нейлоновой пластиной"),
        ("SH-ALTRA-LP", "Altra Lone Peak 8", 13990, "Трейловые беговые кроссовки, широкий носок"),
        ("SH-ALTRA-ESC", "Altra Escalante 3", 11990, "Лёгкие беговые кроссовки с нулевым перепадом"),
        ("SH-UA-HOVR", "Under Armour HOVR Sonic 6", 9990, "Лёгкие беговые кроссовки с отдачей энергии"),
        ("SH-UA-CHRG", "Under Armour Charged Pursuit 3", 6990, "Бюджетные беговые кроссовки для зала и дорожки"),
        ("SH-NIKE-STR", "Nike Structure 25", 12990, "Стабильные беговые кроссовки с поддержкой свода"),
        ("SH-NIKE-WINF", "Nike Winflo 11", 7990, "Лёгкие беговые кроссовки Nike для начинающих"),
        ("SH-ADID-ADZ", "Adidas Adizero Adios 8", 14990, "Лёгкие соревновательные беговые кроссовки"),
        ("SH-ADID-4DF", "Adidas 4DFWD 3", 16990, "Инновационные лёгкие беговые кроссовки 4D-подошва"),
        ("SH-NB-MORE", "New Balance More v5", 15990, "Максимальная амортизация для длинного бега"),
        ("SH-NB-TEM", "New Balance Tempo v2", 13990, "Лёгкие беговые кроссовки для темповых тренировок"),
        ("SH-ASICS-META", "ASICS Metaspeed Sky+", 19990, "Профессиональные лёгкие беговые кроссовки для марафона"),
    ]
    return [_item(sku, name, "Кроссовки", price, desc) for sku, name, price, desc in items]


def _more_kitchen() -> list[dict]:
    items = [
        ("KA-IC-SAM", "Samsung RB33 Ice Maker", 79990, "Холодильник с льдогенератором для кухни, No Frost"),
        ("KA-IC-BOS", "Bosch KGN39", 64990, "Двухкамерный холодильник для кухни, VitaFresh"),
        ("KA-IC-HAI", "Haier HRF-541", 54990, "Холодильник для кухни, инверторный компрессор"),
        ("KA-OV-BOS", "Bosch HBA5780", 44990, "Встраиваемая духовка для кухни, пиролитическая очистка"),
        ("KA-OV-SAM", "Samsung NV75K", 39990, "Духовой шкаф для кухни, конвекция, 75 л"),
        ("KA-ST-RED", "Redmond MC-110", 3990, "Пароварка-мультиварка для кухни, 3 л"),
        ("KA-JU-BREV", "Breville Juice Fountain", 12990, "Соковыжималка для кухни, центробежная"),
        ("KA-JU-PHIL", "Philips HR1863", 5990, "Соковыжималка для кухни, широкая горловина"),
        ("KA-WP-BRA", "Braun MQ9037", 8990, "Кухонный комбайн-процессор с насадками"),
        ("KA-WP-KEN", "Kenwood FDP301", 9990, "Кухонный процессор для нарезки и теста"),
        ("KA-GR-KRUPS", "Krups EA8108", 29990, "Автоматическая кофемашина для кухни, зерновая"),
        ("KA-GR-MEL", "Melitta Caffeo Solo", 24990, "Кофемашина для кухни, эспрессо одним нажатием"),
        ("KA-SL-PHIL", "Philips Viva Collection", 4490, "Электрическая сковорода-сковорода для кухни"),
        ("KA-RK-TEF", "Tefal RK812", 8990, "Рисоварка для кухни, 10 чашек, пароварка"),
        ("KA-WM-LG", "LG F2V5GS0W", 49990, "Стиральная машина для кухни, 8.5 кг, пар"),
        ("KA-WM-SAM", "Samsung WW90", 44990, "Стиральная машина для кухни, EcoBubble, 9 кг"),
        ("KA-DW-SAM", "Samsung DW60", 39990, "Посудомоечная машина для кухни, 14 комплектов"),
        ("KA-DW-ELE", "Electrolux ESM48300", 34990, "Встраиваемая посудомойка для кухни, 60 см"),
        ("KA-HD-KIT", "KitchenAid Artisan Toaster", 8990, "Тостер для кухни, 4 слота, сталь"),
        ("KA-EG-SAL", "Salter EK4367", 3990, "Электрический гриль-вафельница для кухни"),
    ]
    return [_item(sku, name, "Кухонная техника", price, desc) for sku, name, price, desc in items]


def _tv_monitors() -> list[dict]:
    items = [
        ("TV-SAM-55", "Samsung QE55Q80C", 89990, "Телевизор 55\" QLED 4K для дома, Smart TV"),
        ("TV-LG-65", "LG OLED65C3", 149990, "OLED телевизор 65\" 4K, кинематографичная картинка"),
        ("TV-SONY-43", "Sony Bravia XR-43X90L", 69990, "Телевизор 43\" 4K, процессор Cognitive XR"),
        ("TV-XIAOMI-50", "Xiaomi TV A Pro 50", 39990, "Умный телевизор 50\" 4K, Google TV"),
        ("MON-DELL-U", "Dell UltraSharp U2723QE", 54990, "Монитор 27\" 4K IPS для работы, USB-C"),
        ("MON-ASUS-PRO", "ASUS ProArt PA278QV", 39990, "Монитор 27\" для дизайна, 100% sRGB"),
        ("MON-BENQ-MO", "BenQ MOBIUZ EX2710Q", 29990, "Игровой монитор 27\" 165 Гц QHD"),
        ("MON-SAM-OD", "Samsung Odyssey G7 32\"", 44990, "Изогнутый игровой монитор 240 Гц"),
        ("SPK-JBL-BAR", "JBL Bar 500", 49990, "Саундбар для телевизора, Dolby Atmos"),
        ("SPK-SONY-HT", "Sony HT-A3000", 59990, "Саундбар 3.1 для домашнего кинотеатра"),
        ("PRJ-XGIMI", "XGIMI Horizon Pro", 89990, "Проектор 4K для домашнего кино"),
        ("STR-CHROM", "Chromecast with Google TV 4K", 6990, "Стриминговый плеер для телевизора"),
    ]
    cats = {
        "TV": "Телевизоры", "MON": "Мониторы", "SPK": "Аудио",
        "PRJ": "Проекторы", "STR": "Стриминг",
    }
    result = []
    for sku, name, price, desc in items:
        prefix = sku.split("-")[0]
        cat = cats.get(prefix, "Электроника")
        result.append(_item(sku, name, cat, price, desc))
    return result


def _gaming() -> list[dict]:
    items = [
        ("GM-PS5-SLIM", "PlayStation 5 Slim", 54990, "Игровая консоль Sony, SSD 1 ТБ, 4K"),
        ("GM-XBOX-X", "Xbox Series X", 54990, "Игровая консоль Microsoft, 4K 120 fps"),
        ("GM-XBOX-S", "Xbox Series S", 29990, "Компактная игровая консоль, Game Pass"),
        ("GM-SWITCH-OLED", "Nintendo Switch OLED", 34990, "Портативная игровая консоль с OLED экраном"),
        ("GM-STEAM-DECK", "Steam Deck OLED 512GB", 64990, "Портативный игровой ПК от Valve"),
        ("GM-RAZER-KB", "Razer BlackWidow V4", 14990, "Механическая игровая клавиатура RGB"),
        ("GM-CORSAIR-HS", "Corsair HS80 RGB", 9990, "Игровая гарнитура с объёмным звуком"),
        ("GM-STEEL-HD", "SteelSeries Arctis Nova Pro", 24990, "Премиальная игровая гарнитура"),
        ("GM-LOGI-WHEEL", "Logitech G923", 34990, "Игровой руль с Force Feedback для симуляторов"),
        ("GM-PS5-VR2", "PlayStation VR2", 44990, "VR-шлем для PlayStation 5"),
    ]
    console_skus = {
        "GM-PS5-SLIM", "GM-XBOX-X", "GM-XBOX-S",
        "GM-SWITCH-OLED", "GM-STEAM-DECK", "GM-PS5-VR2",
    }
    result = []
    for sku, name, price, desc in items:
        cat = "Игровые консоли" if sku in console_skus else "Игровые аксессуары"
        result.append(_item(sku, name, cat, price, desc))
    return result


def _beauty_care() -> list[dict]:
    items = [
        ("BC-ORAL-B", "Oral-B iO Series 9", 19990, "Электрическая зубная щётка, Bluetooth"),
        ("BC-PHILIPS-SON", "Philips Sonicare 9900", 24990, "Звуковая зубная щётка премиум"),
        ("BC-BRAUN-S9", "Braun Series 9 Pro", 29990, "Электробритва для мужчин, 5 насадок"),
        ("BC-PANASONIC-ES", "Panasonic ES-LV67", 19990, "Электробритва, 5 лезвий, влажная сухая"),
        ("BC-DYSON-HS", "Dyson Airwrap Complete", 54990, "Стайлер для укладки волос, много насадок"),
        ("BC-GHD-PLAT", "GHD Platinum+ Styler", 24990, "Выпрямитель для волос, умный контроль температуры"),
        ("BC-FOREO-LUNA", "Foreo Luna 4", 14990, "Умная щётка для умывания лица"),
        ("BC-PHILIPS-LUME", "Philips Lumea Prestige", 39990, "Фотоэпилятор IPL для дома"),
        ("BC-BRAUN-IPL", "Braun Silk-expert Pro 5", 34990, "IPL-эпилятор для длительного удаления волос"),
        ("BC-WATERPIK", "Waterpik WP-660", 8990, "Ирригатор для полости рта, 10 режимов"),
    ]
    return [_item(sku, name, "Уход и красота", price, desc) for sku, name, price, desc in items]


def _smart_home() -> list[dict]:
    items = [
        ("SH-YANDEX-ST", "Яндекс Станция Макс", 19990, "Умная колонка с Алисой, Zigbee хаб"),
        ("SH-YANDEX-MI", "Яндекс Станция Мини 3", 6990, "Компактная умная колонка с голосовым помощником"),
        ("SH-GOOGLE-NEST", "Google Nest Hub 2", 9990, "Умный дисплей с Google Assistant"),
        ("SH-AMAZON-ECHO", "Amazon Echo Show 10", 14990, "Умный экран с Alexa, вращающийся дисплей"),
        ("SH-AQARA-HUB", "Aqara Hub M2", 4990, "Умный хаб для датчиков Zigbee"),
        ("SH-AQARA-DOOR", "Aqara Door Sensor", 1990, "Датчик открытия двери для умного дома"),
        ("SH-RING-CAM", "Ring Indoor Cam 2", 4990, "Умная камера наблюдения для дома"),
        ("SH-EZVIZ-CAM", "Ezviz C6N", 2990, "Wi-Fi камера видеонаблюдения 360°"),
        ("SH-ROBO-VAC", "Roborock Q Revo", 69990, "Робот-пылесос с базой для умного дома"),
        ("SH-DYSON-HUM", "Dyson Purifier Humidify", 69990, "Увлажнитель и очиститель воздуха для дома"),
        ("SH-XIAOMI-PUR", "Xiaomi Air Purifier 4 Pro", 19990, "Очиститель воздуха HEPA для квартиры"),
        ("SH-TADO-THERM", "tado° Smart Thermostat", 12990, "Умный термостат для отопления"),
    ]
    return [_item(sku, name, "Умный дом", price, desc) for sku, name, price, desc in items]


def _outdoor_camping() -> list[dict]:
    items = [
        ("OD-TENT-QUE", "Quechua Arpenaz 3", 8990, "Палатка 3-местная для кемпинга"),
        ("OD-SLEEP-CF", "Coleman Comfort Sleeping Bag", 4990, "Спальный мешок для походов до -5°C"),
        ("OD-BACK-DEU", "Deuter Aircontact 65+10", 24990, "Туристический рюкзак 65 л для походов"),
        ("OD-STOVE-JET", "Jetboil Flash", 9990, "Горелка для походной кухни, быстрое кипячение"),
        ("OD-FLASK-ST", "Stanley Classic 1.9L", 4990, "Термос для походов и кемпинга"),
        ("OD-KNIFE-MO", "Morakniv Companion", 1990, "Походный нож из нержавеющей стали"),
        ("OD-LAMP-PET", "Petzl Actik Core", 4990, "Налобный фонарь для ночных походов"),
        ("OD-CHAIR-HEL", "Helinox Chair One", 7990, "Складной походный стул, 1.2 кг"),
        ("OD-MAT-THER", "Therm-a-Rest NeoAir", 12990, "Надувной туристический коврик для сна"),
        ("OD-COOL-YET", "YETI Roadie 24", 24990, "Переносной холодильник для пикника"),
    ]
    return [_item(sku, name, "Туризм и кемпинг", price, desc) for sku, name, price, desc in items]


def _office() -> list[dict]:
    items = [
        ("OF-CHAIR-HM", "Herman Miller Aeron", 149990, "Эргономичное офисное кресло для работы"),
        ("OF-CHAIR-SEC", "Secretlab Titan Evo", 49990, "Игровое и офисное кресло, регулировка поясницы"),
        ("OF-DESK-FLEX", "Flexispot E7", 39990, "Стол с электроприводом, сидя-стоя"),
        ("OF-LAMP-BEN", "BenQ ScreenBar Halo", 14990, "Настольная лампа для монитора без бликов"),
        ("OF-PRINT-HP", "HP LaserJet Pro M404", 24990, "Лазерный принтер для офиса, двусторонняя печать"),
        ("OF-SCAN-FUJ", "Fujitsu ScanSnap iX1600", 39990, "Сканер документов для офиса, Wi-Fi"),
        ("OF-WEB-LOGI", "Logitech Brio 4K", 14990, "Веб-камера 4K для видеоконференций"),
        ("OF-MIC-BLUE", "Blue Yeti X", 12990, "USB-микрофон для подкастов и стримов"),
        ("OF-HUB-CAL", "CalDigit TS4", 29990, "Thunderbolt 4 док-станция для ноутбука"),
        ("OF-PAD-VAR", "Varidesk Standing Mat", 4990, "Коврик антиусталости для стоячего стола"),
    ]
    return [_item(sku, name, "Офис", price, desc) for sku, name, price, desc in items]


def _kids_toys() -> list[dict]:
    items = [
        ("KD-LEGO-STAR", "LEGO Star Wars X-Wing", 9990, "Конструктор LEGO 474 детали"),
        ("KD-LEGO-FRI", "LEGO Friends Heartlake", 4990, "Конструктор LEGO для детей 7+"),
        ("KD-BARBIE", "Barbie Dreamhouse", 14990, "Кукольный домик Barbie с аксессуарами"),
        ("KD-HOTWHEELS", "Hot Wheels Ultimate Garage", 8990, "Автотрек Hot Wheels, 3 этажа"),
        ("KD-NERF-ELITE", "Nerf Elite 2.0 Commander", 3990, "Бластер Nerf, 12 дротиков"),
        ("KD-PLAYDOH", "Play-Doh Kitchen Creations", 2990, "Набор пластилина Play-Doh для лепки"),
        ("KD-SCOOTER", "Micro Sprite Scooter", 9990, "Детский самокат, складной, до 100 кг"),
        ("KD-BIKE-WOOM", "Woom 4 Bike 20\"", 39990, "Детский велосипед 20 дюймов, лёгкий алюминий"),
    ]
    return [_item(sku, name, "Детские товары", price, desc) for sku, name, price, desc in items]


def _home_misc() -> list[dict]:
    items = [
        ("JK-NORTH-F", "The North Face Thermoball", 18990, "Лёгкая утеплённая куртка"),
        ("BP-OSPREY", "Osprey Daylite Plus 20L", 7990, "Городской рюкзак 20 л"),
        ("BP-HERSCHEL", "Herschel Little America", 9990, "Рюкзак 25 л"),
        ("GL-RAY-BAN", "Ray-Ban Aviator Classic", 12990, "Солнцезащитные очки"),
        ("LT-PHILIPS-HUE", "Philips Hue White and Color", 4990, "Умная RGB лампа"),
        ("LT-XIAOMI", "Xiaomi Smart LED Bulb", 990, "Умная лампа Wi-Fi"),
        ("AR-DYSON-PUR", "Dyson Purifier Cool TP07", 54990, "Очиститель воздуха HEPA"),
        ("HM-NEST", "Google Nest Thermostat", 14990, "Умный термостат"),
        ("GR-WEBER", "Weber Spirit II E-310", 69990, "Газовый гриль для дачи"),
        ("CF-LAVAZZA", "Lavazza Qualità Oro 250 г", 890, "Молотый кофе для кофемашины"),
        ("CF-ILLY", "illy Classico зерно 1 кг", 3490, "Кофе в зёрнах для эспрессо"),
        ("TE-TWININGS", "Twinings English Breakfast", 590, "Чёрный чай 100 пакетиков"),
        ("KN-ZWILLING", "Zwilling Chef Knife 20 см", 8990, "Поварской нож для кухни"),
        ("PN-CASTIRON", "Lodge Cast Iron Skillet", 4990, "Чугунная сковорода"),
        ("TO-LEGO-TECH", "LEGO Technic Porsche 911", 14990, "Конструктор 1458 деталей"),
        ("BK-HP-1", "Гарри Поттер и философский камень", 690, "Книга на русском"),
        ("GM-PS5-DUAL", "Sony DualSense Edge", 17990, "Геймпад PlayStation 5"),
        ("GM-LOGITECH", "Logitech G Pro X Superlight 2", 14990, "Игровая мышь 60 г"),
        ("CAM-CANON-R50", "Canon EOS R50", 64990, "Беззеркальная камера 4K"),
        ("PEN-IPAD", "Apple Pencil Pro", 12990, "Стилус для iPad"),
        ("CH-ANERGIA", "Anker PowerCore 20000", 4990, "Powerbank 20000 mAh"),
        ("MON-LG-27", "LG UltraGear 27GP850", 34990, "Игровой монитор 27\" 165 Гц"),
        ("KEY-KEYCHRON", "Keychron K2 V2", 8990, "Механическая клавиатура Bluetooth"),
        ("SSD-SAMSUNG", "Samsung 990 Pro 2TB", 14990, "NVMe SSD PCIe 4.0"),
        ("WT-APPLE-S9", "Apple Watch Series 9", 39990, "Умные часы GPS, пульс"),
        ("WT-GAL-W6", "Samsung Galaxy Watch 6", 24990, "Умные часы Wear OS"),
        ("DR-DJI-MINI", "DJI Mini 4 Pro", 89990, "Дрон с 4K камерой"),
    ]
    cats = {
        "JK": "Куртки", "BP": "Рюкзаки", "GL": "Очки", "LT": "Освещение",
        "AR": "Климат", "HM": "Умный дом", "GR": "Грили", "CF": "Кофе",
        "TE": "Чай", "KN": "Кухонные принадлежности", "PN": "Посуда",
        "TO": "Конструкторы", "BK": "Книги", "GM": "Игровые аксессуары",
        "CAM": "Фото", "PEN": "Аксессуары", "CH": "Powerbank", "MON": "Мониторы",
        "KEY": "Клавиатуры", "SSD": "Накопители", "WT": "Умные часы", "DR": "Дроны",
    }
    result = []
    for sku, name, price, desc in items:
        prefix = sku.split("-")[0]
        cat = cats.get(prefix, "Разное")
        result.append(_item(sku, name, cat, price, desc))
    return result


def build_catalog() -> list[dict]:
    raw = (
        _running_shoes()
        + _more_running_shoes()
        + _kitchen_appliances()
        + _more_kitchen()
        + _vacuums_cleaning()
        + _headphones_audio()
        + _phones_laptops()
        + _hair_dryers()
        + _sport_other()
        + _tv_monitors()
        + _gaming()
        + _beauty_care()
        + _smart_home()
        + _outdoor_camping()
        + _office()
        + _kids_toys()
        + _home_misc()
    )
    seen: set[str] = set()
    items: list[dict] = []
    for p in raw:
        if p["sku"] in seen:
            continue
        seen.add(p["sku"])
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
