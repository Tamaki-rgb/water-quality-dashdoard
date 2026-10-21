"""
src/load.py
------------
Генерация синтетического датасета (если файла ещё нет), загрузка данных,
очистка выбросов, заполнение пропусков и расчёт индекса загрязнения (pollution_index).

Почему синтетика: открытых почасовых/подневных гидрохимических рядов именно
по реке Сайрамсу (Сайрамский р-н, ШКО) в открытом доступе нет — Казгидромет
публикует бюллетени в PDF, а data.egov.kz / stat.gov.kz дают только
агрегированную годовую статистику по регионам. Поэтому генерируем физически
правдоподобный ряд для горной реки предгорий Таласского Алатау, а нормативы
(ПДК, оптимальные диапазоны pH/O2 и т.д.) берём из реальных методик ВОЗ и СанПиН РК.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "water.csv")

# Колонки, которые являются реальными измерениями (к ним применяем чистку/интерполяцию)
MEASURED_COLS = [
    "temp_water", "ph", "do_mg_l", "nitrates_mg_l",
    "turbidity_ntu", "discharge_m3s", "bod5_mg_l",
]


def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    """
    Генерирует синтетический датасет качества воды для 5 станций вдоль реки
    Сайрамсу (от истока в горах до выхода к предгорьям Шымкента),
    2019-01-01 .. 2024-12-31, шаг 3 дня.
    """
    rng = np.random.default_rng(seed)

    # Координаты условные (иллюстративные), но геометрически последовательные:
    # река течёт из гор (юго-восток) в сторону Шымкента (северо-запад).
    stations = [
        dict(station_id="ST01", station_name="Исток (ущелье Сайрам)",     lat=42.15, lon=70.10, distance_km=0),
        dict(station_id="ST02", station_name="Выход из ущелья",           lat=42.20, lon=70.00, distance_km=8),
        dict(station_id="ST03", station_name="с. Сайрам",                 lat=42.28, lon=69.85, distance_km=16),
        dict(station_id="ST04", station_name="с. Карабулак",              lat=42.32, lon=69.75, distance_km=24),
        dict(station_id="ST05", station_name="Устье / граница Шымкента",  lat=42.36, lon=69.65, distance_km=33),
    ]

    dates = pd.date_range("2019-01-01", "2024-12-31", freq="3D")
    rows = []

    for st_info in stations:
        frac = st_info["distance_km"] / 33.0  # доля пути от истока (0..1) — прокси антропогенной нагрузки
        for d in dates:
            doy = d.dayofyear

            # --- сезонные компоненты ---
            season_temp = 10 + 9 * np.sin(2 * np.pi * (doy - 80) / 365)   # годовой ход температуры воздуха/воды
            season_snowmelt = np.exp(-((doy - 125) / 30) ** 2)            # пик снеготаяния ~начало мая
            season_summer_heat = np.exp(-((doy - 200) / 40) ** 2)         # пик летней жары ~июль

            # --- температура воды: выше по течению холоднее (высота гор), плюс общая сезонность ---
            temp_water = season_temp - 3 * (1 - frac) + rng.normal(0, 0.8)
            temp_water = np.clip(temp_water, 0, 26)

            # --- расход воды: резкий пик при снеготаянии, рост вниз по течению за счёт притоков ---
            discharge = (0.6 + 1.2 * frac) * (1 + 2.5 * season_snowmelt) + rng.normal(0, 0.15)
            discharge = np.clip(discharge, 0.2, 9)

            # --- мутность: коррелирует с паводком/снеготаянием, растёт ниже по течению ---
            turbidity = 3 + 60 * season_snowmelt + 8 * frac + rng.normal(0, 3)
            turbidity = np.clip(turbidity, 1, 130)

            # --- растворённый кислород: падает летом (жара) и ниже по течению (нагрузка) ---
            do_mg_l = 10.5 - 3.5 * season_summer_heat - 2.0 * frac + rng.normal(0, 0.4)
            do_mg_l = np.clip(do_mg_l, 3.5, 11.5)

            # --- pH: относительно стабилен, слегка смещается антропогенно вниз по течению ---
            ph = 7.8 - 0.5 * frac + rng.normal(0, 0.15)
            ph = np.clip(ph, 6.5, 8.8)

            # --- нитраты: рост ниже населённых пунктов + смыв удобрений весной/летом ---
            nitrates = 0.8 + 10 * frac + 3 * season_snowmelt + rng.normal(0, 0.6)
            nitrates = np.clip(nitrates, 0.2, 18)

            # --- БПК5: антропогенная нагрузка, растёт вниз по течению и летом (застой) ---
            bod5 = 0.5 + 4 * frac + 1.0 * season_summer_heat + rng.normal(0, 0.3)
            bod5 = np.clip(bod5, 0.2, 7)

            rows.append(dict(
                date=d,
                station_id=st_info["station_id"],
                station_name=st_info["station_name"],
                lat=st_info["lat"], lon=st_info["lon"],
                distance_km=st_info["distance_km"],
                temp_water=round(float(temp_water), 2),
                ph=round(float(ph), 2),
                do_mg_l=round(float(do_mg_l), 2),
                nitrates_mg_l=round(float(nitrates), 2),
                turbidity_ntu=round(float(turbidity), 2),
                discharge_m3s=round(float(discharge), 2),
                bod5_mg_l=round(float(bod5), 2),
            ))

    df = pd.DataFrame(rows)

    # --- редкие выбросы: аварийные сбросы / ливни (~1.5% строк) ---
    outlier_idx = rng.choice(df.index, size=int(0.015 * len(df)), replace=False)
    df.loc[outlier_idx, "turbidity_ntu"] *= rng.uniform(2, 4, size=len(outlier_idx))
    df.loc[outlier_idx, "nitrates_mg_l"] *= rng.uniform(1.5, 3, size=len(outlier_idx))

    # --- пропуски в данных: 5-8% случайных значений по каждой измеряемой колонке ---
    for col in MEASURED_COLS:
        share = rng.uniform(0.05, 0.08)
        na_idx = rng.choice(df.index, size=int(share * len(df)), replace=False)
        df.loc[na_idx, col] = np.nan

    return df


def _remove_outliers_iqr(df: pd.DataFrame, cols, k: float = 3.0) -> pd.DataFrame:
    """
    Убирает экстремальные выбросы методом межквартильного размаха (IQR),
    отдельно по каждой станции — т.к. "нормальные" диапазоны отличаются
    в зависимости от точки на реке. Выброс превращается в NaN и позже
    восстанавливается интерполяцией.
    """
    df = df.copy()
    for station, group in df.groupby("station_id"):
        for col in cols:
            q1, q3 = group[col].quantile(0.25), group[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - k * iqr, q3 + k * iqr
            mask = (df["station_id"] == station) & ((df[col] < lower) | (df[col] > upper))
            df.loc[mask, col] = np.nan
    return df


def _fill_missing(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Заполняет пропуски линейной интерполяцией внутри каждой станции по времени."""
    df = df.sort_values(["station_id", "date"]).copy()
    for station in df["station_id"].unique():
        m = df["station_id"] == station
        df.loc[m, cols] = df.loc[m, cols].interpolate(method="linear", limit_direction="both")
    return df


def compute_pollution_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Упрощённый индекс загрязнения (аналог Water Quality Index), диапазон 0..1:
      0   = отличное качество воды
      1   = критическое загрязнение
    Каждый параметр приводится к суб-индексу 0..1 (1 = максимально плохо),
    затем берётся взвешенное среднее по важности параметра для экосистемы.
    """
    df = df.copy()

    def worse_higher(x, good, bad):
        """good -> 0, bad -> 1 (чем выше значение, тем хуже: нитраты, мутность, БПК)."""
        return np.clip((x - good) / (bad - good), 0, 1)

    def worse_lower(x, good, bad):
        """good (высокое) -> 0, bad (низкое) -> 1 (чем ниже значение, тем хуже: кислород)."""
        return np.clip((good - x) / (good - bad), 0, 1)

    sub_ph = np.clip(np.abs(df["ph"] - 7.5) / 1.5, 0, 1)             # отклонение от нейтрального pH
    sub_do = worse_lower(df["do_mg_l"], good=9.0, bad=4.0)           # нехватка кислорода
    sub_no3 = worse_higher(df["nitrates_mg_l"], good=2.0, bad=14.0)  # избыток нитратов
    sub_turb = worse_higher(df["turbidity_ntu"], good=5.0, bad=90.0) # мутность
    sub_bod = worse_higher(df["bod5_mg_l"], good=1.0, bad=5.5)       # органическое загрязнение

    weights = {"ph": 0.15, "do": 0.25, "no3": 0.25, "turb": 0.15, "bod": 0.20}
    df["pollution_index"] = (
        weights["ph"] * sub_ph + weights["do"] * sub_do +
        weights["no3"] * sub_no3 + weights["turb"] * sub_turb +
        weights["bod"] * sub_bod
    ).round(3)
    return df


@st.cache_data(show_spinner="Загрузка и очистка данных о качестве воды...")
def load_data() -> pd.DataFrame:
    """
    Главная точка входа для остальных модулей:
    читает data/water.csv (генерирует при первом запуске, если файла нет),
    чистит выбросы, восстанавливает пропуски, считает индекс загрязнения
    и добавляет календарные признаки (год/месяц/сезон).
    """
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    else:
        df = generate_synthetic_data()
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    df = _remove_outliers_iqr(df, MEASURED_COLS, k=3.0)
    df = _fill_missing(df, MEASURED_COLS)
    df = compute_pollution_index(df)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    season_map = {12: "Зима", 1: "Зима", 2: "Зима",
                  3: "Весна", 4: "Весна", 5: "Весна",
                  6: "Лето", 7: "Лето", 8: "Лето",
                  9: "Осень", 10: "Осень", 11: "Осень"}
    df["season"] = df["month"].map(season_map)

    return df


if __name__ == "__main__":
    # Позволяет пересобрать датасет вручную: python src/load.py
    generated = generate_synthetic_data()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    generated.to_csv(DATA_PATH, index=False)
    print(f"Сгенерировано {len(generated)} строк -> {DATA_PATH}")
