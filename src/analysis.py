"""
src/analysis.py
-----------------
Функции аналитики для вкладок "Обзор" и "Аналитика":
динамика во времени, сезонность, корреляции, распределения,
классификация индекса качества воды (WQI).
"""

import numpy as np
import pandas as pd
import plotly.express as px

# Единый визуальный стиль для всех графиков дашборда
PLOT_TEMPLATE = "plotly_dark"
ACCENT = "#00b4d8"

# Человекочитаемые подписи для параметров (используются в селекторах и заголовках)
PARAM_LABELS = {
    "temp_water": "Температура воды, °C",
    "ph": "pH",
    "do_mg_l": "Растворённый кислород, мг/л",
    "nitrates_mg_l": "Нитраты, мг/л",
    "turbidity_ntu": "Мутность, NTU",
    "discharge_m3s": "Расход воды, м³/с",
    "bod5_mg_l": "БПК5, мг/л",
    "pollution_index": "Индекс загрязнения (0-1)",
}


def classify_wqi(value: float) -> str:
    """Категория качества воды по индексу загрязнения (0..1)."""
    if value < 0.2:
        return "Отличное"
    elif value < 0.4:
        return "Хорошее"
    elif value < 0.6:
        return "Удовлетворительное"
    elif value < 0.8:
        return "Плохое"
    return "Критическое"


WQI_COLORS = {
    "Отличное": "#2ecc71",
    "Хорошее": "#a3e635",
    "Удовлетворительное": "#facc15",
    "Плохое": "#fb923c",
    "Критическое": "#ef4444",
}


def fig_timeseries(df: pd.DataFrame, parameter: str, stations=None, rolling: int = 7):
    """Динамика параметра во времени со скользящим средним по каждой станции."""
    data = df.copy()
    if stations:
        data = data[data["station_id"].isin(stations)]
    data = data.sort_values("date")
    data["rolling"] = data.groupby("station_id")[parameter].transform(
        lambda s: s.rolling(rolling, min_periods=1).mean()
    )
    label = PARAM_LABELS.get(parameter, parameter)
    fig = px.line(
        data, x="date", y="rolling", color="station_name",
        template=PLOT_TEMPLATE,
        labels={"rolling": label, "date": "Дата", "station_name": "Станция"},
        title=f"Динамика: {label} (скользящее среднее, {rolling} дн.)",
    )
    fig.update_traces(mode="lines", line=dict(width=2))
    fig.update_layout(hovermode="x unified", legend_title_text="Станция", height=460)
    return fig


def fig_seasonality(df: pd.DataFrame, parameter: str, stations=None):
    """Сезонность параметра по месяцам (boxplot, окрашен по сезону)."""
    data = df.copy()
    if stations:
        data = data[data["station_id"].isin(stations)]
    month_names = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
    data["month_name"] = data["month"].apply(lambda m: month_names[m - 1])
    label = PARAM_LABELS.get(parameter, parameter)
    fig = px.box(
        data, x="month_name", y=parameter, color="season",
        category_orders={"month_name": month_names},
        template=PLOT_TEMPLATE,
        title=f"Сезонность: {label} по месяцам",
        labels={"month_name": "Месяц", parameter: label},
    )
    fig.update_layout(height=440, legend_title_text="Сезон")
    return fig


def fig_correlation_heatmap(df: pd.DataFrame, cols):
    """Тепловая карта корреляций между параметрами качества воды."""
    corr = df[cols].corr(numeric_only=True).round(2)
    labels = [PARAM_LABELS.get(c, c) for c in cols]
    fig = px.imshow(
        corr.values, x=labels, y=labels,
        text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        template=PLOT_TEMPLATE, title="Корреляция между показателями качества воды",
        aspect="auto",
    )
    fig.update_layout(height=520)
    return fig


def fig_distribution(df: pd.DataFrame, parameter: str, stations=None):
    """Распределение параметра по станциям (violin + встроенный boxplot)."""
    data = df.copy()
    if stations:
        data = data[data["station_id"].isin(stations)]
    label = PARAM_LABELS.get(parameter, parameter)
    fig = px.violin(
        data, x="station_name", y=parameter, color="station_name", box=True, points=False,
        template=PLOT_TEMPLATE, title=f"Распределение: {label} по станциям",
        labels={"station_name": "Станция", parameter: label},
    )
    fig.update_layout(showlegend=False, height=460)
    return fig


def wqi_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Сводная таблица: средний индекс загрязнения и категория WQI по станциям."""
    summary = df.groupby(["station_name", "distance_km"])["pollution_index"].mean().reset_index()
    summary["Категория"] = summary["pollution_index"].apply(classify_wqi)
    summary = summary.sort_values("distance_km")
    summary.columns = ["Станция", "Расстояние от истока, км", "Средний индекс загрязнения", "Категория"]
    summary["Средний индекс загрязнения"] = summary["Средний индекс загрязнения"].round(3)
    return summary


def peak_pollution_period(df: pd.DataFrame) -> dict:
    """Находит станцию и период (месяц) с наибольшим средним загрязнением — для вкладки Выводы."""
    worst_station = df.groupby("station_name")["pollution_index"].mean().idxmax()
    worst_month_num = df.groupby("month")["pollution_index"].mean().idxmax()
    month_names = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
                   "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"]
    return {
        "worst_station": worst_station,
        "worst_month": month_names[worst_month_num - 1],
        "worst_value": round(df.groupby("station_name")["pollution_index"].mean().max(), 3),
    }
