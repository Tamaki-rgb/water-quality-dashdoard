"""
app.py
-------
Главный файл дашборда "Качество воды: аналитика и прогноз".
Река Сайрамсу, Сайрамский район, Шымкент, Казахстан.

Запуск: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.load import load_data, MEASURED_COLS
from src import analysis
from src import model as model_module
from src import forecast as forecast_module

# ========================= НАСТРОЙКА СТРАНИЦЫ =========================
st.set_page_config(
    page_title="Качество воды — р. Сайрамсу",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Немного кастомного CSS для более "живого" вида карточек и вкладок
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 18px 22px; border-radius: 16px; color: white;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
    }
    div[data-testid="stMetric"] {
        background-color: rgba(38, 39, 48, 0.55);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 12px 16px; border-radius: 14px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 16px; padding: 8px 18px; }
    h1, h2, h3 { letter-spacing: 0.3px; }
</style>
""", unsafe_allow_html=True)

st.title("💧 Качество воды: аналитика и прогноз")
st.caption("Река Сайрамсу · Сайрамский район · Шымкентская область, Казахстан "
           "(синтетический датасет, физически откалиброванный под горную реку предгорий Таласского Алатау)")

# ========================= ЗАГРУЗКА ДАННЫХ =========================
df = load_data()

# ========================= САЙДБАР: ГЛОБАЛЬНЫЕ ФИЛЬТРЫ =========================
st.sidebar.header("⚙️ Фильтры")

all_stations = df[["station_id", "station_name", "distance_km"]].drop_duplicates().sort_values("distance_km")
station_options = all_stations["station_id"].tolist()
station_labels = dict(zip(all_stations["station_id"], all_stations["station_name"]))

selected_stations = st.sidebar.multiselect(
    "Станции мониторинга",
    options=station_options,
    default=station_options,
    format_func=lambda sid: station_labels[sid],
)

date_min, date_max = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input(
    "Период", value=(date_min, date_max), min_value=date_min, max_value=date_max,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_min, date_max

param_choice = st.sidebar.selectbox(
    "Параметр для анализа",
    options=list(analysis.PARAM_LABELS.keys()),
    format_func=lambda p: analysis.PARAM_LABELS[p],
    index=list(analysis.PARAM_LABELS.keys()).index("pollution_index"),
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "📌 Данные синтетические (реальных открытых рядов по р. Сайрамсу нет — "
    "см. вкладку «Выводы» → «Источники»). Нормативы и логика генерации "
    "откалиброваны по методикам ВОЗ и СанПиН РК."
)

# Применяем фильтры
if not selected_stations:
    st.warning("Выберите хотя бы одну станцию мониторинга в боковой панели.")
    st.stop()

mask = (
    df["station_id"].isin(selected_stations)
    & (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
)
fdf = df.loc[mask].copy()

# ========================= ВКЛАДКИ =========================
tab_overview, tab_analytics, tab_map, tab_math, tab_forecast, tab_conclusions = st.tabs(
    ["🏠 Обзор", "📊 Аналитика", "🗺️ Карта", "🧮 Мат.модель", "🔮 Прогноз", "📝 Выводы"]
)

# ------------------------------------------------------------------
# ВКЛАДКА 1: ОБЗОР
# ------------------------------------------------------------------
with tab_overview:
    st.subheader("Общая картина по выбранному периоду и станциям")

    col1, col2, col3, col4 = st.columns(4)
    avg_pollution = fdf["pollution_index"].mean()
    worst_row = fdf.loc[fdf["pollution_index"].idxmax()]
    best_row = fdf.loc[fdf["pollution_index"].idxmin()]

    col1.metric("Средний индекс загрязнения", f"{avg_pollution:.3f}",
                help="0 = отличное качество, 1 = критическое загрязнение")
    col2.metric("Категория (в среднем)", analysis.classify_wqi(avg_pollution))
    col3.metric("Худший замер", f"{worst_row['pollution_index']:.3f}",
                help=f"{worst_row['station_name']}, {worst_row['date'].date()}")
    col4.metric("Станций в выборке", f"{fdf['station_id'].nunique()} / {df['station_id'].nunique()}")

    st.markdown("### 📈 Динамика индекса загрязнения")
    st.plotly_chart(
        analysis.fig_timeseries(fdf, "pollution_index", selected_stations, rolling=7),
        width="stretch", key="overview_timeseries",
    )

    st.markdown("### 🏆 Рейтинг станций по качеству воды")
    summary_table = analysis.wqi_summary_table(fdf)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.dataframe(
            summary_table.style.background_gradient(
                subset=["Средний индекс загрязнения"], cmap="RdYlGn_r"
            ),
            width="stretch", hide_index=True,
        )
    with c2:
        fig_bar = px.bar(
            summary_table, x="Средний индекс загрязнения", y="Станция",
            color="Категория", orientation="h",
            color_discrete_map=analysis.WQI_COLORS,
            template=analysis.PLOT_TEMPLATE, title="Загрязнение по станциям",
        )
        fig_bar.update_layout(height=380, showlegend=True)
        st.plotly_chart(fig_bar, width="stretch", key="overview_bar")

    with st.expander("ℹ️ Как считается индекс загрязнения (WQI)?"):
        st.markdown(
            "Индекс — взвешенное среднее пяти суб-индексов (0..1, где 1 = плохо): "
            "**pH** (отклонение от нейтрального), **растворённый кислород** (чем меньше — тем хуже), "
            "**нитраты**, **мутность**, **БПК5**. Веса: O₂ и NO₃⁻ — по 25%, БПК5 — 20%, pH и мутность — по 15%."
        )

# ------------------------------------------------------------------
# ВКЛАДКА 2: АНАЛИТИКА
# ------------------------------------------------------------------
with tab_analytics:
    st.subheader(f"Углублённая аналитика: {analysis.PARAM_LABELS[param_choice]}")

    st.plotly_chart(
        analysis.fig_timeseries(fdf, param_choice, selected_stations, rolling=7),
        width="stretch", key="analytics_timeseries",
    )

    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(analysis.fig_seasonality(fdf, param_choice, selected_stations),
                         width="stretch", key="analytics_seasonality")
    with colB:
        st.plotly_chart(analysis.fig_distribution(fdf, param_choice, selected_stations),
                         width="stretch", key="analytics_distribution")

    st.markdown("### 🔗 Матрица корреляций")
    corr_cols = MEASURED_COLS + ["pollution_index"]
    st.plotly_chart(analysis.fig_correlation_heatmap(fdf, corr_cols),
                     width="stretch", key="analytics_heatmap")
    st.caption(
        "Обратите внимание на связь БПК5 и нитратов с индексом загрязнения — "
        "это ключевые индикаторы антропогенной нагрузки на реку."
    )

# ------------------------------------------------------------------
# ВКЛАДКА 3: КАРТА
# ------------------------------------------------------------------
with tab_map:
    st.subheader("Карта станций мониторинга")
    st.caption("Цвет точки = средний индекс загрязнения за выбранный период. "
               "Кликните на точку, чтобы увидеть карточку показателей.")

    map_data = fdf.groupby(
        ["station_id", "station_name", "lat", "lon", "distance_km"], as_index=False
    ).agg(
        pollution_index=("pollution_index", "mean"),
        ph=("ph", "mean"), do_mg_l=("do_mg_l", "mean"),
        nitrates_mg_l=("nitrates_mg_l", "mean"), turbidity_ntu=("turbidity_ntu", "mean"),
        temp_water=("temp_water", "mean"), discharge_m3s=("discharge_m3s", "mean"),
        bod5_mg_l=("bod5_mg_l", "mean"),
    )
    map_data["Категория"] = map_data["pollution_index"].apply(analysis.classify_wqi)

    fig_map = px.scatter_map(
        map_data, lat="lat", lon="lon",
        color="pollution_index", size=[28] * len(map_data),
        color_continuous_scale="RdYlGn_r", range_color=(0, 0.8),
        hover_name="station_name",
        hover_data={"lat": False, "lon": False, "pollution_index": ":.3f", "distance_km": True},
        zoom=9.3, center={"lat": 42.30, "lon": 69.90},
        map_style="carto-darkmatter",
        template=analysis.PLOT_TEMPLATE, height=560,
    )
    fig_map.update_traces(marker=dict(sizemin=14))
    fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_colorbar_title="Загрязнение")

    map_event = st.plotly_chart(
        fig_map, width="stretch", on_select="rerun", key="water_map", selection_mode="points"
    )

    # Определяем выбранную (кликнутую) станцию: либо через клик на карте, либо через selectbox-фолбэк
    clicked_idx = None
    try:
        points = map_event["selection"]["points"]
        if points:
            clicked_idx = points[0]["point_index"]
    except Exception:
        clicked_idx = None

    if clicked_idx is None:
        picked_name = st.selectbox(
            "Или выберите станцию вручную:",
            options=map_data["station_name"].tolist(),
        )
        card_row = map_data[map_data["station_name"] == picked_name].iloc[0]
    else:
        card_row = map_data.iloc[clicked_idx]

    st.markdown("#### 📋 Карточка показателей станции")
    st.markdown(f"**{card_row['station_name']}** · {card_row['distance_km']:.0f} км от истока")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Индекс загрязнения", f"{card_row['pollution_index']:.3f}", card_row["Категория"])
    m2.metric("pH", f"{card_row['ph']:.2f}")
    m3.metric("Растворённый O₂", f"{card_row['do_mg_l']:.2f} мг/л")
    m4.metric("Нитраты", f"{card_row['nitrates_mg_l']:.2f} мг/л")
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Мутность", f"{card_row['turbidity_ntu']:.1f} NTU")
    m6.metric("Температура воды", f"{card_row['temp_water']:.1f} °C")
    m7.metric("Расход воды", f"{card_row['discharge_m3s']:.2f} м³/с")
    m8.metric("БПК5", f"{card_row['bod5_mg_l']:.2f} мг/л")

# ------------------------------------------------------------------
# ВКЛАДКА 4: МАТЕМАТИЧЕСКАЯ МОДЕЛЬ
# ------------------------------------------------------------------
with tab_math:
    st.subheader("Математическая модель загрязнения")

    sub_regression, sub_pde = st.tabs(["📉 Регрессия факторов", "🌊 Перенос загрязнителя (PDE)"])

    # --- (а) Регрессия ---
    with sub_regression:
        st.markdown("Модель предсказывает **индекс загрязнения** по гидрохимическим факторам.")
        model_type_label = st.radio(
            "Тип модели", ["Линейная регрессия", "Random Forest"], horizontal=True,
        )
        model_type = "linear" if model_type_label == "Линейная регрессия" else "forest"

        reg_model, scaler, metrics, importance, result_df = model_module.train_regression(
            fdf, model_type=model_type
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("R² (тест)", f"{metrics['R2']:.3f}")
        r2.metric("MAE (тест)", f"{metrics['MAE']:.4f}")
        r3.metric("Размер выборки", f"{metrics['n_train']} train / {metrics['n_test']} test")

        colL, colR = st.columns(2)
        with colL:
            imp_df = importance.reset_index()
            imp_df.columns = ["Фактор", "Значимость"]
            imp_df["Фактор"] = imp_df["Фактор"].map(analysis.PARAM_LABELS).fillna(imp_df["Фактор"])
            title = "Коэффициенты влияния (стандартизованные)" if model_type == "linear" else "Важность признаков"
            fig_imp = px.bar(
                imp_df, x="Значимость", y="Фактор", orientation="h",
                template=analysis.PLOT_TEMPLATE, title=title,
                color="Значимость", color_continuous_scale="Teal",
            )
            fig_imp.update_layout(height=420)
            st.plotly_chart(fig_imp, width="stretch", key=f"reg_importance_{model_type}")
        with colR:
            fig_scatter = px.scatter(
                result_df, x="Факт", y="Прогноз",
                template=analysis.PLOT_TEMPLATE, title="Факт vs Прогноз (тестовая выборка)",
                opacity=0.5,
            )
            fig_scatter.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="red", dash="dash")
            )
            fig_scatter.update_layout(height=420)
            st.plotly_chart(fig_scatter, width="stretch", key=f"reg_scatter_{model_type}")

    # --- (б) PDE перенос загрязнителя ---
    with sub_pde:
        st.markdown("Одномерное уравнение адвекции-диффузии-реакции для переноса загрязнителя вдоль русла:")
        st.latex(r"\frac{\partial C}{\partial t} = D \frac{\partial^2 C}{\partial x^2} - u \frac{\partial C}{\partial x} - kC")
        st.caption(
            "C(x,t) — концентрация загрязнителя, D — коэффициент турбулентной диффузии, "
            "u — скорость течения, k — константа распада (самоочищение реки). "
            "Решается методом конечных разностей (явная схема FTCS)."
        )

        p1, p2, p3 = st.columns(3)
        u_val = p1.slider("Скорость течения u, м/с", 0.05, 2.0, 0.3, 0.05)
        D_val = p2.slider("Коэффициент диффузии D, м²/с", 0.1, 20.0, 5.0, 0.1)
        k_val = p3.slider("Константа распада k, 1/с", 0.0, 0.001, 0.0001, 0.00005, format="%.5f")

        p4, p5, p6 = st.columns(3)
        C0_val = p4.slider("Начальная концентрация C₀, мг/л", 1.0, 50.0, 10.0, 1.0)
        L_val = p5.slider("Длина участка L, м", 500.0, 20000.0, 5000.0, 500.0)
        T_val = p6.slider("Время моделирования T, сек", 600.0, 21600.0, 3600.0, 600.0)

        x, t_snap, C_hist, dt_used, n_steps = model_module.solve_advection_diffusion(
            u=u_val, D=D_val, k=k_val, C0=C0_val, L=L_val, T=T_val, nx=200
        )
        st.caption(f"Шаг сетки Δx={L_val/199:.1f} м, шаг по времени Δt={dt_used:.2f} с, число шагов схемы: {n_steps}")

        # Анимированный график распространения загрязнения
        frames = [
            go.Frame(
                data=[go.Scatter(x=x, y=C_hist[i], mode="lines", fill="tozeroy",
                                  line=dict(color="#00b4d8", width=3))],
                name=str(i),
            )
            for i in range(len(t_snap))
        ]
        fig_pde = go.Figure(
            data=[go.Scatter(x=x, y=C_hist[0], mode="lines", fill="tozeroy",
                              line=dict(color="#00b4d8", width=3))],
            frames=frames,
        )
        fig_pde.update_layout(
            template=analysis.PLOT_TEMPLATE, height=480,
            title="Распространение загрязнителя вдоль реки, C(x, t)",
            xaxis_title="Расстояние от точки сброса x, м",
            yaxis_title="Концентрация C, мг/л",
            yaxis_range=[0, C0_val * 1.05],
            updatemenus=[dict(
                type="buttons", showactive=False,
                buttons=[
                    dict(label="▶ Пуск", method="animate",
                         args=[None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}]),
                    dict(label="⏸ Пауза", method="animate",
                         args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                ],
            )],
            sliders=[dict(
                steps=[dict(method="animate", args=[[str(i)], {"mode": "immediate"}],
                            label=f"{t_snap[i]:.0f} c") for i in range(len(t_snap))],
                currentvalue={"prefix": "t = "},
            )],
        )
        st.plotly_chart(fig_pde, width="stretch", key="pde_animation")

        st.markdown("##### Тепловая карта C(x, t)")
        fig_heat = px.imshow(
            C_hist, x=x, y=t_snap, aspect="auto",
            color_continuous_scale="Turbo",
            labels={"x": "Расстояние, м", "y": "Время, с", "color": "C, мг/л"},
            template=analysis.PLOT_TEMPLATE,
        )
        fig_heat.update_layout(height=420, title="Эволюция загрязнения во времени и пространстве")
        st.plotly_chart(fig_heat, width="stretch", key="pde_heatmap")

# ------------------------------------------------------------------
# ВКЛАДКА 5: ПРОГНОЗ
# ------------------------------------------------------------------
with tab_forecast:
    st.subheader("Прогноз индекса загрязнения")

    fcol1, fcol2 = st.columns([1, 1])
    forecast_station = fcol1.selectbox(
        "Станция для прогноза",
        options=selected_stations,
        format_func=lambda sid: station_labels[sid],
    )
    horizon_days = fcol2.select_slider("Горизонт прогноза", options=[7, 30, 90], value=30)
    horizon_steps = max(horizon_days // 3, 1)  # частота ряда - раз в 3 дня

    series = forecast_module.prepare_series(df[df["station_id"] == forecast_station], forecast_station)

    with st.spinner("Обучение модели SARIMAX и бэктест..."):
        bt_metrics, test_series, pred_series = forecast_module.backtest_metrics(series, horizon_steps=horizon_steps)
        future_mean, future_ci = forecast_module.forecast_future(series, steps=horizon_steps)

    b1, b2, b3 = st.columns(3)
    b1.metric("MAE (бэктест)", bt_metrics["MAE"])
    b2.metric("RMSE (бэктест)", bt_metrics["RMSE"])
    b3.metric("MAPE, % (бэктест)", bt_metrics["MAPE_%"])

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=series.index[-120:], y=series.values[-120:],
        mode="lines", name="Факт (история)", line=dict(color="#00b4d8"),
    ))
    fig_fc.add_trace(go.Scatter(
        x=future_ci.index.tolist() + future_ci.index[::-1].tolist(),
        y=future_ci["upper"].tolist() + future_ci["lower"][::-1].tolist(),
        fill="toself", fillcolor="rgba(250,204,21,0.2)",
        line=dict(color="rgba(255,255,255,0)"), name="Доверительный интервал 95%",
        showlegend=True,
    ))
    fig_fc.add_trace(go.Scatter(
        x=future_mean.index, y=future_mean.values,
        mode="lines+markers", name=f"Прогноз на {horizon_days} дн.",
        line=dict(color="#facc15", dash="dash"),
    ))
    fig_fc.update_layout(
        template=analysis.PLOT_TEMPLATE, height=480,
        title=f"Прогноз индекса загрязнения — {station_labels[forecast_station]}",
        xaxis_title="Дата", yaxis_title="Индекс загрязнения", hovermode="x unified",
    )
    st.plotly_chart(fig_fc, width="stretch", key="forecast_chart")

    st.caption(
        "Модель: ARIMA(2,1,2) с регрессорами Фурье (2 гармоники, период ≈121 шаг "
        "= 1 год при частоте ряда 3 дня) — учитывает годовую сезонность быстрее, "
        "чем полный сезонный SARIMAX. Метрики получены бэктестом на отложенном "
        "хвосте ряда той же длины, что и горизонт прогноза."
    )

# ------------------------------------------------------------------
# ВКЛАДКА 6: ВЫВОДЫ
# ------------------------------------------------------------------
with tab_conclusions:
    st.subheader("Выводы и рекомендации")

    peak_info = analysis.peak_pollution_period(fdf)

    c1, c2, c3 = st.columns(3)
    c1.metric("Наиболее загрязнённая станция", peak_info["worst_station"])
    c2.metric("Пиковый месяц загрязнения", peak_info["worst_month"])
    c3.metric("Средний индекс в этой точке", peak_info["worst_value"])

    st.markdown(f"""
### Основные наблюдения

1. **Градиент "исток → устье"**: качество воды устойчиво ухудшается вниз по течению —
   от «отличного» у истока в горах до «удовлетворительного» на выходе к Шымкенту.
   Это указывает на накопление антропогенной нагрузки (стоки населённых пунктов,
   сельхозугодья) по мере движения воды к городу.

2. **Сезонность**: пик проблем приходится на **{peak_info['worst_month']}** — совпадает
   с летней жарой (падение растворённого кислорода) и снижением расхода воды
   (меньшее разбавление загрязнителей). Весной (снеготаяние) растёт мутность и расход,
   но одновременно смывает нитраты с полей.

3. **Ключевые факторы риска** (по регрессионной модели): БПК5 и концентрация нитратов —
   сильнейшие предикторы индекса загрязнения, что говорит об органическом и
   сельскохозяйственном происхождении загрязнения, а не природных факторах (pH, температура).

4. **Прогноз**: модель SARIMAX показывает MAPE {bt_metrics['MAPE_%']}%
   на горизонте {horizon_days} дней — приемлемая точность для
   раннего предупреждения о росте загрязнения.

### Рекомендации

- Усилить контроль сбросов в районе **{peak_info['worst_station']}** — там сходится
  максимальная антропогенная нагрузка.
- Ввести мониторинг БПК5 и нитратов как приоритетных индикаторов (наиболее информативны).
- Обратить особое внимание на летний период (**{peak_info['worst_month']}**) — минимальный
  расход воды снижает способность реки к самоочищению.
- Для дальнейшего развития проекта — заменить синтетические данные на реальные ряды
  Казгидромета при получении доступа к первичным измерениям.

### 📚 Источники и методология

- **Казгидромет (kazhydromet.kz)** — официальный мониторинг качества поверхностных вод РК (бюллетени).
- **data.egov.kz / stat.gov.kz** — агрегированная государственная статистика по экологии.
- **WHO Guidelines for Drinking-water Quality** — нормативы для pH, нитратов, кислорода,
  использованные при калибровке индекса загрязнения.
- **СанПиН РК** — санитарные нормы качества воды в РК.
- Данные в этом дашборде — **синтетические**, сгенерированы с учётом реальной физики
  горной реки (снеготаяние, сезонность, антропогенный градиент), см. `src/load.py`.
""")
