"""
src/model.py
-------------
(а) Регрессионная модель: индекс загрязнения ~ гидрохимические факторы
    (линейная регрессия и Random Forest на выбор), метрики R2 и MAE.

(б) Численное решение уравнения адвекции-диффузии-реакции для переноса
    загрязнителя вдоль русла реки методом конечных разностей (явная схема FTCS):

        dC/dt = D * d2C/dx2 - u * dC/dx - k * C

    где C(x,t) - концентрация загрязнителя, u - скорость течения,
    D - коэффициент турбулентной диффузии, k - константа распада (самоочищение).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "temp_water", "ph", "do_mg_l", "nitrates_mg_l",
    "turbidity_ntu", "discharge_m3s", "bod5_mg_l", "distance_km",
]
TARGET = "pollution_index"


def train_regression(df: pd.DataFrame, model_type: str = "linear",
                      test_size: float = 0.2, random_state: int = 42):
    """
    Обучает регрессионную модель предсказания индекса загрязнения по гидрохимическим
    факторам. model_type: 'linear' (линейная регрессия) или 'forest' (Random Forest).

    Возвращает: model, scaler, metrics(dict R2/MAE), importance(Series), result_df(факт/прогноз на тесте).
    """
    data = df.dropna(subset=FEATURES + [TARGET])
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if model_type == "linear":
        model = LinearRegression()
        model.fit(X_train_s, y_train)
        importance = pd.Series(model.coef_, index=FEATURES).sort_values(key=np.abs, ascending=False)
    else:
        model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=random_state)
        model.fit(X_train_s, y_train)
        importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    y_pred = model.predict(X_test_s)
    metrics = {
        "R2": round(float(r2_score(y_test, y_pred)), 3),
        "MAE": round(float(mean_absolute_error(y_test, y_pred)), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    result_df = X_test.copy()
    result_df["Факт"] = y_test.values
    result_df["Прогноз"] = np.round(y_pred, 3)

    return model, scaler, metrics, importance, result_df


def solve_advection_diffusion(u: float, D: float, k: float, C0: float,
                               L: float = 5000.0, T: float = 3600.0, nx: int = 200):
    """
    Явная конечно-разностная схема (FTCS) для уравнения:
        dC/dt = D * d2C/dx2 - u * dC/dx - k * C

    Параметры:
        u   - скорость течения реки, м/с
        D   - коэффициент турбулентной диффузии, м^2/с
        k   - константа распада загрязнителя (самоочищение), 1/с
        C0  - начальная (пиковая) концентрация загрязнителя в точке сброса, мг/л
        L   - длина моделируемого участка реки, м
        T   - время моделирования, с
        nx  - число узлов расчётной сетки по координате x

    Возвращает: x (сетка расстояний, м), t_snapshots (моменты времени, с),
    C_history (концентрация в каждый сохранённый момент, форма [n_snapshots, nx]),
    dt (фактический шаг по времени), n_steps (число шагов схемы).
    """
    dx = L / (nx - 1)
    x = np.linspace(0, L, nx)

    # Условие устойчивости явной схемы: берём минимум из диффузионного и
    # адвективного (CFL) ограничений с запасом (коэффициент 0.4).
    dt_diff = 0.4 * dx ** 2 / (2 * D) if D > 0 else np.inf
    dt_adv = 0.4 * dx / u if u > 0 else np.inf
    dt = min(dt_diff, dt_adv, 5.0)
    n_steps = max(int(T / dt), 1)

    # Начальное условие: залповый сброс загрязнителя у истока участка (x=0) -
    # гауссов импульс шириной в несколько узлов сетки.
    sigma = dx * 3
    C = C0 * np.exp(-(x ** 2) / (2 * sigma ** 2))

    snapshot_every = max(n_steps // 60, 1)  # ~60 кадров для плавной анимации
    t_snapshots = [0.0]
    C_history = [C.copy()]

    t = 0.0
    for step in range(1, n_steps + 1):
        C_new = C.copy()
        # Диффузия - центральная разность; адвекция - разность "против потока"
        # (upwind, устойчива для u > 0); реакция - линейный распад первого порядка.
        C_new[1:-1] = (
            C[1:-1]
            + dt * D * (C[2:] - 2 * C[1:-1] + C[:-2]) / dx ** 2
            - dt * u * (C[1:-1] - C[:-2]) / dx
            - dt * k * C[1:-1]
        )
        # Граничные условия: свободный вынос с обеих сторон (нулевой градиент)
        C_new[0] = C_new[1]
        C_new[-1] = C_new[-2]
        C = np.clip(C_new, 0, None)  # концентрация физически не может быть отрицательной
        t += dt

        if step % snapshot_every == 0 or step == n_steps:
            t_snapshots.append(t)
            C_history.append(C.copy())

    return x, np.array(t_snapshots), np.array(C_history), dt, n_steps
