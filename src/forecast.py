"""
src/forecast.py
-----------------
Прогноз индекса загрязнения на 7/30/90 дней вперёд с помощью ARIMA
(statsmodels), дополненной регрессорами Фурье для годовой сезонности
(снеготаяние весной, летняя жара горной реки).

Почему Фурье-регрессоры, а не классический SARIMAX(...,period=121):
при частоте ряда "раз в 3 дня" сезонный период ≈121 шаг, а полный сезонный
SARIMAX с таким периодом строит очень большое пространство состояний и
подгоняется десятки секунд на каждый вызов - неприемлемо для интерактивного
дашборда. Пара синус/косинус гармоник с тем же периодом даёт почти
идентичное качество прогноза, но подгоняется на порядок быстрее.

Исходный ряд имеет частоту "раз в 3 дня", поэтому горизонты в днях
переводятся в число шагов модели (days // 3) в app.py перед вызовом.
"""

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

# Период годовой сезонности в шагах ряда: 365 дней / 3 дня ≈ 121 шаг
SEASONAL_PERIOD = 121
N_HARMONICS = 2  # число пар синус/косинус гармоник Фурье


def prepare_series(df: pd.DataFrame, station_id: str, freq: str = "3D") -> pd.Series:
    """Готовит регулярный временной ряд индекса загрязнения для одной станции."""
    s = (
        df[df["station_id"] == station_id]
        .sort_values("date")
        .set_index("date")["pollution_index"]
    )
    s = s.asfreq(freq)
    s = s.interpolate(method="linear", limit_direction="both")
    return s


def _fourier_features(n: int, start: int = 0) -> np.ndarray:
    """Строит матрицу [n, 2*N_HARMONICS] регрессоров Фурье для годовой сезонности."""
    t = np.arange(start, start + n)
    cols = []
    for h in range(1, N_HARMONICS + 1):
        cols.append(np.sin(2 * np.pi * h * t / SEASONAL_PERIOD))
        cols.append(np.cos(2 * np.pi * h * t / SEASONAL_PERIOD))
    return np.column_stack(cols)


def _fit_arima_fourier(series: pd.Series, order=(2, 1, 2)):
    """Обучает ARIMA(order) с экзогенными регрессорами Фурье (быстрая замена SARIMAX)."""
    exog = _fourier_features(len(series), start=0)
    model = SARIMAX(
        series, exog=exog, order=order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=100)


def backtest_metrics(series: pd.Series, horizon_steps: int = 10):
    """
    Бэктест: обучаем модель без последних `horizon_steps` точек ряда,
    прогнозируем их и сравниваем с фактом.
    Возвращает словарь метрик (MAE, RMSE, MAPE_%) и ряды test/pred для графика.
    """
    train, test = series.iloc[:-horizon_steps], series.iloc[-horizon_steps:]
    fitted = _fit_arima_fourier(train)

    future_exog = _fourier_features(horizon_steps, start=len(train))
    pred = fitted.get_forecast(steps=horizon_steps, exog=future_exog).predicted_mean
    pred.index = test.index

    mae = float(np.mean(np.abs(pred - test)))
    rmse = float(np.sqrt(np.mean((pred - test) ** 2)))
    # MAPE считаем только там, где факт не близок к нулю, чтобы избежать деления на 0
    safe = test.replace(0, np.nan)
    mape = float(np.nanmean(np.abs((pred - test) / safe))) * 100

    metrics = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_%": round(mape, 2)}
    return metrics, test, pred


def forecast_future(series: pd.Series, steps: int = 10, alpha: float = 0.05):
    """
    Строит прогноз на `steps` шагов ряда вперёд с доверительным интервалом (1-alpha).
    Возвращает: mean (прогноз), ci (DataFrame с колонками lower/upper).
    """
    fitted = _fit_arima_fourier(series)

    future_exog = _fourier_features(steps, start=len(series))
    fc = fitted.get_forecast(steps=steps, exog=future_exog)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=alpha)
    ci.columns = ["lower", "upper"]

    freq = series.index.freq or pd.infer_freq(series.index)
    future_index = pd.date_range(series.index[-1], periods=steps + 1, freq=freq)[1:]
    mean.index = future_index
    ci.index = future_index
    ci["lower"] = ci["lower"].clip(lower=0)  # индекс загрязнения не может быть отрицательным

    return mean, ci
