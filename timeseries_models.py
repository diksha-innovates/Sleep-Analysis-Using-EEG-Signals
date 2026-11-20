"""Six time-series models applied to one real signal: a subject's delta-band
EEG power across the epochs of a full night. This replaces the original
notebook's ARIMA/SARIMA cell, which fit on `np.random.randn(500)` (pure
noise, disconnected from the actual EEG) and reported a fabricated
`random.uniform(85, 95)` "accuracy" instead of any computed metric.

Each model forecasts the held-out tail of the night from the preceding
epochs; evaluation.regression_metrics scores the forecast against what
actually happened. pmdarima / prophet / tensorflow are optional -- if one
isn't installed, that model is skipped (reported in the comparison table)
rather than crashing the run.
"""

import time
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

import config
import evaluation

warnings.filterwarnings("ignore")

try:
    import pmdarima as pm
    _HAS_PMDARIMA = True
except ImportError:
    _HAS_PMDARIMA = False

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except ImportError:
    _HAS_PROPHET = False

try:
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.models import Sequential
    _HAS_TENSORFLOW = True
except ImportError:
    _HAS_TENSORFLOW = False


def train_test_split_series(series: pd.Series, test_size=config.TS_TEST_SIZE):
    n_test = max(1, int(len(series) * test_size))
    return series.iloc[:-n_test], series.iloc[-n_test:]


def _fit_arima(train, horizon):
    fitted = ARIMA(train.values, order=config.ARIMA_ORDER).fit()
    return np.asarray(fitted.forecast(steps=horizon)), fitted.aic


def _fit_sarima(train, horizon):
    model = SARIMAX(
        train.values,
        order=config.SARIMA_ORDER,
        seasonal_order=(*config.SARIMA_ORDER, config.SARIMA_SEASONAL_PERIOD),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return np.asarray(fitted.forecast(steps=horizon)), fitted.aic


def _fit_holt_winters(train, horizon):
    seasonal_periods = config.SARIMA_SEASONAL_PERIOD if len(train) >= 2 * config.SARIMA_SEASONAL_PERIOD else None
    model = ExponentialSmoothing(
        train.values,
        trend="add",
        seasonal="add" if seasonal_periods else None,
        seasonal_periods=seasonal_periods,
    )
    fitted = model.fit()
    return np.asarray(fitted.forecast(horizon)), getattr(fitted, "aic", np.nan)


def _fit_auto_arima(train, horizon):
    if not _HAS_PMDARIMA:
        return None, None
    model = pm.auto_arima(train.values, seasonal=False, suppress_warnings=True, error_action="ignore")
    return np.asarray(model.predict(n_periods=horizon)), model.aic()


def _fit_prophet(train, horizon):
    if not _HAS_PROPHET:
        return None, None
    df = pd.DataFrame({
        "ds": pd.date_range("2024-01-01", periods=len(train), freq="30s"),
        "y": train.values,
    })
    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon, freq="30s", include_history=False)
    forecast = model.predict(future)["yhat"].to_numpy()
    return forecast, None


def _fit_lstm(train, horizon, lookback=config.LSTM_LOOKBACK, epochs=config.LSTM_EPOCHS):
    if not _HAS_TENSORFLOW:
        return None, None
    values = train.values.astype("float32")
    mean, std = values.mean(), values.std() or 1.0
    normed = (values - mean) / std

    X, y = [], []
    for i in range(len(normed) - lookback):
        X.append(normed[i:i + lookback])
        y.append(normed[i + lookback])
    if len(X) < 5:
        return None, None
    X, y = np.array(X).reshape(-1, lookback, 1), np.array(y)

    model = Sequential([LSTM(16, activation="tanh", input_shape=(lookback, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, verbose=0)

    window = list(normed[-lookback:])
    preds = []
    for _ in range(horizon):
        next_val = model.predict(np.array(window[-lookback:]).reshape(1, lookback, 1), verbose=0)[0, 0]
        preds.append(next_val)
        window.append(next_val)

    return np.array(preds) * std + mean, None


MODEL_REGISTRY = {
    "ARIMA": _fit_arima,
    "SARIMA": _fit_sarima,
    "Holt-Winters": _fit_holt_winters,
    "Auto-ARIMA": _fit_auto_arima,
    "Prophet": _fit_prophet,
    "LSTM": _fit_lstm,
}


def run_all_models(series: pd.Series, test_size=config.TS_TEST_SIZE):
    """Fits every model in MODEL_REGISTRY on the same train/test split and
    scores each forecast. Returns (train, test, results_by_model_name)."""
    train, test = train_test_split_series(series, test_size)
    horizon = len(test)
    results = {}

    for name, fit_fn in MODEL_REGISTRY.items():
        start = time.time()
        try:
            forecast, aic = fit_fn(train, horizon)
        except Exception as exc:  # model fitting on real-world data can fail to converge
            results[name] = {"status": f"failed: {exc}"}
            continue

        if forecast is None:
            results[name] = {"status": "skipped (optional dependency not installed)"}
            continue

        metrics = evaluation.regression_metrics(test.values, forecast)
        results[name] = {
            "status": "ok",
            "forecast": forecast,
            "fit_seconds": round(time.time() - start, 2),
            "aic": aic,
            **metrics,
        }

    return train, test, results


def compare_models(results: dict) -> pd.DataFrame:
    """One row per model, ranked by RMSE (NaN/failed/skipped sort last)."""
    rows = []
    for name, res in results.items():
        if res.get("status") != "ok":
            rows.append({"model": name, "status": res.get("status"), "RMSE": np.nan,
                         "MAE": np.nan, "MAPE_%": np.nan, "fit_seconds": np.nan})
            continue
        rows.append({
            "model": name, "status": "ok", "RMSE": res["RMSE"], "MAE": res["MAE"],
            "MAPE_%": res["MAPE_%"], "fit_seconds": res["fit_seconds"],
        })
    return pd.DataFrame(rows).sort_values("RMSE", na_position="last").reset_index(drop=True)
