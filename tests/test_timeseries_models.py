import numpy as np
import pandas as pd

import timeseries_models as tsm


def _synthetic_series(n=250):
    rng = np.random.default_rng(0)
    t = np.arange(n)
    trend = 0.01 * t
    noise = rng.normal(0, 0.3, size=n)
    return pd.Series(10 + trend + noise)


def test_train_test_split_series_is_chronological():
    series = _synthetic_series(100)
    train, test = tsm.train_test_split_series(series, test_size=0.2)
    assert len(train) == 80
    assert len(test) == 20
    assert train.index[-1] < test.index[0]  # no shuffling


def test_core_models_produce_correct_length_forecasts():
    """ARIMA/SARIMA/Holt-Winters ship with statsmodels (no optional
    dependency), so these must always run successfully."""
    series = _synthetic_series()
    train, test = tsm.train_test_split_series(series)
    horizon = len(test)

    for fit_fn in (tsm._fit_arima, tsm._fit_sarima, tsm._fit_holt_winters):
        forecast, _ = fit_fn(train, horizon)
        assert forecast is not None
        assert len(forecast) == horizon
        assert np.isfinite(forecast).all()


def test_run_all_models_never_crashes_on_optional_deps():
    """Whether or not pmdarima/prophet/tensorflow are installed, every
    model must come back as either 'ok' or a clear skip/failure status --
    never raise and take the whole run down with it."""
    series = _synthetic_series()
    train, test, results = tsm.run_all_models(series)

    assert set(results.keys()) == set(tsm.MODEL_REGISTRY.keys())
    for name, res in results.items():
        assert "status" in res
        if res["status"] == "ok":
            assert len(res["forecast"]) == len(test)
            assert res["RMSE"] >= 0


def test_compare_models_sorts_by_rmse():
    series = _synthetic_series()
    _, _, results = tsm.run_all_models(series)
    comparison_df = tsm.compare_models(results)

    ok_rmses = comparison_df.loc[comparison_df["status"] == "ok", "RMSE"].tolist()
    assert ok_rmses == sorted(ok_rmses)
