from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dashboard.dashboard_data import (
    FINAL_TEST_RUNS,
    build_recent_plot_series,
    format_final_test_runs,
)


def test_build_recent_plot_series_separates_history_from_24_hour_forecast():
    timestamps = pd.date_range("2020-01-01", periods=24 * 9, freq="h")
    target = pd.Series(np.ones(len(timestamps)), index=timestamps)
    target.iloc[-1] = np.nan
    latest_complete = timestamps[-2]
    forecast_index = pd.date_range(
        start=latest_complete + pd.Timedelta(hours=1),
        periods=24,
        freq="h",
    )
    predictions = pd.Series(np.arange(24, dtype=float), index=forecast_index)

    recent_history, forecast = build_recent_plot_series(target, predictions)

    assert recent_history.index[-1] == latest_complete
    assert len(recent_history) == 24 * 7
    assert forecast.index[0] == latest_complete + pd.Timedelta(hours=1)
    assert len(forecast) == 24


def test_build_recent_plot_series_rejects_forecast_not_after_latest_observation():
    timestamps = pd.date_range("2020-01-01", periods=48, freq="h")
    target = pd.Series(np.ones(48), index=timestamps)
    predictions = pd.Series(
        np.ones(24),
        index=pd.date_range("2020-01-03 01:00", periods=24, freq="h"),
    )

    with pytest.raises(ValueError, match="immediately after"):
        build_recent_plot_series(target, predictions)


def test_format_final_test_runs_uses_existing_test_metrics_only():
    runs = []
    expected_values = {}
    for offset, (run_name, method_name) in enumerate(FINAL_TEST_RUNS.items()):
        mae = 0.4 + offset / 100
        rmse = 0.6 + offset / 100
        expected_values[method_name] = (mae, rmse)
        runs.append(
            SimpleNamespace(
                info=SimpleNamespace(run_name=run_name),
                data=SimpleNamespace(
                    metrics={
                        "test_mae_kwh": mae,
                        "test_rmse_kwh": rmse,
                    },
                    params={"valid_windows": "193"},
                ),
            )
        )

    result = format_final_test_runs(runs)

    assert result["Method"].tolist() == list(FINAL_TEST_RUNS.values())
    assert result["Daily windows"].tolist() == [193, 193, 193]
    for _, row in result.iterrows():
        assert (row["MAE (kWh)"], row["RMSE (kWh)"]) == expected_values[
            row["Method"]
        ]


def test_format_final_test_runs_fails_when_a_final_method_is_missing():
    with pytest.raises(ValueError, match="Missing final-test MLflow runs"):
        format_final_test_runs([])
