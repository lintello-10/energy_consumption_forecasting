from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.dashboard_data import (
    FINAL_TEST_RUNS,
    build_recent_plot_series,
    format_final_test_runs,
    load_demo_forecast,
    load_demo_metrics,
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


def test_load_demo_forecast_reads_history_and_following_24_predictions(tmp_path):
    timestamps = pd.date_range("2010-11-25 20:00", periods=48, freq="h")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "observed_kwh": [1.0] * 24 + [np.nan] * 24,
            "forecast_kwh": [np.nan] * 24 + [0.5] * 24,
        }
    )
    path = tmp_path / "demo_forecast.csv"
    frame.to_csv(path, index=False)

    target, predictions = load_demo_forecast(path)

    assert target.notna().sum() == 24
    assert predictions.eq(0.5).all()
    assert predictions.index[0] == target.last_valid_index() + pd.Timedelta(hours=1)
    assert len(predictions) == 24


def test_load_demo_forecast_rejects_nonconsecutive_predictions(tmp_path):
    timestamps = list(pd.date_range("2010-11-25 20:00", periods=24, freq="h"))
    timestamps.extend(
        pd.date_range("2010-11-26 21:00", periods=24, freq="h")
    )
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "observed_kwh": [1.0] * 24 + [np.nan] * 24,
            "forecast_kwh": [np.nan] * 24 + [0.5] * 24,
        }
    )
    path = tmp_path / "demo_forecast.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="immediately after"):
        load_demo_forecast(path)


def test_load_demo_metrics_reads_all_published_methods(tmp_path):
    metrics = pd.DataFrame(
        [
            {
                "Method": method,
                "MAE (kWh)": 0.5,
                "RMSE (kWh)": 0.7,
                "Daily windows": 193,
            }
            for method in FINAL_TEST_RUNS.values()
        ]
    )
    path = tmp_path / "demo_metrics.csv"
    metrics.to_csv(path, index=False)

    result = load_demo_metrics(path)

    assert set(result["Method"]) == set(FINAL_TEST_RUNS.values())
    assert result["Daily windows"].tolist() == [193, 193, 193]


def test_bundled_demo_assets_are_loadable_and_time_aligned():
    dashboard_dir = Path(__file__).resolve().parents[1] / "dashboard"
    target, predictions = load_demo_forecast(
        dashboard_dir / "demo_forecast.csv"
    )
    metrics = load_demo_metrics(dashboard_dir / "demo_metrics.csv")

    history, validated_predictions = build_recent_plot_series(
        target,
        predictions,
        history_hours=14 * 24,
    )

    assert len(history) == 14 * 24
    assert len(validated_predictions) == 24
    assert metrics["Daily windows"].eq(193).all()


def test_streamlit_app_renders_from_dashboard_directory():
    app_path = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert len(app.exception) == 0
