"""Data helpers for the historical forecasting dashboard."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FINAL_TEST_RUNS = {
    "final-test-previous-day seasonal naive": "Previous-day seasonal naive",
    "final-test-weekday-hour average": "Weekday-hour average",
    "final-test-gradient boosting": "Gradient boosting",
}


def load_demo_forecast(path: str | Path) -> tuple[pd.Series, pd.Series]:
    """Load the bundled historical example without accessing MLflow or raw data."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Bundled demo forecast not found: {path}")

    frame = pd.read_csv(path)
    required_columns = {"timestamp", "observed_kwh", "forecast_kwh"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Demo forecast is missing columns: {sorted(missing_columns)}"
        )

    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    if timestamps.isna().any() or timestamps.duplicated().any():
        raise ValueError("Demo timestamps must be present and unique.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Demo timestamps must be sorted chronologically.")

    observed = pd.to_numeric(frame["observed_kwh"], errors="raise")
    forecast = pd.to_numeric(frame["forecast_kwh"], errors="raise")
    for column, values in (
        ("observed_kwh", observed),
        ("forecast_kwh", forecast),
    ):
        if not np.isfinite(values.dropna()).all():
            raise ValueError(f"Demo {column} values must be finite.")

    index = pd.DatetimeIndex(timestamps, name="timestamp")
    target = pd.Series(
        observed.to_numpy(dtype=float),
        index=index,
        name="target_kwh",
    )
    predictions = pd.Series(
        forecast.dropna().to_numpy(dtype=float),
        index=index[forecast.notna()],
        name="prediction_kwh",
    )

    last_observation = target.last_valid_index()
    if last_observation is None:
        raise ValueError("Demo forecast must include historical observations.")
    if len(predictions) != 24:
        raise ValueError("Demo forecast must contain exactly 24 predictions.")
    if (observed.notna() & forecast.notna()).any():
        raise ValueError(
            "A timestamp cannot contain both an observation and a prediction."
        )

    expected_index = pd.date_range(
        start=last_observation + pd.Timedelta(hours=1),
        periods=24,
        freq="h",
        name="timestamp",
    )
    if not predictions.index.equals(expected_index):
        raise ValueError(
            "Demo predictions must cover 24 consecutive hours immediately "
            "after the last observation."
        )

    return target, predictions


def load_demo_metrics(path: str | Path) -> pd.DataFrame:
    """Load the published evaluation summary bundled with the dashboard."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Bundled demo metrics not found: {path}")

    metrics = pd.read_csv(path)
    required_columns = {
        "Method",
        "MAE (kWh)",
        "RMSE (kWh)",
        "Daily windows",
    }
    missing_columns = required_columns.difference(metrics.columns)
    if missing_columns:
        raise ValueError(
            f"Demo metrics are missing columns: {sorted(missing_columns)}"
        )
    if metrics["Method"].duplicated().any():
        raise ValueError("Demo metric methods must be unique.")

    expected_methods = list(FINAL_TEST_RUNS.values())
    if set(metrics["Method"]) != set(expected_methods):
        raise ValueError("Demo metrics must include all three evaluated methods.")

    for column in ("MAE (kWh)", "RMSE (kWh)", "Daily windows"):
        values = pd.to_numeric(metrics[column], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError(f"Demo metric {column!r} values must be finite.")
        metrics[column] = values
    if (metrics["Daily windows"] <= 0).any():
        raise ValueError("Demo metrics must report positive daily-window counts.")

    return metrics


def build_recent_plot_series(
    target: pd.Series,
    predictions: pd.Series,
    history_hours: int = 24 * 7,
) -> tuple[pd.Series, pd.Series]:
    """Return recent observed history and a validated 24-hour forecast."""
    if not isinstance(target.index, pd.DatetimeIndex):
        raise TypeError("target must have a DatetimeIndex.")
    if target.index.has_duplicates or not target.index.is_monotonic_increasing:
        raise ValueError("target timestamps must be unique and chronological.")
    if history_hours <= 0:
        raise ValueError("history_hours must be positive.")

    last_complete_timestamp = target.last_valid_index()
    if last_complete_timestamp is None:
        raise ValueError("The target contains no complete hourly observations.")

    complete_target = target.loc[:last_complete_timestamp]
    if not isinstance(predictions.index, pd.DatetimeIndex):
        raise TypeError("predictions must have a DatetimeIndex.")
    if len(predictions) != 24:
        raise ValueError("The dashboard requires exactly 24 hourly predictions.")
    if (
        predictions.index.has_duplicates
        or not predictions.index.is_monotonic_increasing
    ):
        raise ValueError("Prediction timestamps must be unique and chronological.")
    if predictions.isna().any() or not np.isfinite(predictions).all():
        raise ValueError("Predictions must contain only finite values.")

    expected_index = pd.date_range(
        start=last_complete_timestamp + pd.Timedelta(hours=1),
        periods=24,
        freq="h",
        name=predictions.index.name,
    )
    if not predictions.index.equals(expected_index):
        raise ValueError(
            "Predictions must cover the 24 hours immediately after "
            "the latest complete target."
        )

    first_history_timestamp = last_complete_timestamp - pd.Timedelta(
        hours=history_hours - 1
    )
    recent_history = complete_target.loc[
        complete_target.index >= first_history_timestamp
    ]
    if recent_history.empty:
        raise ValueError("No recent complete history is available to display.")

    return recent_history, predictions


def format_final_test_runs(runs: Iterable[object]) -> pd.DataFrame:
    """Format the existing final-test MLflow runs without creating new runs."""
    runs_by_name = {run.info.run_name: run for run in runs}
    missing_runs = set(FINAL_TEST_RUNS).difference(runs_by_name)
    if missing_runs:
        raise ValueError(
            "Missing final-test MLflow runs: "
            + ", ".join(sorted(missing_runs))
        )

    rows = []
    for run_name, method_name in FINAL_TEST_RUNS.items():
        run = runs_by_name[run_name]
        metrics = run.data.metrics
        missing_metrics = {
            "test_mae_kwh",
            "test_rmse_kwh",
        }.difference(metrics)
        if missing_metrics:
            raise ValueError(
                f"Run {run_name!r} is missing metrics: "
                + ", ".join(sorted(missing_metrics))
            )

        valid_windows = run.data.params.get("valid_windows")
        if valid_windows is None:
            raise ValueError(f"Run {run_name!r} is missing valid_windows.")

        rows.append(
            {
                "Method": method_name,
                "MAE (kWh)": metrics["test_mae_kwh"],
                "RMSE (kWh)": metrics["test_rmse_kwh"],
                "Daily windows": int(valid_windows),
            }
        )

    return pd.DataFrame(rows)
