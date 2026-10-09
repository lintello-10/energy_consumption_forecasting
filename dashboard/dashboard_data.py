"""Data helpers for the historical forecasting dashboard."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from energy_forecasting.data import load_hourly_target

FINAL_TEST_RUNS = {
    "final-test-previous-day seasonal naive": "Previous-day seasonal naive",
    "final-test-weekday-hour average": "Weekday-hour average",
    "final-test-gradient boosting": "Gradient boosting",
}


def load_forecast_data(
    raw_path: str | Path,
    model_uri: str,
    tracking_uri: str,
) -> tuple[pd.Series, pd.Series]:
    """Load the historical target and forecast from a trusted MLflow model."""
    raw_path = Path(raw_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")
    if not model_uri.strip():
        raise ValueError("A trusted MLflow model URI must be provided.")

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    from energy_forecasting.predict import forecast_from_model_uri

    target = load_hourly_target(raw_path)
    predictions = forecast_from_model_uri(model_uri, target)
    return target, predictions


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


def load_final_test_metrics(tracking_uri: str) -> pd.DataFrame:
    """Read the existing final-test metrics from the local MLflow store."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(
        "household-electricity-forecasting"
    )
    if experiment is None:
        raise ValueError(
            "MLflow experiment 'household-electricity-forecasting' was not found."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.evaluation_split = 'final_test'",
        max_results=1000,
    )
    return format_final_test_runs(runs)
