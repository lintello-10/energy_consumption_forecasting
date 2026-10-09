"""Load a logged MLflow model and forecast the next 24 hourly values."""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from energy_forecasting.data import load_hourly_target
from energy_forecasting.models import forecast_next_24_hours


def forecast_from_model_uri(model_uri: str, history: pd.Series) -> pd.Series:
    """Load an MLflow model and forecast after the latest complete hour."""
    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("history must have a DatetimeIndex.")
    if history.index.has_duplicates:
        raise ValueError("history must not contain duplicate timestamps.")
    if not history.index.is_monotonic_increasing:
        raise ValueError("history must be sorted chronologically.")

    last_valid_timestamp = history.last_valid_index()
    if last_valid_timestamp is None:
        raise ValueError("history must contain at least one complete hourly target.")

    history = history.loc[:last_valid_timestamp]
    model = mlflow.sklearn.load_model(model_uri)
    return forecast_next_24_hours(model, history)


def main() -> None:
    """Forecast the next 24 hours and write the predictions as CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-uri",
        required=True,
        help="MLflow model URI printed by energy_forecasting.train_model.",
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        help="Path to the raw household power dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path; predictions print to stdout by default.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    raw_path = args.raw_data or (
        project_root / "data" / "raw" / "household_power_consumption.txt"
    )
    tracking_db = project_root / "mlflow.db"

    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    history = load_hourly_target(raw_path)
    predictions = forecast_from_model_uri(args.model_uri, history)
    result = predictions.rename_axis("timestamp").reset_index()

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Saved 24 hourly predictions to {args.output}")
    else:
        print(result.to_csv(index=False), end="")


if __name__ == "__main__":
    main()
