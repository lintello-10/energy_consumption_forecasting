"""Train and log the operational forecasting model to MLflow."""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn

from energy_forecasting.data import load_hourly_target
from energy_forecasting.models import (
    SELECTED_MODEL_PARAMETERS,
    fit_gradient_boosting,
)

SKOPS_TRUSTED_TYPES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
]


def main() -> None:
    """Fit on all available history and log the model without test metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-data",
        type=Path,
        help="Path to the raw household power dataset.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    raw_path = args.raw_data or (
        project_root / "data" / "raw" / "household_power_consumption.txt"
    )
    tracking_db = project_root / "mlflow.db"

    target = load_hourly_target(raw_path)
    model = fit_gradient_boosting(target, **SELECTED_MODEL_PARAMETERS)

    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    mlflow.set_experiment("household-electricity-forecasting")

    with mlflow.start_run(run_name="operational-model-full-history") as run:
        mlflow.set_tag("model_role", "operational_forecasting")
        mlflow.set_tag("training_data_scope", "all_available_history")
        mlflow.log_params(SELECTED_MODEL_PARAMETERS)
        mlflow.log_param("training_valid_targets", int(target.notna().sum()))
        mlflow.log_param("forecast_horizon_hours", 24)
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        model_uri = f"runs:/{run.info.run_id}/model"

    print(f"Training data: {raw_path}")
    print(f"Valid hourly targets used: {int(target.notna().sum())}")
    print("No validation or test metrics were calculated.")
    print(f"Logged model URI: {model_uri}")


if __name__ == "__main__":
    main()
