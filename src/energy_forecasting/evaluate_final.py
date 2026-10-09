from pathlib import Path

import mlflow
import pandas as pd

from energy_forecasting.data import load_hourly_target
from energy_forecasting.evaluation import (
    forecast_metrics,
    select_valid_daily_origins,
    split_chronologically,
)
from energy_forecasting.models import (
    SELECTED_MODEL_PARAMETERS,
    fit_gradient_boosting,
    fit_weekday_hour_profile,
    forecast_next_24_hours,
    forecast_previous_day,
    forecast_weekday_hour_profile,
)


def main():
    """Evaluate the selected model and baselines once on the final holdout."""
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "raw" / "household_power_consumption.txt"

    target = load_hourly_target(raw_path)
    train, validation, test = split_chronologically(target.to_frame())

    # Fit the selected configuration using development data only.
    development_target = pd.concat(
        [train["target_kwh"], validation["target_kwh"]]
    ).sort_index()

    model_parameters = SELECTED_MODEL_PARAMETERS
    model = fit_gradient_boosting(
        development_target,
        **model_parameters,
    )
    weekday_hour_profile = fit_weekday_hour_profile(development_target)

    # Keep only test windows with complete 24-hour history and targets.
    origins = select_valid_daily_origins(target, test.index)
    if not origins:
        raise ValueError("No complete daily windows were found in the test split.")

    actual_by_method = {
        "previous-day seasonal naive": [],
        "weekday-hour average": [],
        "gradient boosting": [],
    }
    predictions_by_method = {
        name: [] for name in actual_by_method
    }

    # Forecast once per day, using only information available before each origin.
    for origin in origins:
        history = target.loc[target.index < origin]
        forecast_index = pd.date_range(
            start=origin,
            periods=24,
            freq="h",
        )
        actual = target.reindex(forecast_index)

        forecasts = {
            "previous-day seasonal naive": forecast_previous_day(history),
            "weekday-hour average": forecast_weekday_hour_profile(
                weekday_hour_profile,
                origin,
            ),
            "gradient boosting": forecast_next_24_hours(model, history),
        }

        for name, forecast in forecasts.items():
            predictions_by_method[name].extend(forecast.to_numpy(dtype=float))
            actual_by_method[name].extend(actual.to_numpy(dtype=float))

    # Store the final-holdout metrics in the existing MLflow experiment.
    tracking_db = project_root / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    mlflow.set_experiment("household-electricity-forecasting")

    results = []

    for name in actual_by_method:
        metrics = forecast_metrics(
            actual_by_method[name],
            predictions_by_method[name],
        )

        with mlflow.start_run(run_name=f"final-test-{name}"):
            mlflow.set_tag("evaluation_split", "final_test")
            mlflow.set_tag(
                "test_period_note",
                "Partially examined during earlier exploratory analysis.",
            )
            mlflow.log_param("forecast_horizon_hours", 24)
            mlflow.log_param("valid_windows", len(origins))

            if name == "gradient boosting":
                mlflow.log_params(model_parameters)

            mlflow.log_metrics(
                {
                    "test_mae_kwh": metrics["mae"],
                    "test_rmse_kwh": metrics["rmse"],
                }
            )

        results.append(
            {
                "model": name,
                "MAE_kWh": metrics["mae"],
                "RMSE_kWh": metrics["rmse"],
            }
        )

    print(f"Test period: {test.index.min()} to {test.index.max()}")
    print(f"Complete 24-hour test windows: {len(origins)}")
    print(f"Hourly predictions per method: {len(origins) * 24}")
    print("Test was not used to choose or tune the model.")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()