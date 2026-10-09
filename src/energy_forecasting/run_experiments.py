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
    fit_gradient_boosting,
    fit_weekday_hour_profile,
    forecast_next_24_hours,
    forecast_previous_day,
    forecast_weekday_hour_profile,
)


def evaluate_forecasts(target, origins, forecast_function):
    """Evaluate one forecasting method on shared complete daily windows."""
    actual_values = []
    predicted_values = []

    for origin in origins:
        history = target.loc[target.index < origin]
        prediction = forecast_function(history, origin)
        actual = target.reindex(prediction.index)

        if actual.isna().any() or prediction.isna().any():
            raise ValueError("Evaluation windows and forecasts must be complete.")

        actual_values.extend(actual.to_numpy(dtype=float))
        predicted_values.extend(prediction.to_numpy(dtype=float))

    return forecast_metrics(actual_values, predicted_values)


def log_validation_run(name, model_family, parameters, metrics, window_count):
    """Log one validation experiment to the local MLflow tracking store."""
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_family", model_family)
        mlflow.set_tag("evaluation_split", "validation")
        mlflow.log_param("forecast_horizon_hours", 24)
        mlflow.log_param("valid_windows", window_count)
        mlflow.log_params(parameters)
        mlflow.log_metrics(
            {
                "validation_mae_kwh": metrics["mae"],
                "validation_rmse_kwh": metrics["rmse"],
            }
        )


def main():
    """Run baseline and gradient boosting validation experiments."""
    project_root = Path(__file__).resolve().parents[2]
    raw_path = project_root / "data" / "raw" / "household_power_consumption.txt"

    target = load_hourly_target(raw_path)
    train, validation, _test = split_chronologically(target.to_frame())
    training_target = train["target_kwh"]

    origins = select_valid_daily_origins(target, validation.index)
    if not origins:
        raise ValueError("No complete daily validation windows were found.")

    tracking_db = project_root / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    mlflow.set_experiment("household-electricity-forecasting")

    results = []

    # Evaluate the previous-day seasonal-naive baseline.
    seasonal_metrics = evaluate_forecasts(
        target,
        origins,
        lambda history, _origin: forecast_previous_day(history),
    )
    log_validation_run(
        "previous-day-seasonal-naive",
        "seasonal_naive",
        {},
        seasonal_metrics,
        len(origins),
    )
    results.append(
        {
            "model": "Previous-day seasonal naive",
            "mae_kwh": seasonal_metrics["mae"],
            "rmse_kwh": seasonal_metrics["rmse"],
        }
    )

    # Fit the weekday-hour profile on training data only.
    weekday_hour_profile = fit_weekday_hour_profile(training_target)
    calendar_metrics = evaluate_forecasts(
        target,
        origins,
        lambda _history, origin: forecast_weekday_hour_profile(
            weekday_hour_profile,
            origin,
        ),
    )
    log_validation_run(
        "weekday-hour-average",
        "weekday_hour_average",
        {},
        calendar_metrics,
        len(origins),
    )
    results.append(
        {
            "model": "Weekday-hour average",
            "mae_kwh": calendar_metrics["mae"],
            "rmse_kwh": calendar_metrics["rmse"],
        }
    )

    # Compare a small set of gradient boosting configurations.
    configurations = [
        {
            "max_iter": 100,
            "learning_rate": 0.1,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
        },
        {
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
        },
        {
            "max_iter": 100,
            "learning_rate": 0.1,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
        },
        {
            "max_iter": 100,
            "learning_rate": 0.1,
            "max_leaf_nodes": 15,
            "l2_regularization": 5.0,
        },
    ]

    for number, parameters in enumerate(configurations, start=1):
        model = fit_gradient_boosting(training_target, **parameters)

        model_metrics = evaluate_forecasts(
            target,
            origins,
            lambda history, _origin: forecast_next_24_hours(model, history),
        )
        log_validation_run(
            f"gradient-boosting-{number}",
            "hist_gradient_boosting",
            parameters,
            model_metrics,
            len(origins),
        )
        results.append(
            {
                "model": f"Gradient boosting {number}",
                "mae_kwh": model_metrics["mae"],
                "rmse_kwh": model_metrics["rmse"],
            }
        )

    results_table = pd.DataFrame(results).sort_values("mae_kwh")
    print(f"Training period: {train.index.min()} to {train.index.max()}")
    print(
        "Validation period: "
        f"{validation.index.min()} to {validation.index.max()}"
    )
    print(f"Complete validation windows: {len(origins)}")
    print("The final test split was not evaluated.")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(results_table.to_string(index=False))


if __name__ == "__main__":
    main()
    