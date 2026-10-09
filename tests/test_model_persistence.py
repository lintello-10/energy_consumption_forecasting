import numpy as np
import pandas as pd
import pytest

from energy_forecasting.models import fit_gradient_boosting

SKOPS_TRUSTED_TYPES = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
]


def test_logged_model_can_be_reloaded_for_24_hour_forecast(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    mlflow_sklearn = pytest.importorskip("mlflow.sklearn")
    from energy_forecasting.predict import forecast_from_model_uri

    tracking_db = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
    experiment_id = mlflow.create_experiment(
        "forecast-model-persistence-test",
        artifact_location=(tmp_path / "mlartifacts").as_uri(),
    )

    timestamps = pd.date_range("2020-01-01", periods=24 * 8, freq="h")
    history = pd.Series(
        1.0 + 0.2 * np.sin(2 * np.pi * timestamps.hour / 24),
        index=timestamps,
        name="target_kwh",
    )
    model = fit_gradient_boosting(history)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow_sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        model_uri = f"runs:/{run.info.run_id}/model"

    assert list(tmp_path.rglob("model.skops"))

    history_with_incomplete_final_hour = pd.concat(
        [
            history,
            pd.Series(
                [np.nan],
                index=[timestamps[-1] + pd.Timedelta(hours=1)],
                name="target_kwh",
            ),
        ]
    )
    forecast = forecast_from_model_uri(
        model_uri,
        history_with_incomplete_final_hour,
    )

    assert len(forecast) == 24
    assert forecast.index[0] == timestamps[-1] + pd.Timedelta(hours=1)
    assert forecast.index[-1] == timestamps[-1] + pd.Timedelta(hours=24)
    assert np.isfinite(forecast).all()
