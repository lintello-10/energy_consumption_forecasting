import numpy as np
import pandas as pd
import pytest

from energy_forecasting.evaluation import select_valid_daily_origins
from energy_forecasting.evaluation import split_chronologically



def test_split_chronologically_preserves_order_and_missing_targets():
    timestamps = pd.date_range(
        "2020-01-01",
        periods=100,
        freq="h",
    )
    data = pd.DataFrame(
        {"target_kwh": np.arange(100, dtype=float)},
        index=timestamps,
    )
    data.iloc[75] = float("nan")

    train, validation, test = split_chronologically(data)

    assert len(train) == 70
    assert len(validation) == 15
    assert len(test) == 15
    assert train.index.max() < validation.index.min()
    assert validation.index.max() < test.index.min()
    assert pd.isna(validation.iloc[5]["target_kwh"])


def test_split_chronologically_rejects_invalid_fractions():
    data = pd.DataFrame(
        {"target_kwh": [1.0, 2.0, 3.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="h"),
    )

    with pytest.raises(ValueError, match="sum to less than 1"):
        split_chronologically(
            data,
            train_fraction=0.8,
            validation_fraction=0.2,
        )
        
import numpy as np

from energy_forecasting.evaluation import forecast_metrics


def test_forecast_metrics_calculates_mae_and_rmse():
    metrics = forecast_metrics(
        actual=[1.0, 2.0, 3.0],
        predicted=[1.0, 1.0, 5.0],
    )

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(5 / 3))


def test_forecast_metrics_rejects_missing_values():
    with pytest.raises(ValueError, match="finite values"):
        forecast_metrics(
            actual=[1.0, float("nan")],
            predicted=[1.0, 2.0],
        )        
        
def test_select_valid_daily_origins_skips_incomplete_windows():
    timestamps = pd.date_range(
        "2020-01-01",
        periods=240,
        freq="h",
    )
    target = pd.Series(1.0, index=timestamps, name="target_kwh")
    target.iloc[130] = float("nan")
    target.iloc[150] = float("nan")

    evaluation_index = timestamps[120:]

    origins = select_valid_daily_origins(target, evaluation_index)

    assert origins == [
        pd.Timestamp("2020-01-09 00:00:00"),
        pd.Timestamp("2020-01-10 00:00:00"),
    ]        