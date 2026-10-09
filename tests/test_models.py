import numpy as np
import pandas as pd
import pytest

from energy_forecasting.models import (
    fit_weekday_hour_profile,
    forecast_previous_day,
    forecast_weekday_hour_profile,
)

from energy_forecasting.models import (
    fit_gradient_boosting,
    forecast_next_24_hours,
)


def test_gradient_boosting_returns_24_hour_forecast():
    timestamps = pd.date_range(
        "2020-01-01",
        periods=120,
        freq="h",
    )
    values = pd.Series(
        1.0 + 0.2 * np.sin(2 * np.pi * timestamps.hour / 24),
        index=timestamps,
        name="target_kwh",
    )

    model = fit_gradient_boosting(values.iloc[:96])
    forecast = forecast_next_24_hours(model, values.iloc[:96])

    assert len(forecast) == 24
    assert forecast.index[0] == timestamps[96]
    assert forecast.index[-1] == timestamps[119]
    assert np.isfinite(forecast).all()


def test_forecast_rejects_incomplete_recent_history():
    timestamps = pd.date_range(
        "2020-01-01",
        periods=72,
        freq="h",
    )
    values = pd.Series(np.ones(72), index=timestamps, name="target_kwh")
    model = fit_gradient_boosting(values.iloc[:48])

    incomplete_history = values.iloc[:48].copy()
    incomplete_history.iloc[-5] = np.nan

    with pytest.raises(ValueError, match="most recent 24"):
        forecast_next_24_hours(model, incomplete_history)
        
        
def test_previous_day_baseline_repeats_last_24_targets():
    timestamps = pd.date_range("2020-01-01", periods=48, freq="h")
    history = pd.Series(np.arange(48, dtype=float), index=timestamps)

    forecast = forecast_previous_day(history)

    assert len(forecast) == 24
    np.testing.assert_array_equal(forecast.to_numpy(), np.arange(24, 48))
    assert forecast.index[0] == timestamps[-1] + pd.Timedelta(hours=1)


def test_weekday_hour_profile_predicts_from_training_means():
    timestamps = pd.date_range("2020-01-06", periods=7 * 24, freq="h")
    training_target = pd.Series(np.arange(7 * 24, dtype=float), index=timestamps)

    profile = fit_weekday_hour_profile(training_target)
    origin = timestamps[-1] + pd.Timedelta(hours=1)
    forecast = forecast_weekday_hour_profile(profile, origin)

    assert len(forecast) == 24
    assert np.isfinite(forecast).all()
    assert forecast.index[0] == origin        