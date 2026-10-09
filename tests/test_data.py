import pandas as pd
import pytest

from energy_forecasting.data import load_hourly_target


def test_load_hourly_target_keeps_incomplete_hours_missing(tmp_path):
    timestamps = pd.date_range(
        "2006-12-16 17:00:00",
        periods=120,
        freq="min",
    )
    power_kw = [1.0] * len(timestamps)
    power_kw[75] = float("nan")

    raw_data = pd.DataFrame(
        {
            "Date": timestamps.strftime("%d/%m/%Y"),
            "Time": timestamps.strftime("%H:%M:%S"),
            "Global_active_power": power_kw,
        }
    )
    raw_path = tmp_path / "minute_data.txt"
    raw_data.to_csv(raw_path, sep=";", index=False)

    target = load_hourly_target(raw_path)

    assert len(target) == 2
    assert target.iloc[0] == pytest.approx(1.0)
    assert pd.isna(target.iloc[1])


def test_load_hourly_target_rejects_minute_misaligned_timestamps(tmp_path):
    timestamps = pd.date_range(
        "2006-12-16 17:00:00",
        periods=120,
        freq="min",
    )
    timestamps = timestamps.to_series(index=range(len(timestamps)))
    timestamps.iloc[30] += pd.Timedelta(seconds=30)
    power_kw = [1.0] * len(timestamps)

    raw_data = pd.DataFrame(
        {
            "Date": timestamps.dt.strftime("%d/%m/%Y"),
            "Time": timestamps.dt.strftime("%H:%M:%S"),
            "Global_active_power": power_kw,
        }
    )
    raw_path = tmp_path / "misaligned_minute_data.txt"
    raw_data.to_csv(raw_path, sep=";", index=False)

    target = load_hourly_target(raw_path)

    assert len(target) == 2
    assert pd.isna(target.iloc[0])
    assert target.iloc[1] == pytest.approx(1.0)