from pathlib import Path

import pandas as pd


def load_hourly_target(raw_path: str | Path) -> pd.Series:
    """Load minute-level data and return complete hourly energy targets in kWh."""
    data = pd.read_csv(raw_path, sep=";", na_values=["?"], low_memory=False)

    required_columns = {"Date", "Time", "Global_active_power"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    timestamps = pd.to_datetime(
        data["Date"] + " " + data["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="raise",
    )

    if timestamps.isna().any():
        raise ValueError("The dataset contains missing timestamps.")
    if timestamps.duplicated().any():
        raise ValueError("The dataset contains duplicate timestamps.")

    power_kw = pd.to_numeric(data["Global_active_power"], errors="raise")
    minute_data = pd.DataFrame(
        {
            "power_kw": power_kw.to_numpy(),
            "minute_aligned": (
                timestamps == timestamps.dt.floor("min")
            ).to_numpy(dtype="int8"),
        },
        index=pd.DatetimeIndex(timestamps, name="timestamp"),
    )

    hourly_groups = minute_data.groupby(
        minute_data.index.floor("h"),
        sort=True,
    )
    hourly = hourly_groups.agg(
        timestamp_count=("power_kw", "size"),
        valid_minutes=("power_kw", "count"),
        aligned_minutes=("minute_aligned", "sum"),
        power_sum_kw=("power_kw", "sum"),
    )

    full_hour_index = pd.date_range(
        start=hourly.index.min(),
        end=hourly.index.max(),
        freq="h",
        name="timestamp",
    )
    hourly = hourly.reindex(full_hour_index)

    complete_hours = (
        hourly["timestamp_count"].eq(60)
        & hourly["valid_minutes"].eq(60)
        & hourly["aligned_minutes"].eq(60)
    )
    target_kwh = (hourly["power_sum_kw"] / 60).where(complete_hours)
    target_kwh.name = "target_kwh"

    return target_kwh
