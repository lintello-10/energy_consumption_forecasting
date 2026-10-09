import numpy as np
import pandas as pd


def make_supervised_frame(target: pd.Series) -> pd.DataFrame:
    """Build one-step-ahead calendar and lag features without imputing values."""
    if not isinstance(target.index, pd.DatetimeIndex):
        raise TypeError("The target must have a DatetimeIndex.")
    if target.index.has_duplicates:
        raise ValueError("The target index must not contain duplicate timestamps.")
    if not target.index.is_monotonic_increasing:
        raise ValueError("The target index must be sorted in chronological order.")
    if len(target) > 1 and not (target.index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError("The target index must have a regular one-hour frequency.")

    features = pd.DataFrame(index=target.index)
    features["hour_sin"] = np.sin(2 * np.pi * target.index.hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * target.index.hour / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * target.index.dayofweek / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * target.index.dayofweek / 7)
    features["lag_1"] = target.shift(1)
    features["lag_24"] = target.shift(24)
    features["target_kwh"] = target

    return features