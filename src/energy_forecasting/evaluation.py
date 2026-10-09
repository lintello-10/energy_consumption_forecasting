import pandas as pd
import numpy as np

import numpy as np


def forecast_metrics(
    actual: list[float] | np.ndarray,
    predicted: list[float] | np.ndarray,
) -> dict[str, float]:
    """Calculate MAE and RMSE for complete, matching forecast arrays."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)

    if actual_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("Actual and predicted values must be one-dimensional.")
    if actual_values.size == 0:
        raise ValueError("Forecast arrays must not be empty.")
    if actual_values.shape != predicted_values.shape:
        raise ValueError("Actual and predicted arrays must have the same shape.")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Forecast arrays must contain only finite values.")

    errors = actual_values - predicted_values

    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
    }

def split_chronologically(
    data: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-indexed data into ordered train, validation, and test sets."""
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must have a DatetimeIndex.")
    if data.index.has_duplicates:
        raise ValueError("The datetime index must not contain duplicates.")
    if not data.index.is_monotonic_increasing:
        raise ValueError("The data must be sorted chronologically.")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must sum to less than 1.")

    row_count = len(data)
    train_end = int(row_count * train_fraction)
    validation_end = int(row_count * (train_fraction + validation_fraction))

    if train_end == 0 or validation_end == train_end or validation_end >= row_count:
        raise ValueError("The fractions must produce three non-empty time splits.")

    train = data.iloc[:train_end].copy()
    validation = data.iloc[train_end:validation_end].copy()
    test = data.iloc[validation_end:].copy()

    return train, validation, test

def select_valid_daily_origins(
    target: pd.Series,
    evaluation_index: pd.DatetimeIndex,
    horizon: int = 24,
    history_size: int = 24,
) -> list[pd.Timestamp]:
    """Select daily origins with complete history and evaluation targets."""
    if not isinstance(target.index, pd.DatetimeIndex):
        raise TypeError("target must have a DatetimeIndex.")
    if not isinstance(evaluation_index, pd.DatetimeIndex):
        raise TypeError("evaluation_index must be a DatetimeIndex.")
    if target.index.has_duplicates or evaluation_index.has_duplicates:
        raise ValueError("Datetime indexes must not contain duplicates.")
    if not target.index.is_monotonic_increasing:
        raise ValueError("target must be sorted chronologically.")
    if not evaluation_index.is_monotonic_increasing:
        raise ValueError("evaluation_index must be sorted chronologically.")
    if horizon <= 0 or history_size <= 0:
        raise ValueError("horizon and history_size must be positive.")
    if len(target) < 2 or len(evaluation_index) == 0:
        raise ValueError("target and evaluation_index must not be empty.")
    if not evaluation_index.isin(target.index).all():
        raise ValueError("evaluation_index must be contained in target.index.")

    one_hour = pd.Timedelta(hours=1)

    if not (target.index.to_series().diff().iloc[1:] == one_hour).all():
        raise ValueError("target must have a regular one-hour frequency.")
    if len(evaluation_index) > 1 and not (
        evaluation_index.to_series().diff().iloc[1:] == one_hour
    ).all():
        raise ValueError("evaluation_index must be a continuous hourly range.")

    first_origin = evaluation_index[0].ceil("D")
    last_origin = evaluation_index[-1] - (horizon - 1) * one_hour

    if first_origin > last_origin:
        return []

    candidate_origins = pd.date_range(
        start=first_origin,
        end=last_origin,
        freq="24h",
    )

    valid_origins = []

    for origin in candidate_origins:
        history_index = pd.date_range(
            end=origin - one_hour,
            periods=history_size,
            freq="h",
        )
        forecast_index = pd.date_range(
            start=origin,
            periods=horizon,
            freq="h",
        )

        history = target.reindex(history_index)
        actual_targets = target.reindex(forecast_index)

        if history.notna().all() and actual_targets.notna().all():
            valid_origins.append(origin)

    return valid_origins