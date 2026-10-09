import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.utils.validation import check_is_fitted

from energy_forecasting.features import make_supervised_frame


FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "lag_1",
    "lag_24",
]

SELECTED_MODEL_PARAMETERS = {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
}


def fit_gradient_boosting(
    training_target: pd.Series,
    *,
    max_iter: int = 100,
    learning_rate: float = 0.1,
    max_leaf_nodes: int = 15,
    l2_regularization: float = 1.0,
) -> HistGradientBoostingRegressor:
    """Fit a gradient boosting model using valid training examples only."""
    supervised = make_supervised_frame(training_target)
    training_rows = supervised.dropna()

    if training_rows.empty:
        raise ValueError("Not enough complete training observations to fit the model.")

    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=l2_regularization,
        random_state=42,
    )
    model.fit(
        training_rows[FEATURE_COLUMNS],
        training_rows["target_kwh"],
    )
    return model


def forecast_next_24_hours(
    model: HistGradientBoostingRegressor,
    history: pd.Series,
) -> pd.Series:
    """Recursively forecast the 24 hours following a complete hourly history."""
    check_is_fitted(model)

    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("history must have a DatetimeIndex.")
    if history.index.has_duplicates:
        raise ValueError("history must not contain duplicate timestamps.")
    if not history.index.is_monotonic_increasing:
        raise ValueError("history must be sorted chronologically.")
    if len(history) < 24:
        raise ValueError("At least 24 historical hourly targets are required.")

    history = history.astype(float).copy()
    recent_history = history.iloc[-24:]

    if recent_history.isna().any() or not np.isfinite(recent_history).all():
        raise ValueError("The most recent 24 historical targets must be complete.")
    if not (
        recent_history.index.to_series().diff().iloc[1:]
        == pd.Timedelta(hours=1)
    ).all():
        raise ValueError("The most recent 24 timestamps must be one hour apart.")

    forecast_index = pd.date_range(
        start=history.index[-1] + pd.Timedelta(hours=1),
        periods=24,
        freq="h",
        name=history.index.name,
    )
    predictions = []

    for timestamp in forecast_index:
        lag_1 = history.loc[timestamp - pd.Timedelta(hours=1)]
        lag_24 = history.loc[timestamp - pd.Timedelta(hours=24)]

        row = pd.DataFrame(
            {
                "hour_sin": [np.sin(2 * np.pi * timestamp.hour / 24)],
                "hour_cos": [np.cos(2 * np.pi * timestamp.hour / 24)],
                "weekday_sin": [
                    np.sin(2 * np.pi * timestamp.dayofweek / 7)
                ],
                "weekday_cos": [
                    np.cos(2 * np.pi * timestamp.dayofweek / 7)
                ],
                "lag_1": [lag_1],
                "lag_24": [lag_24],
            },
            index=[timestamp],
        )

        prediction = float(model.predict(row[FEATURE_COLUMNS])[0])
        predictions.append(prediction)
        history.loc[timestamp] = prediction

    return pd.Series(predictions, index=forecast_index, name="prediction_kwh")


def forecast_previous_day(history: pd.Series) -> pd.Series:
    """Repeat the previous 24 hourly targets as the next-day forecast."""
    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("history must have a DatetimeIndex.")
    if len(history) < 24:
        raise ValueError("At least 24 historical targets are required.")

    recent_history = history.iloc[-24:]

    if recent_history.isna().any() or not np.isfinite(recent_history).all():
        raise ValueError("The most recent 24 targets must be complete.")
    if not (
        recent_history.index.to_series().diff().iloc[1:]
        == pd.Timedelta(hours=1)
    ).all():
        raise ValueError("The most recent 24 timestamps must be hourly.")

    forecast_index = pd.date_range(
        start=history.index[-1] + pd.Timedelta(hours=1),
        periods=24,
        freq="h",
        name=history.index.name,
    )

    return pd.Series(
        recent_history.to_numpy(dtype=float),
        index=forecast_index,
        name="prediction_kwh",
    )


def fit_weekday_hour_profile(training_target: pd.Series) -> pd.Series:
    """Calculate mean target energy by weekday and hour from training data."""
    if not isinstance(training_target.index, pd.DatetimeIndex):
        raise TypeError("training_target must have a DatetimeIndex.")

    valid_target = training_target.dropna()
    if valid_target.empty:
        raise ValueError("training_target must contain valid targets.")

    profile_data = pd.DataFrame(
        {
            "weekday": valid_target.index.dayofweek,
            "hour": valid_target.index.hour,
            "target_kwh": valid_target.to_numpy(dtype=float),
        }
    )

    return profile_data.groupby(["weekday", "hour"])["target_kwh"].mean()


def forecast_weekday_hour_profile(
    profile: pd.Series,
    origin: pd.Timestamp,
) -> pd.Series:
    """Forecast 24 hours using training means for each weekday-hour pair."""
    forecast_index = pd.date_range(
        start=origin,
        periods=24,
        freq="h",
    )
    profile_keys = pd.MultiIndex.from_arrays(
        [forecast_index.dayofweek, forecast_index.hour],
        names=profile.index.names,
    )
    predictions = profile.reindex(profile_keys)

    if predictions.isna().any():
        raise ValueError("The profile is missing one or more weekday-hour groups.")

    return pd.Series(
        predictions.to_numpy(dtype=float),
        index=forecast_index,
        name="prediction_kwh",
    )