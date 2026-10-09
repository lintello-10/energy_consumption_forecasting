import numpy as np
import pandas as pd
import pytest

from energy_forecasting.features import make_supervised_frame


def test_make_supervised_frame_creates_lags_without_imputing():
    timestamps = pd.date_range(
        "2020-01-01",
        periods=30,
        freq="h",
    )
    values = pd.Series(np.arange(30, dtype=float), index=timestamps)
    values.iloc[25] = float("nan")

    frame = make_supervised_frame(values)

    assert frame.loc[timestamps[24], "lag_1"] == pytest.approx(23.0)
    assert frame.loc[timestamps[24], "lag_24"] == pytest.approx(0.0)
    assert pd.isna(frame.loc[timestamps[25], "target_kwh"])
    assert pd.isna(frame.loc[timestamps[26], "lag_1"])