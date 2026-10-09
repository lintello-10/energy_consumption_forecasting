# Historical Forecast Dashboard

The Streamlit app displays a bundled 14-day historical sample and a
precomputed 24-hour forecast from the UCI household electricity dataset. The
source data ends in November 2010, so this is a historical demonstration, not
a forecast of current consumption. The hosted app does not access the raw
dataset, load an MLflow model, or generate new predictions.

## Install and run

From the project root, with the project virtual environment activated:

```bash
python -m pip install -e ".[dashboard]"
python -m streamlit run dashboard/app.py
```

The dashboard reads `demo_forecast.csv` and `demo_metrics.csv` from its own
directory, so it can run locally or on Streamlit Community Cloud without the
source dataset, an MLflow tracking database, or a model artifact. Keep the
terminal running and open the local URL printed by Streamlit.

The bundled forecast was generated once from the existing operational model
and the historical observations immediately before its forecast origin. The
bundled evaluation metrics are copied from the published results report; the
dashboard does not train, launch an experiment, or rerun the final evaluation.

## What the dashboard shows

- A clear historical-data warning and the latest complete observation.
- A fixed, precomputed 24-hour forecast after the latest bundled observation.
- Recent observed hourly energy and the following 24 model predictions, with
  the forecast period visually distinct.
- A 3-, 7-, or 14-day historical display window and a forecast-hour selector
  that highlights the selected prediction and shows its value.
- A timestamped table of the 24 predictions in kWh.
- A CSV download for the full 24-hour forecast.
- MAE and RMSE from the recorded chronological final holdout, labelled as
  previously examined and not fully blind.
- The model, units, missing-hour handling, and limitations.

The operational model was trained on all available historical data to produce
the bundled demonstration forecast. It is separate from the development
pipeline evaluated on the final holdout; the displayed holdout scores do not
evaluate that operational artifact.

The interactive controls inspect or display the existing historical forecast;
they do not run inference, change the model, retrain it, or let users select
arbitrary forecast dates. The raw dataset, MLflow tracking database, and model
artifact are not included in the dashboard deployment.

## Appearance and interaction checks

Use the **Appearance** selector in the sidebar to choose:

- **System**: use the theme currently exposed to the app by Streamlit for the
  browser. If that theme is unavailable, the app falls back to Streamlit's
  configured theme and then to Light. A theme explicitly selected in
  Streamlit's browser menu can take precedence over the operating-system
  preference.
- **Light** or **Dark**: override the detected theme for this dashboard.

To check the interface manually, switch through all three appearance options
and confirm that text, metric cards, and chart labels remain readable. Then
change the historical context and forecast lead hour; the chart highlight and
selected-hour metric should update. The CSV download should still contain the
complete 24-hour forecast.
