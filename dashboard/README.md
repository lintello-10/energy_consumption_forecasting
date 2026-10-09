# Historical Forecast Dashboard

The Streamlit app visualizes a 24-hour forecast made from the latest complete
hour in the UCI household electricity dataset. The source data ends in
November 2010, so this is a historical demonstration, not a forecast of
current consumption.

## Install and run

From the project root, with the project virtual environment activated:

```bash
python -m pip install -e ".[dashboard]"
export FORECAST_MODEL_URI="runs:/<run-id>/model"
python -m streamlit run dashboard/app.py
```

Replace `<run-id>` with the URI printed by
`python -m energy_forecasting.train_model`. The dashboard extra installs
Streamlit, Plotly, and MLflow. Keep the terminal running and open the local
URL printed by Streamlit.

The dashboard also reads the existing final-test metrics from the local
MLflow tracking store. It does not train a model, launch an experiment, or
rerun the final evaluation. Only use trusted MLflow model artifacts created
by this project.

## What the dashboard shows

- A clear historical-data warning and the latest complete observation.
- Recent observed hourly energy and the following 24 model predictions, with
  the forecast period visually distinct.
- A 3-, 7-, or 14-day historical display window and a forecast-hour selector
  that highlights the selected prediction and shows its value.
- A timestamped table of the 24 predictions in kWh.
- A CSV download for the full 24-hour forecast.
- MAE and RMSE from the recorded chronological final holdout, labelled as
  previously examined and not fully blind.
- The model, units, missing-hour handling, and limitations.

The operational model is trained on all available historical data to produce
the demonstration forecast. It is separate from the development pipeline
evaluated on the final holdout; the displayed holdout scores do not evaluate
that operational artifact.

The interactive controls inspect or display the existing historical forecast;
they do not change the model, retrain it, or let users select arbitrary
forecast dates.

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
