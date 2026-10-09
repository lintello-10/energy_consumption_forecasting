# Household Electricity Consumption Forecasting

## Objective

Forecast the next 24 hourly household electricity-consumption values using historical consumption and calendar information, without weather data.

## Data and preparation

The project uses the UCI Individual Household Electric Power Consumption dataset, covering December 2006 to November 2010. The source data contains 2,075,259 minute-level rows.

The hourly target is calculated from `Global_active_power`, measured in kW. For a complete hour, the 60 minute readings are summed and divided by 60 to obtain kWh. Hours without all 60 readings are left missing; no values are interpolated. This results in 34,085 valid hourly targets out of 34,589 time slots.

## Historical exploratory 80/20 experiment

An earlier prototype used the latest 20% of hourly time slots as a holdout, starting on 11 February 2010 at 16:00. It evaluated three methods on 258 daily windows (6,192 hourly predictions). These exploratory metrics are retained for transparency, but are **not** the final evaluation and should not be compared directly with the final 70/15/15 results below.

| Exploratory prototype | MAE (kWh) | RMSE (kWh) |
|---|---:|---:|
| Previous-day seasonal naive | 0.5227 | 0.7788 |
| Weekday-hour average, fitted on training data | 0.5020 | 0.6709 |
| Gradient boosting with calendar features and 1- and 24-hour lags | 0.4732 | 0.6329 |

The prototype gradient boosting model had the lowest scores in that experiment. The dataset, including this earlier overlapping holdout period, was examined during exploratory analysis. This experiment is not evidence of a fully blind test.

## Time-series analysis

ADF and KPSS tests were applied during exploration to the longest uninterrupted segment of valid targets in the exploratory partition's earlier 80% segment. With a constant, both tests rejected their respective null hypotheses at the 5% level. With a deterministic linear trend, ADF rejected the unit-root null, while KPSS did not reject the trend-stationarity null at the 5% level. These results are compatible with trend-stationarity for that segment, but do not prove that the full series is stationary.

The detrended ACF showed short-term dependence and recurring peaks around daily lags, with a positive association also visible at the weekly lag. These are descriptive diagnostics, not automatic model-selection rules.

## Limitations

- Results are from one chronological test period and may not generalize to other periods.
- Missing periods reduce the number of complete evaluation windows.
- The evaluation uses daily rolling origins, where recent observations are available before each new forecast.
- The stationarity tests cover one continuous training segment, not the full dataset.
- Weather and other external predictors were not used.
- The gradient boosting model is an initial candidate; further tuning should use training/validation data, not this test set.

## Reproducibility

The notebook documents data quality, target construction, calendar patterns, and time-series diagnostics. The reusable experiment and final evaluation code is in `src/energy_forecasting/`. The raw dataset is not modified.

## Final chronological 70/15/15 holdout evaluation

The data was split chronologically into training, validation, and test periods (70/15/15). Model selection used validation only. The selected gradient boosting configuration was refitted on the combined training and validation periods before evaluation on the final test period, 24 April 2010 at 17:00 through 26 November 2010 at 21:00.

The final evaluation used 193 complete daily test windows, corresponding to 4,632 hourly predictions per method. Forecasts were issued at daily rolling origins using the observations available before each origin; this is not one forecast made once for the entire test period.

| Method | MAE (kWh) | RMSE (kWh) |
|---|---:|---:|
| Previous-day seasonal naive | 0.5040 | 0.7554 |
| Weekday-hour average | 0.5108 | 0.6790 |
| Gradient boosting | 0.4765 | 0.6314 |

Gradient boosting achieved the lowest MAE and RMSE on these test windows. Compared with the weekday-hour average baseline, its MAE was approximately 6.7% lower and its RMSE approximately 7.0% lower.

The final test split was excluded from fitting and hyperparameter selection in the final pipeline. However, the dataset and an earlier overlapping holdout were examined during exploratory analysis. This is therefore a chronological holdout evaluation, not a fully blind test. These results were not used for further model tuning.