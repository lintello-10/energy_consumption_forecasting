"""Streamlit dashboard for the historical electricity forecast."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard_data import (
    build_recent_plot_series,
    load_demo_forecast,
    load_demo_metrics,
)

DASHBOARD_DIR = Path(__file__).resolve().parent
DEMO_FORECAST_PATH = DASHBOARD_DIR / "demo_forecast.csv"
DEMO_METRICS_PATH = DASHBOARD_DIR / "demo_metrics.csv"

st.set_page_config(
    page_title="Household Energy | Forecast Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME_PALETTES = {
    "light": {
        "background": "#f5f7f2",
        "surface": "#ffffff",
        "sidebar": "#142a25",
        "sidebar_text": "#eff5ec",
        "sidebar_muted": "#bdccc2",
        "text": "#14211e",
        "muted": "#596a63",
        "primary": "#176b55",
        "accent": "#d98522",
        "selected": "#142a25",
        "border": "#d8e2da",
        "grid": "rgba(70,95,82,.16)",
        "warning_background": "#fff2df",
        "warning_text": "#593b18",
        "pill_background": "#e8f2d1",
        "pill_text": "#38551d",
        "status_dot": "#65952e",
        "color_scheme": "light",
    },
    "dark": {
        "background": "#101714",
        "surface": "#1b2521",
        "sidebar": "#0b1210",
        "sidebar_text": "#edf4ef",
        "sidebar_muted": "#b3c2b9",
        "text": "#edf4ef",
        "muted": "#b3c2b9",
        "primary": "#78d6b0",
        "accent": "#ffc266",
        "selected": "#edf4ef",
        "border": "#35443d",
        "grid": "rgba(205,225,213,.14)",
        "warning_background": "#3a2d1b",
        "warning_text": "#ffe0ad",
        "pill_background": "#27371e",
        "pill_text": "#d8efaa",
        "status_dot": "#c0e56b",
        "color_scheme": "dark",
    },
}

with st.sidebar:
    st.markdown("## ⚡ Forecast Lab")
    st.caption("Household energy · historical case study")
    appearance = st.selectbox(
        "Appearance",
        ["System", "Light", "Dark"],
        index=0,
        help=(
            "System follows the theme currently active in Streamlit for "
            "this browser. Choose Light or Dark to override it."
        ),
        key="appearance",
    )

client_theme = st.context.theme.get("type")
if appearance == "Light":
    resolved_theme = "light"
elif appearance == "Dark":
    resolved_theme = "dark"
elif client_theme in THEME_PALETTES:
    resolved_theme = client_theme
else:
    configured_theme = st.get_option("theme.base")
    resolved_theme = (
        configured_theme
        if configured_theme in THEME_PALETTES
        else "light"
    )

palette = THEME_PALETTES[resolved_theme]
st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: {palette["color_scheme"]};
        --energy-ink: {palette["text"]};
        --energy-muted: {palette["muted"]};
        --energy-green: {palette["primary"]};
        --energy-amber: {palette["accent"]};
        --energy-paper: {palette["background"]};
        --energy-surface: {palette["surface"]};
        --energy-sidebar: {palette["sidebar"]};
        --energy-sidebar-text: {palette["sidebar_text"]};
        --energy-sidebar-muted: {palette["sidebar_muted"]};
        --energy-line: {palette["border"]};
        --energy-warning-bg: {palette["warning_background"]};
        --energy-warning-text: {palette["warning_text"]};
        --energy-pill-bg: {palette["pill_background"]};
        --energy-pill-text: {palette["pill_text"]};
        --energy-status-dot: {palette["status_dot"]};
    }}
    .stApp {{
        background: var(--energy-paper);
        color: var(--energy-ink);
    }}
    [data-testid="stHeader"] {{
        background: var(--energy-paper);
    }}
    [data-testid="stSidebar"] {{
        background: var(--energy-sidebar);
    }}
    [data-testid="stSidebar"] * {{
        color: var(--energy-sidebar-text);
    }}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: var(--energy-sidebar-muted);
    }}
    .block-container {{
        max-width: 1320px;
        padding-top: 4rem;
        padding-bottom: 4rem;
    }}
    .eyebrow {{
        color: var(--energy-green);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }}
    .hero-title {{
        color: var(--energy-ink);
        font-size: clamp(2rem, 4vw, 3.35rem);
        font-weight: 780;
        letter-spacing: -.045em;
        line-height: 1.04;
        margin: 0;
    }}
    .hero-copy {{
        color: var(--energy-muted);
        font-size: 1.04rem;
        max-width: 54rem;
        line-height: 1.65;
        margin-top: .85rem;
    }}
    .status-pill {{
        display: inline-flex;
        align-items: center;
        gap: .5rem;
        border: 1px solid var(--energy-line);
        border-radius: 999px;
        background: var(--energy-pill-bg);
        color: var(--energy-pill-text);
        font-size: .78rem;
        font-weight: 750;
        padding: .45rem .8rem;
        margin: .9rem 0 1.6rem;
    }}
    .status-dot {{
        width: .48rem;
        height: .48rem;
        background: var(--energy-status-dot);
        border-radius: 50%;
    }}
    .section-kicker {{
        color: var(--energy-green);
        font-size: .76rem;
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin: 1.35rem 0 .4rem;
    }}
    .surface-card {{
        border: 1px solid var(--energy-line);
        border-radius: 1rem;
        background: var(--energy-surface);
        box-shadow: 0 12px 35px rgba(25,55,43,.055);
        padding: 1.15rem 1.25rem;
    }}
    .metric-label {{
        color: var(--energy-muted);
        font-size: .82rem;
        font-weight: 650;
    }}
    .metric-value {{
        color: var(--energy-ink);
        font-size: clamp(1.2rem, 2vw, 1.75rem);
        font-weight: 760;
        letter-spacing: -.035em;
        margin-top: .3rem;
    }}
    .metric-note {{
        color: var(--energy-green);
        font-size: .78rem;
        font-weight: 650;
        margin-top: .25rem;
    }}
    .warning-card {{
        border-left: 4px solid var(--energy-amber);
        border-radius: .35rem .85rem .85rem .35rem;
        background: var(--energy-warning-bg);
        color: var(--energy-warning-text);
        line-height: 1.55;
        padding: .9rem 1rem;
        margin: .5rem 0 1rem;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid var(--energy-line);
        border-radius: .8rem;
        overflow: hidden;
    }}
    div[data-testid="stMetric"] {{
        border: 1px solid var(--energy-line);
        border-radius: .9rem;
        background: var(--energy-surface);
        box-shadow: 0 8px 22px rgba(0,0,0,.08);
        padding: 1rem 1.15rem;
    }}
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"],
    div[data-testid="stMetric"] [data-testid="stMetricValue"],
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
        color: var(--energy-ink);
    }}
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    h1, h2, h3, h4, label {{
        color: var(--energy-ink);
    }}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{
        color: var(--energy-muted);
    }}
    [data-testid="stDataFrame"] {{
        border-color: var(--energy-line);
    }}
    .stButton button, .stDownloadButton button {{
        border-radius: .65rem;
        font-weight: 700;
    }}
    @media (max-width: 700px) {{
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, ttl=3600)
def get_forecast_data(
    demo_forecast_path: str,
    demo_forecast_mtime_ns: int,
) -> tuple[pd.Series, pd.Series]:
    """Cache the bundled historical demonstration until its input changes."""
    del demo_forecast_mtime_ns
    return load_demo_forecast(demo_forecast_path)


@st.cache_data(show_spinner=False, ttl=300)
def get_evaluation_metrics(
    demo_metrics_path: str,
    demo_metrics_mtime_ns: int,
) -> pd.DataFrame:
    """Cache the bundled evaluation summary until its input changes."""
    del demo_metrics_mtime_ns
    return load_demo_metrics(demo_metrics_path)


def render_header() -> None:
    """Render the product header and historical-data disclosure."""
    st.markdown('<div class="eyebrow">Forecast lab · UCI household power</div>', unsafe_allow_html=True)
    st.markdown(
        '<h1 class="hero-title">Energy, hour by hour.</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-copy">Explore a 24-hour household energy forecast, '
        'inspect the historical evaluation, and understand how the model '
        'was built.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="status-pill"><span class="status-dot"></span>'
        'HISTORICAL DEMONSTRATION · DATA THROUGH NOV 2010</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="warning-card"><strong>Historical data, not a live forecast.</strong> '
        'The UCI dataset ends in November 2010. Dates and predictions shown '
        'here describe that historical series only.</div>',
        unsafe_allow_html=True,
    )


def render_forecast_page() -> None:
    """Render the interactive forecast view."""
    st.markdown('<div class="section-kicker">01 · Forecast explorer</div>', unsafe_allow_html=True)
    st.header("The next 24 hours")
    st.write(
        "Select a forecast lead and a historical context window from the "
        "sidebar. This is a fixed historical example bundled with the app; "
        "the controls explore it but do not run inference or retrain a model."
    )
    st.info(
        "The displayed series is a precomputed demonstration from November "
        "2010. The hosted app does not load the source dataset or generate "
        "new predictions."
    )

    try:
        demo_mtime_ns = DEMO_FORECAST_PATH.stat().st_mtime_ns
        with st.spinner("Loading the bundled historical demonstration..."):
            target, predictions = get_forecast_data(
                str(DEMO_FORECAST_PATH),
                demo_mtime_ns,
            )
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        st.error(f"Unable to load the bundled forecast demonstration: {exc}")
        return

    try:
        default_history, predictions = build_recent_plot_series(target, predictions)
    except (TypeError, ValueError) as exc:
        st.error(f"The bundled forecast demonstration is invalid: {exc}")
        return
    last_observation = default_history.index[-1]
    forecast_origin = predictions.index[0]

    st.sidebar.markdown("### Explore the forecast")
    history_days = st.sidebar.select_slider(
        "Historical context",
        options=[3, 7, 14],
        value=7,
        format_func=lambda days: f"{days} days",
        key="history_days",
    )
    lead_hour = st.sidebar.slider(
        "Forecast lead hour",
        min_value=1,
        max_value=24,
        value=1,
        key="lead_hour",
        help="Move the selector to highlight a forecast point.",
    )

    recent_history, predictions = build_recent_plot_series(
        target,
        predictions,
        history_hours=history_days * 24,
    )
    selected_timestamp = predictions.index[lead_hour - 1]
    selected_value = float(predictions.iloc[lead_hour - 1])
    peak_timestamp = predictions.idxmax()

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Forecast origin",
        last_observation.strftime("%d %b %Y · %H:%M"),
    )
    metric_columns[1].metric(
        "Energy over 24 h",
        f"{predictions.sum():.2f} kWh",
    )
    metric_columns[2].metric(
        "Peak forecast",
        f"{predictions.max():.2f} kWh",
        peak_timestamp.strftime("%d %b · %H:%M"),
    )
    metric_columns[3].metric(
        f"Selected · hour {lead_hour}",
        f"{selected_value:.3f} kWh",
        selected_timestamp.strftime("%a · %d %b · %H:%M"),
    )

    st.caption(
        f"Latest complete observation: {last_observation:%A, %d %B %Y at %H:%M}. "
        f"Forecast: {predictions.index[0]:%d %b %H:%M} — "
        f"{predictions.index[-1]:%d %b %H:%M}."
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=recent_history.index,
            y=recent_history,
            mode="lines",
            name="Observed · kWh",
            connectgaps=False,
            line={"color": palette["primary"], "width": 2.4},
            hovertemplate="%{x|%a, %d %b %Y · %H:%M}<br>"
            "Observed: %{y:.3f} kWh<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions,
            mode="lines+markers",
            name="Forecast · kWh",
            line={"color": palette["accent"], "width": 2.8, "dash": "dash"},
            marker={"size": 6, "color": palette["accent"]},
            hovertemplate="%{x|%a, %d %b %Y · %H:%M}<br>"
            "Forecast: %{y:.3f} kWh<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[selected_timestamp],
            y=[selected_value],
            mode="markers",
            name=f"Selected · hour {lead_hour}",
            marker={
                "color": palette["selected"],
                "size": 14,
                "line": {"color": palette["surface"], "width": 2},
                "symbol": "circle",
            },
            hovertemplate="%{x|%a, %d %b %Y · %H:%M}<br>"
            "Selected forecast: %{y:.3f} kWh<extra></extra>",
        )
    )
    figure.add_vrect(
        x0=forecast_origin,
        x1=predictions.index[-1],
        fillcolor=palette["accent"],
        opacity=0.08,
        line_width=0,
        layer="below",
    )
    figure.add_vline(
        x=forecast_origin,
        line_dash="dot",
        line_color="#60736a",
        annotation_text="Forecast starts",
        annotation_position="top left",
    )
    figure.update_layout(
        title={"text": "Observed history and forecast", "x": 0.02},
        xaxis_title="Date and time",
        yaxis_title="Hourly energy (kWh)",
        hovermode="x unified",
        height=470,
        margin={"l": 20, "r": 20, "t": 65, "b": 25},
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        font={"color": palette["text"]},
    )
    figure.update_xaxes(
        showgrid=True,
        gridcolor=palette["grid"],
        rangeslider_visible=False,
    )
    figure.update_yaxes(
        rangemode="tozero",
        showgrid=True,
        gridcolor=palette["grid"],
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )

    left_column, right_column = st.columns([1.5, 1], gap="large")
    with left_column:
        st.subheader("Hourly forecast")
        forecast_table = pd.DataFrame(
            {
                "Date and time": predictions.index.strftime(
                    "%a, %d %b %Y · %H:%M"
                ),
                "Forecast energy (kWh)": predictions.to_numpy(dtype=float),
            }
        )
        st.dataframe(
            forecast_table,
            hide_index=True,
            use_container_width=True,
            height=430,
            column_config={
                "Forecast energy (kWh)": st.column_config.NumberColumn(
                    "Forecast energy (kWh)",
                    format="%.3f",
                ),
            },
        )
    with right_column:
        st.subheader("Selected hour")
        st.markdown(
            f"""
            <div class="surface-card">
                <div class="metric-label">FORECAST LEAD · {lead_hour} HOUR(S)</div>
                <div class="metric-value">{selected_value:.3f} kWh</div>
                <div class="metric-note">{selected_timestamp:%A, %d %B %Y · %H:%M}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.download_button(
            "Download the 24-hour forecast",
            data=forecast_table.to_csv(index=False),
            file_name="household_energy_forecast_24h.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(
            "Hourly energy is reported in kWh. The source has no observations "
            "after the forecast origin."
        )


def render_evaluation_page() -> None:
    """Render the recorded final-holdout metrics with their caveat."""
    st.markdown('<div class="section-kicker">02 · Model evaluation</div>', unsafe_allow_html=True)
    st.header("How did the methods compare?")
    st.markdown(
        '<div class="warning-card"><strong>Read this result carefully.</strong> '
        'The final chronological holdout was previously examined during '
        'exploratory analysis, so it is not a fully blind test. These metrics '
        'evaluate the selected development pipeline, not the operational '
        'model refitted on all available history.</div>',
        unsafe_allow_html=True,
    )

    try:
        demo_mtime_ns = DEMO_METRICS_PATH.stat().st_mtime_ns
        with st.spinner("Reading the bundled evaluation summary..."):
            metrics = get_evaluation_metrics(
                str(DEMO_METRICS_PATH),
                demo_mtime_ns,
            )
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Unable to load the bundled evaluation summary: {exc}")
        return

    gradient_boosting = metrics.loc[
        metrics["Method"] == "Gradient boosting"
    ]
    if not gradient_boosting.empty:
        mae = float(gradient_boosting.iloc[0]["MAE (kWh)"])
        rmse = float(gradient_boosting.iloc[0]["RMSE (kWh)"])
        windows = int(gradient_boosting.iloc[0]["Daily windows"])
        metric_columns = st.columns(3)
        metric_columns[0].metric("Gradient boosting · MAE", f"{mae:.4f} kWh")
        metric_columns[1].metric("Gradient boosting · RMSE", f"{rmse:.4f} kWh")
        metric_columns[2].metric("Complete daily windows", f"{windows}")

    st.subheader("Final chronological holdout")
    st.dataframe(
        metrics,
        hide_index=True,
        use_container_width=True,
        column_config={
            "MAE (kWh)": st.column_config.NumberColumn(format="%.4f"),
            "RMSE (kWh)": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        "The 193 daily rolling-origin windows contain 4,632 hourly forecasts "
        "per method. These previously recorded results are bundled with the "
        "app and are not recalculated at runtime."
    )


def render_methodology_page() -> None:
    """Explain the dataset, model design, and methodological boundaries."""
    st.markdown('<div class="section-kicker">03 · Methodology</div>', unsafe_allow_html=True)
    st.header("What is behind this forecast?")
    st.write(
        "A transparent forecasting workflow built around hourly household "
        "energy data, chronological validation, and explicit handling of "
        "missing measurements."
    )

    columns = st.columns(3, gap="medium")
    with columns[0]:
        st.markdown("#### 01 · Target")
        st.write(
            "Minute-level `Global_active_power` is measured in kW. For each "
            "complete hour, its 60 valid minute readings are summed and "
            "divided by 60 to obtain hourly energy in kWh."
        )
    with columns[1]:
        st.markdown("#### 02 · Features")
        st.write(
            "The model uses cyclical hour-of-day and weekday features, "
            "together with 1-hour and 24-hour consumption lags. No weather, "
            "tariff, or occupancy data is included."
        )
    with columns[2]:
        st.markdown("#### 03 · Forecast")
        st.write(
            "A histogram-based gradient-boosting model predicts recursively: "
            "each generated value becomes available as a lag for later "
            "hours in the 24-hour forecast."
        )

    st.subheader("Evaluation design")
    st.write(
        "Model development used chronological training and validation "
        "periods. The selected configuration was evaluated on a later "
        "chronological holdout using shared complete daily forecast windows. "
        "The holdout was not used for fitting or hyperparameter selection, "
        "but its period had already been examined during exploration."
    )
    st.subheader("Important limitations")
    st.markdown(
        """
        - **Historical coverage:** the source data runs from December 2006 to
          November 2010. The displayed forecast is a historical demonstration,
          not a current prediction.
        - **Hosted demonstration:** the app displays a bundled 14-day sample
          and precomputed 24-hour forecast; it does not access the raw dataset,
          load an MLflow model, or calculate new predictions.
        - **Incomplete measurements:** partial or invalid hours remain
          missing. No interpolation or silent imputation is performed.
        - **Evaluation scope:** results come from one chronological period and
          daily rolling-origin windows; they do not guarantee generalization.
        - **Exposure disclosure:** the final holdout was previously viewed
          during exploratory analysis, so it is not fully blind.
        - **Operational artifact:** the displayed forecast comes from a model
          refitted on all available historical data. The holdout metrics do
          not measure this refitted model.
        """
    )


with st.sidebar:
    page = st.radio(
        "Navigate",
        ["Forecast", "Evaluation", "Methodology"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("UCI source period")
    st.markdown("**Dec 2006 — Nov 2010**")
    st.caption("24-hour horizon · hourly kWh · no weather data")

render_header()

if page == "Forecast":
    render_forecast_page()
elif page == "Evaluation":
    render_evaluation_page()
else:
    render_methodology_page()

st.divider()
st.caption(
    "Household Energy Forecasting · Historical research demonstration · "
    "No current-consumption claims"
)
