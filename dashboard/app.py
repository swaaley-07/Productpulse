
from pathlib import Path
import re
from textwrap import dedent

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "sales_longevity_with_prices.parquet"
MODEL_PATH = ROOT / "models" / "sales_longevity_final_candidate.joblib"
RAW_SALES_PATH = ROOT / "data" / "raw" / "sales_train_evaluation.csv"
CALENDAR_PATH = ROOT / "data" / "raw" / "calendar.csv"

REPORT_PATH = ROOT / "reports" / "final_candidate_evaluation.txt"
TEMPORAL_REPORT = ROOT / "reports" / "temporal_model_comparison.csv"
MODEL_REPORT = ROOT / "reports" / "model_comparison.csv"
FEATURE_IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.csv"
ABLATION_PATH = ROOT / "reports" / "feature_ablation.csv"

CHART_DIR = ROOT / "visualisation"

TARGET_UNITS = 20
HORIZON_DAYS = 90

DEFAULT_FEATURES = [
    "units_last_28d",
    "units_last_90d",
    "active_days_last_28d",
    "active_days_last_90d",
    "sales_change_recent_vs_previous_28d",
    "price_at_prediction",
    "price_change_4w_pct",
    "category",
]

STORE_NAMES = {
    "CA_1": "California · Store 1",
    "CA_2": "California · Store 2",
    "CA_3": "California · Store 3",
    "CA_4": "California · Store 4",
    "TX_1": "Texas · Store 1",
    "TX_2": "Texas · Store 2",
    "TX_3": "Texas · Store 3",
    "WI_1": "Wisconsin · Store 1",
    "WI_2": "Wisconsin · Store 2",
    "WI_3": "Wisconsin · Store 3",
}

CATEGORY_NAMES = {
    "FOODS": "Food & Grocery",
    "HOBBIES": "Hobbies & Leisure Products",
    "HOUSEHOLD": "Household Products",
}

st.set_page_config(
    page_title="ProductPulse | Retail Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DARK NAVY THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --background: #0b1426;
        --background-secondary: #101d33;
        --surface: #15243b;
        --surface-hover: #1b2e4b;
        --border: #293e5e;
        --text: #edf4ff;
        --muted: #a0b6d3;
        --blue: #6da8ff;
        --blue-dark: #367bd5;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #0b1426;
        color: #edf4ff;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 20, 38, 0.96);
    }

    [data-testid="stAppViewContainer"] {
        background: #0b1426;
    }

    [data-testid="stSidebar"] {
        background: #101d33;
        border-right: 1px solid #293e5e;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #a0b6d3;
        line-height: 1.75;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #dce8f8 !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Manrope', sans-serif !important;
        color: #edf4ff !important;
        letter-spacing: -0.025em;
    }

    p {
        color: #bdcce0;
        line-height: 1.75;
    }

    .block-container {
        max-width: 1600px;
        padding: 2rem 3rem 4rem 3rem;
    }

    [data-testid="stVerticalBlock"] {
        gap: 1.2rem;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 1.25rem;
        align-items: stretch;
    }

    .pp-eyebrow {
        color: #7eb3ff;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    /* Main hero */
    .pp-hero {
        background: linear-gradient(
            125deg,
            #152b49 0%,
            #172743 55%,
            #202b4b 100%
        );
        border: 1px solid #2c4568;
        border-radius: 22px;
        padding: 2.5rem 2.7rem;
        margin-bottom: 2.4rem;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.15);
    }

    .pp-hero h1 {
        font-size: clamp(2.3rem, 4vw, 3.5rem);
        color: #f0f6ff !important;
        line-height: 1.2;
        letter-spacing: -0.055em;
        margin: 0.5rem 0 1rem 0;
    }

    .pp-hero p {
        color: #c2d4ed;
        font-size: 1rem;
        max-width: 900px;
        margin-bottom: 0;
    }

    .pp-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.3rem;
    }

    .pp-tag {
        display: inline-block;
        background: #203652;
        border: 1px solid #355174;
        color: #d9e9ff;
        border-radius: 999px;
        padding: 0.42rem 0.8rem;
        font-size: 0.78rem;
    }

    /* Custom metric cards */
    .pp-card {
        background: #15243b;
        border: 1px solid #293e5e;
        border-radius: 17px;
        padding: 1.4rem;
        min-height: 150px;
        height: 100%;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.09);
    }

    .pp-label {
        color: #a8bfdf;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        line-height: 1.5;
    }

    .pp-value {
        color: #f0f6ff;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.25;
        margin: 0.7rem 0 0.5rem 0;
    }

    .pp-help {
        color: #a0b6d3;
        font-size: 0.8rem;
        line-height: 1.55;
    }

    .pp-panel {
        background: #15243b;
        border: 1px solid #293e5e;
        border-radius: 18px;
        padding: 1.6rem;
        box-shadow: 0 5px 18px rgba(0, 0, 0, 0.09);
    }

    .pp-muted {
        color: #a0b6d3;
        font-size: 0.94rem;
        line-height: 1.8;
    }

    .pp-kicker {
        color: #a8bfdf;
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .pp-big-number {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.3;
        color: #edf4ff;
        margin-top: 0.6rem;
        margin-bottom: 0.4rem;
    }

    .pp-small {
        font-size: 0.82rem;
        color: #a0b6d3;
        line-height: 1.6;
    }

    .pp-section {
        margin-top: 2.4rem;
        margin-bottom: 1.2rem;
    }

    /* Native Streamlit metrics */
    div[data-testid="stMetric"] {
        background: #15243b;
        border: 1px solid #293e5e;
        border-radius: 15px;
        padding: 1.15rem 1.3rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
        min-height: 108px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a8bfdf !important;
    }

    div[data-testid="stMetricValue"] {
        color: #edf4ff !important;
    }

    div[data-testid="stMetricDelta"] {
        color: #83c9b1 !important;
    }

    /* Data tables */
    div[data-testid="stDataFrame"] {
        background: #15243b;
        border: 1px solid #293e5e;
        border-radius: 13px;
        overflow: hidden;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #15243b;
        border: 1px solid #293e5e;
        border-radius: 13px;
    }

    [data-testid="stExpander"] summary p {
        color: #edf4ff !important;
    }

    /* Inputs */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div {
        background: #15243b;
        border-color: #355174;
        border-radius: 9px;
        color: #edf4ff;
    }

    [data-baseweb="select"] input,
    [data-baseweb="input"] input {
        color: #edf4ff !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"] {
        background: #15243b;
    }

    [role="option"] {
        background: #15243b;
        color: #edf4ff;
    }

    [role="option"]:hover {
        background: #203652;
    }

    /* Buttons */
    .stButton button,
    .stDownloadButton button {
        border-radius: 10px;
        min-height: 2.5rem;
    }

    .stButton button[kind="primary"] {
        background: #367bd5;
        border-color: #367bd5;
        color: #ffffff;
    }

    .stButton button[kind="secondary"],
    .stDownloadButton button {
        background: #203652;
        border-color: #355174;
        color: #edf4ff;
    }

    .stButton button:hover,
    .stDownloadButton button:hover {
        border-color: #7eb3ff;
        color: #ffffff;
    }

    .stProgress > div > div {
        background: #6da8ff;
    }

    /* Radio buttons and checkboxes */
    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label,
    [data-testid="stMultiSelect"] label {
        color: #dce8f8 !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: #293e5e;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Tabs */
    [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    [data-baseweb="tab"] {
        color: #a8bfdf;
    }

    [aria-selected="true"][data-baseweb="tab"] {
        color: #7eb3ff !important;
    }

    /* Code blocks and captions */
    [data-testid="stCaptionContainer"] {
        color: #a0b6d3;
    }

    [data-testid="stCode"] {
        border: 1px solid #293e5e;
        border-radius: 10px;
    }

    @media (max-width: 900px) {
        .block-container {
            padding: 1.2rem 1rem 3rem 1rem;
        }

        .pp-hero {
            padding: 1.5rem;
        }

        .pp-card {
            padding: 1.1rem;
        }

        .pp-value {
            font-size: 1.7rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(show_spinner="Loading processed sales data...")
def load_data(path):
    frame = pd.read_parquet(path)

    if "prediction_date" in frame.columns:
        frame["prediction_date"] = pd.to_datetime(
            frame["prediction_date"], errors="coerce"
        )

    return frame


@st.cache_resource(show_spinner="Loading trained model...")
def load_model(path):
    return joblib.load(path)


@st.cache_data(show_spinner="Loading historical sales...")
def load_history(sales_path, calendar_path):
    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path)
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce")
    return sales, calendar


def find_column(columns, candidates):
    return next((col for col in candidates if col in columns), None)


def fmt(value, decimals=0):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return "N/A"


def metric_card(label, value, note):
    html = f"""
    <div class="pp-card">
        <div class="pp-label">{label}</div>
        <div class="pp-value">{value}</div>
        <div class="pp-help">{note}</div>
    </div>
    """
    st.markdown(dedent(html), unsafe_allow_html=True)


def styled_plot(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#d6e4f7"),
        margin=dict(l=18, r=18, t=45, b=18),
        hoverlabel=dict(
            bgcolor="#15243b",
            bordercolor="#355174",
            font_color="#edf4ff",
        ),
    )

    if height:
        fig.update_layout(height=height)

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#293e5e",
        zeroline=False,
        linecolor="#355174",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#293e5e",
        zeroline=False,
        linecolor="#355174",
    )

    return fig


def get_probability(model, row, features):
    model_input = pd.DataFrame(
        [{feature: row.get(feature, np.nan) for feature in features}],
        columns=features,
    )

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_input)[0]
        classes = list(
            getattr(model, "classes_", range(len(probabilities)))
        )

        if 1 not in classes:
            raise ValueError(
                "Class 1 was not found in the model. "
                "Check the target encoding."
            )

        probability = float(probabilities[classes.index(1)])
        return probability, int(probability >= 0.5)

    return None, int(model.predict(model_input)[0])


def history_chart(item_id, store_id, prediction_date, sales, calendar):
    if sales is None or calendar is None:
        return None, "Raw sales history files are unavailable."

    if not {"item_id", "store_id"}.issubset(sales.columns):
        return None, "Expected item_id and store_id columns were not found."

    match = sales[
        (sales["item_id"].astype(str) == str(item_id))
        & (sales["store_id"].astype(str) == str(store_id))
    ]

    if match.empty:
        return None, "No daily sales series found for this product and store."

    calendar_map = calendar.set_index("d")["date"].to_dict()
    sales_row = match.iloc[0]
    records = []

    for column in [c for c in sales.columns if c.startswith("d_")]:
        date = calendar_map.get(column)

        if date is None or pd.isna(date):
            continue

        date = pd.Timestamp(date)

        if date <= prediction_date:
            records.append({
                "date": date,
                "units": pd.to_numeric(
                    sales_row[column], errors="coerce"
                ),
            })

    history = pd.DataFrame(records).dropna().sort_values("date")
    history = history[
        history["date"] >= prediction_date - pd.Timedelta(days=180)
    ]

    if history.empty:
        return None, "No sales found in the 180 days before this prediction date."

    history["7-day average"] = (
        history["units"].rolling(7, min_periods=1).mean()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["units"],
            name="Daily sales",
            mode="lines",
            line=dict(color="#6da8ff", width=2),
            hovertemplate=(
                "%{x|%d %b %Y}<br>Units sold: %{y:,.0f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["7-day average"],
            name="7-day moving average",
            mode="lines",
            line=dict(color="#f0b45a", width=2, dash="dot"),
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                "7-day average: %{y:.2f}<extra></extra>"
            ),
        )
    )

    fig.add_vline(
        x=prediction_date,
        line_dash="dash",
        line_color="#b9a2ff",
        annotation_text="Prediction date",
        annotation_position="top",
    )

    fig.update_layout(
        legend=dict(orientation="h", y=1.12, x=0),
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title="Units sold",
    )

    fig.update_yaxes(rangemode="tozero")

    return styled_plot(fig, 380), None


def risk_label(probability):
    if probability >= 0.60:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"


def load_report_text(path):
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return None


def find_metric_value(text, key):
    pattern = (
        rf"(?im)^\s*{re.escape(key)}\s*:\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
    )

    match = re.search(pattern, text)

    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def render_report_table(report_text):
    rows = []
    in_table = False

    for line in report_text.splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        if all(
            word in stripped.lower()
            for word in ["precision", "recall", "f1-score", "support"]
        ):
            in_table = True
            continue

        if not in_table:
            continue

        if stripped.lower().startswith(
            ("confusion matrix", "summary metrics")
        ):
            break

        parts = stripped.split()

        if len(parts) < 4:
            continue

        try:
            precision = float(parts[-4])
            recall = float(parts[-3])
            f1 = float(parts[-2])
            support = int(float(parts[-1]))
            label = " ".join(parts[:-4])

            if (
                label
                and 0 <= precision <= 1
                and 0 <= recall <= 1
                and 0 <= f1 <= 1
            ):
                rows.append({
                    "Class / average": label,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-score": f1,
                    "Support": support,
                })

        except (ValueError, IndexError):
            continue

    return pd.DataFrame(rows)


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

missing_files = [
    path for path in (DATA_PATH, MODEL_PATH) if not path.exists()
]

if missing_files:
    st.error(
        "Required project files are missing. Keep app.py inside "
        "the dashboard folder and check these paths:"
    )
    for path in missing_files:
        st.code(str(path))
    st.stop()

try:
    data = load_data(str(DATA_PATH))
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the dataset or saved model: {exc}")
    st.stop()

if data.empty:
    st.error("The processed dataset is empty.")
    st.stop()

item_col = find_column(
    data.columns,
    ["item_id", "product_id", "product", "item"],
)

store_col = find_column(data.columns, ["store_id", "store"])

category_col = find_column(
    data.columns,
    ["category", "cat_id", "product_category"],
)

date_col = find_column(
    data.columns,
    ["prediction_date", "date"],
)

if not all([item_col, store_col, date_col]):
    st.error(
        "Required product, store, or prediction-date columns were not found."
    )
    st.write("Available columns:", list(data.columns))
    st.stop()

data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
data = data.dropna(subset=[date_col]).copy()

features = list(
    getattr(model, "feature_names_in_", DEFAULT_FEATURES)
)

missing_features = [
    feature for feature in features if feature not in data.columns
]

if missing_features:
    st.error(
        "The saved model expects columns missing from the processed dataset."
    )
    st.write(missing_features)
    st.stop()

raw_sales = None
calendar = None
history_message = None

if RAW_SALES_PATH.exists() and CALENDAR_PATH.exists():
    try:
        raw_sales, calendar = load_history(
            str(RAW_SALES_PATH),
            str(CALENDAR_PATH),
        )
    except Exception as exc:
        history_message = f"Could not load raw history: {exc}"
else:
    history_message = "Raw sales/calendar files are unavailable."


# ============================================================
# SIDEBAR AND FILTERS
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="pp-eyebrow">PRODUCTPULSE</div>
        <h2 style="margin-top:0">Sales Explorer</h2>
        <p style="color:#a0b6d3;line-height:1.7">
            Explore products, stores and demand signals.
        </p>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Product Explorer",
            "Risk Analysis",
            "Model Performance",
            "Data Explorer",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Narrow your analysis")

    filtered = data.copy()

    if category_col:
        categories = sorted(
            filtered[category_col].dropna().astype(str).unique()
        )

        category_options = {
            CATEGORY_NAMES.get(value, value): value
            for value in categories
        }

        selected_category_label = st.selectbox(
            "Product category",
            ["All categories"] + list(category_options.keys()),
        )

        if selected_category_label == "All categories":
            category = "All categories"
        else:
            category = category_options[selected_category_label]
            filtered = filtered[
                filtered[category_col].astype(str) == category
            ]

        category_display = selected_category_label
    else:
        category = "All categories"
        category_display = category

    stores = sorted(
        filtered[store_col].dropna().astype(str).unique()
    )

    store_options = {
        "All stores": None,
        **{
            STORE_NAMES.get(value, value): value
            for value in stores
        },
    }

    selected_store_label = st.selectbox(
        "Store",
        list(store_options.keys()),
    )

    store = store_options[selected_store_label]

    if store is not None:
        filtered = filtered[
            filtered[store_col].astype(str) == store
        ]

    if filtered.empty:
        st.warning("No rows match these filters.")
        st.stop()

    dates = sorted(
        pd.Timestamp(value)
        for value in filtered[date_col].dropna().unique()
    )

    prediction_date = st.selectbox(
        "Prediction date",
        dates,
        index=len(dates) - 1,
        format_func=lambda value: value.strftime("%d %b %Y"),
    )

    filtered = filtered[
        filtered[date_col] == prediction_date
    ].copy()

    products = sorted(
        filtered[item_col].dropna().astype(str).unique()
    )

    product_search = st.text_input(
        "Search product ID",
        placeholder="e.g. FOODS_1_001",
    )

    if product_search.strip():
        matches = [
            value
            for value in products
            if product_search.strip().lower() in value.lower()
        ]

        if matches:
            products = matches
        else:
            st.caption("No matching product ID found.")

    if not products:
        st.warning("No products are available for this selection.")
        st.stop()

    product = st.selectbox("Product", products)

    st.divider()

    st.caption(
        "Predictions estimate recorded sales. They do not confirm "
        "product discontinuation."
    )


# ============================================================
# SELECTED PRODUCT AND PREDICTION
# ============================================================

selected_rows = filtered[
    filtered[item_col].astype(str) == product
]

if selected_rows.empty:
    st.error("No row exists for the selected product and filters.")
    st.stop()

selected = selected_rows.iloc[0]

try:
    probability, predicted_class = get_probability(
        model, selected, features
    )
    prediction_error = None
except Exception as exc:
    probability = None
    predicted_class = None
    prediction_error = str(exc)

units_28 = pd.to_numeric(
    selected.get("units_last_28d", np.nan), errors="coerce"
)
units_90 = pd.to_numeric(
    selected.get("units_last_90d", np.nan), errors="coerce"
)
active_90 = pd.to_numeric(
    selected.get("active_days_last_90d", np.nan), errors="coerce"
)
price = pd.to_numeric(
    selected.get("price_at_prediction", np.nan), errors="coerce"
)

avg_daily = units_28 / 28 if pd.notna(units_28) else np.nan
active_pct = active_90 / 90 * 100 if pd.notna(active_90) else np.nan


# ============================================================
# HERO BANNER
# ============================================================

hero_html = f"""
<div class="pp-hero">
    <div class="pp-eyebrow">RETAIL INTELLIGENCE · MACHINE LEARNING</div>
    <h1>ProductPulse <span style="color:#7eb3ff">↗</span></h1>
    <p>
        Explore product-level sales patterns and estimate the likelihood
        of fewer than {TARGET_UNITS} units being sold over the following
        {HORIZON_DAYS} days.
    </p>
    <div class="pp-tags">
        <span class="pp-tag">{category_display}</span>
        <span class="pp-tag">{selected_store_label}</span>
        <span class="pp-tag">{product}</span>
        <span class="pp-tag">{prediction_date:%d %b %Y}</span>
    </div>
</div>
"""

st.markdown(dedent(hero_html), unsafe_allow_html=True)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="pp-eyebrow">01 · Product snapshot</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4, gap="large")

    with c1:
        metric_card(
            "Units · Last 28 days",
            fmt(units_28),
            "Recent demand",
        )

    with c2:
        metric_card(
            "Units · Last 90 days",
            fmt(units_90),
            "Quarter-length demand",
        )

    with c3:
        metric_card(
            "Active selling days",
            fmt(active_90),
            "Days with sales in the last 90 days",
        )

    with c4:
        metric_card(
            "Price at prediction",
            fmt(price, 2),
            "Recorded price; currency not assumed",
        )

    st.markdown(
        '<div class="pp-section"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="pp-eyebrow">02 · 90-day model outlook</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        if prediction_error:
            st.error(f"Prediction failed: {prediction_error}")

        elif probability is None:
            st.warning(
                "The model did not provide a probability estimate."
            )
            st.write(f"Predicted class: {predicted_class}")

        else:
            if predicted_class == 1:
                title = "Higher low-future-sales signal"
                description = (
                    f"The model estimates that this product-store "
                    f"combination may sell fewer than {TARGET_UNITS} "
                    f"units in the next {HORIZON_DAYS} days."
                )
                panel_background = "#30252e"
                panel_border = "#65404a"
                accent = "#ffaaa6"
            else:
                title = "Stronger future-sales signal"
                description = (
                    f"The model estimates at least {TARGET_UNITS} "
                    f"units sold in the next {HORIZON_DAYS} days."
                )
                panel_background = "#142f35"
                panel_border = "#286052"
                accent = "#83d9bb"

            prediction_html = f"""
            <div style="
                background:{panel_background};
                border:1px solid {panel_border};
                border-radius:19px;
                padding:2rem;
                min-height:285px;
            ">
                <div class="pp-eyebrow" style="color:{accent}">
                    MODEL PREDICTION
                </div>
                <h2 style="color:#f0f6ff!important;margin-top:0.6rem">
                    {title}
                </h2>
                <p style="color:#d0dceb;line-height:1.8">
                    {description}
                </p>
                <div style="color:#b1c4da;font-size:0.88rem;margin-top:1.4rem">
                    Estimated probability of fewer than {TARGET_UNITS} future units
                </div>
                <div style="font-size:3rem;font-weight:800;color:{accent};margin-top:0.3rem">
                    {probability:.1%}
                </div>
            </div>
            """

            st.markdown(
                dedent(prediction_html),
                unsafe_allow_html=True,
            )

            st.progress(
                float(np.clip(probability, 0, 1)),
                text="Low-future-sales probability",
            )

            st.caption(
                "This is a model estimate, not a guarantee. Low future sales "
                "does not prove discontinuation or inventory unavailability."
            )

    with right:
        st.markdown("### Demand signals")

        st.metric(
            "Average units per day · last 28 days",
            fmt(avg_daily, 2),
        )

        st.metric(
            "Selling-day frequency · last 90 days",
            f"{active_pct:.1f}%" if pd.notna(active_pct) else "N/A",
        )

        with st.expander("Inspect model input features"):
            for feature in features:
                st.write(
                    f"**{feature}:** {selected.get(feature, 'Missing')}"
                )

    st.markdown(
        '<div class="pp-section"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="pp-eyebrow">03 · Sales history</div>',
        unsafe_allow_html=True,
    )

    if store is None:
        st.info(
            "Select a specific store in the sidebar to view daily sales history."
        )
    else:
        fig, message = history_chart(
            product, store, prediction_date, raw_sales, calendar
        )

        if fig is not None:
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False},
            )
        else:
            st.info(
                message or history_message or "Sales history unavailable."
            )

    st.markdown(
        '<div class="pp-section"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="pp-eyebrow">04 · Products to investigate</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Products with the strongest low-sales signals")

    st.caption(
        "Ranked by estimated low-future-sales probability for the selected date."
    )

    ranked = filtered.copy()

    try:
        probabilities = model.predict_proba(ranked[features])
        classes = list(
            getattr(model, "classes_", range(probabilities.shape[1]))
        )

        if 1 not in classes:
            raise ValueError("Class 1 was not found in the saved model.")

        ranked["Low-sales probability"] = probabilities[
            :, classes.index(1)
        ]

        ranked = ranked.sort_values(
            "Low-sales probability", ascending=False
        )

        ranked["Low-sales probability"] = ranked[
            "Low-sales probability"
        ].map(lambda value: f"{value:.1%}")

        display_cols = [
            col for col in [
                item_col,
                store_col,
                "units_last_28d",
                "units_last_90d",
                "active_days_last_90d",
                "Low-sales probability",
            ]
            if col in ranked.columns
        ]

        display_data = ranked[display_cols].rename(
            columns={
                "units_last_28d": "Sales · 28d",
                "units_last_90d": "Sales · 90d",
                "active_days_last_90d": "Active days · 90d",
            }
        )

        st.dataframe(
            display_data.head(10),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download ranked products (CSV)",
            ranked.to_csv(index=False).encode("utf-8"),
            "productpulse_ranked_products.csv",
            "text/csv",
        )

    except Exception as exc:
        st.info(f"Product ranking unavailable: {exc}")


# ============================================================
# PRODUCT EXPLORER
# ============================================================

elif page == "Product Explorer":

    st.markdown(
        '<div class="pp-eyebrow">Product details</div>',
        unsafe_allow_html=True,
    )

    st.title("Product Explorer")

    st.markdown(
        """
        <p class="pp-muted">
            Explore the selected product's historical sales and the
            variables used by the machine-learning model.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Sales history")

    if store is None:
        st.info(
            "Select a specific store in the sidebar to view daily sales history."
        )
    else:
        fig, message = history_chart(
            product, store, prediction_date, raw_sales, calendar
        )

        if fig is not None:
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False},
            )
        else:
            st.info(
                message or history_message or "Sales history unavailable."
            )

    st.markdown(
        '<div class="pp-section"></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("### Sales and activity")
        st.metric("Units in last 28 days", fmt(units_28))
        st.metric("Units in last 90 days", fmt(units_90))
        st.metric("Active days in last 90 days", fmt(active_90))
        st.metric("Price at prediction", fmt(price, 2))

    with right:
        st.markdown("### Model input features")

        feature_data = pd.DataFrame({
            "Feature": features,
            "Value": [
                selected.get(feature, np.nan)
                for feature in features
            ],
        })

        st.dataframe(
            feature_data,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These are model inputs, not proof that any one variable "
            "causes future sales to rise or fall."
        )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.markdown(
        '<div class="pp-eyebrow">Portfolio-level analysis</div>',
        unsafe_allow_html=True,
    )

    st.title("Risk Analysis")

    st.markdown(
        """
        <p class="pp-muted">
            Identify product-store combinations with higher estimated
            probabilities of low future sales.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Risk bands are exploratory display groups, not validated business "
        "decision thresholds. A high score does not confirm discontinuation."
    )

    try:
        probabilities = model.predict_proba(filtered[features])
        classes = list(
            getattr(model, "classes_", range(probabilities.shape[1]))
        )

        if 1 not in classes:
            raise ValueError("Class 1 was not found in the saved model.")

        risk = filtered.copy()
        risk["Low-sales probability"] = probabilities[
            :, classes.index(1)
        ]
        risk["Risk level"] = risk[
            "Low-sales probability"
        ].map(risk_label)
        risk["Low-sales probability (%)"] = (
            risk["Low-sales probability"] * 100
        ).round(1)

        st.markdown("### Filter risk levels")

        chosen = st.multiselect(
            "Risk level",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )

        risk = risk[
            risk["Risk level"].isin(chosen)
        ].sort_values(
            "Low-sales probability", ascending=False
        )

        st.markdown(
            '<div class="pp-section"></div>',
            unsafe_allow_html=True,
        )

        a, b, c = st.columns(3, gap="large")

        a.metric("Product-store pairs", f"{len(risk):,}")

        b.metric(
            "High-risk pairs",
            f"{(risk['Risk level'] == 'High').sum():,}",
        )

        c.metric(
            "Average low-sales probability",
            f"{risk['Low-sales probability'].mean():.1%}"
            if not risk.empty else "N/A",
        )

        st.markdown(
            '<div class="pp-section"></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Ranked product-store combinations")

        columns = [
            col for col in [
                item_col,
                store_col,
                category_col,
                "units_last_28d",
                "units_last_90d",
                "active_days_last_90d",
                "Low-sales probability (%)",
                "Risk level",
            ]
            if col and col in risk.columns
        ]

        display_risk = risk[columns].rename(
            columns={
                "units_last_28d": "Sales · 28d",
                "units_last_90d": "Sales · 90d",
                "active_days_last_90d": "Active days · 90d",
            }
        )

        st.dataframe(
            display_risk,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download risk analysis (CSV)",
            risk.to_csv(index=False).encode("utf-8"),
            "productpulse_risk_analysis.csv",
            "text/csv",
        )

    except Exception as exc:
        st.error(f"Could not generate risk analysis: {exc}")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="pp-eyebrow">Model validation</div>',
        unsafe_allow_html=True,
    )

    st.title("Model Performance")

    st.markdown(
        """
        <p class="pp-muted">
            Review the final candidate model's held-out evaluation,
            classification results, confusion matrix, and model comparisons.
        </p>
        """,
        unsafe_allow_html=True,
    )

    report_text = load_report_text(REPORT_PATH)

    if report_text:

        metrics = {
            "Balanced accuracy": find_metric_value(
                report_text, "balanced_accuracy"
            ),
            "F1 score": find_metric_value(report_text, "f1"),
            "PR-AUC": find_metric_value(report_text, "pr_auc"),
            "ROC-AUC": find_metric_value(report_text, "roc_auc"),
            "Accuracy": find_metric_value(report_text, "accuracy"),
        }

        test_rows = find_metric_value(report_text, "test_rows")
        train_rows = find_metric_value(report_text, "training_rows")

        held_out = None
        date_match = re.search(
            r"(?im)^\s*held_out_date\s*:\s*(\d{4}-\d{2}-\d{2})",
            report_text,
        )

        if date_match:
            held_out = date_match.group(1)

        st.markdown("### Evaluation metrics")

        available_metrics = [
            (name, value)
            for name, value in metrics.items()
            if value is not None
        ]

        if available_metrics:
            metric_columns = st.columns(
                min(len(available_metrics), 4),
                gap="large",
            )

            for column, (name, value) in zip(
                metric_columns, available_metrics
            ):
                with column:
                    st.metric(name, f"{value:.3f}")

        st.markdown(
            '<div class="pp-section"></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Evaluation setup")

        context_cols = st.columns(3, gap="large")

        with context_cols[0]:
            html = f"""
            <div class="pp-panel">
                <div class="pp-kicker">Held-out date</div>
                <div class="pp-big-number" style="font-size:1.45rem">
                    {held_out or "Not reported"}
                </div>
                <div class="pp-small">Chronological test period</div>
            </div>
            """
            st.markdown(dedent(html), unsafe_allow_html=True)

        with context_cols[1]:
            html = f"""
            <div class="pp-panel">
                <div class="pp-kicker">Training rows</div>
                <div class="pp-big-number">{fmt(train_rows)}</div>
                <div class="pp-small">Rows used for model fitting</div>
            </div>
            """
            st.markdown(dedent(html), unsafe_allow_html=True)

        with context_cols[2]:
            html = f"""
            <div class="pp-panel">
                <div class="pp-kicker">Test rows</div>
                <div class="pp-big-number">{fmt(test_rows)}</div>
                <div class="pp-small">Held-out observations</div>
            </div>
            """
            st.markdown(dedent(html), unsafe_allow_html=True)

        st.markdown(
            '<div class="pp-section"></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Classification results")

        report_table = render_report_table(report_text)

        if not report_table.empty:
            st.dataframe(
                report_table.style.format({
                    "Precision": "{:.2f}",
                    "Recall": "{:.2f}",
                    "F1-score": "{:.2f}",
                    "Support": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "Could not parse the classification report automatically."
            )

        cm_match = re.search(
            r"(?is)confusion matrix\s*:\s*"
            r"\[\[\s*(\d+)\s+(\d+)\s*\]\s*"
            r"\[\s*(\d+)\s+(\d+)\s*\]\s*\]",
            report_text,
        )

        if cm_match:
            tn, fp, fn, tp = map(int, cm_match.groups())

            st.markdown(
                '<div class="pp-section"></div>',
                unsafe_allow_html=True,
            )

            st.markdown("### Confusion matrix")

            st.caption(
                "Rows are actual outcomes; columns are model predictions."
            )

            left, right = st.columns([1.15, 0.85], gap="large")

            with left:
                fig_cm = go.Figure(
                    data=go.Heatmap(
                        z=[[tn, fp], [fn, tp]],
                        x=[
                            "Predicted 20+ units",
                            "Predicted low sales",
                        ],
                        y=[
                            "Actual 20+ units",
                            "Actual low sales",
                        ],
                        colorscale=[
                            [0, "#1b2b43"],
                            [0.5, "#426b9d"],
                            [1, "#8abaff"],
                        ],
                        text=[
                            [f"{tn:,}", f"{fp:,}"],
                            [f"{fn:,}", f"{tp:,}"],
                        ],
                        texttemplate="%{text}",
                        textfont={"size": 17, "color": "#ffffff"},
                        hovertemplate=(
                            "Actual: %{y}<br>%{x}<br>"
                            "Count: %{z:,}<extra></extra>"
                        ),
                        showscale=False,
                    )
                )

                fig_cm.update_layout(
                    xaxis_title=None,
                    yaxis_title=None,
                    margin=dict(l=15, r=15, t=20, b=20),
                    height=350,
                )

                st.plotly_chart(
                    styled_plot(fig_cm, 350),
                    use_container_width=True,
                    config={"displaylogo": False},
                )

            with right:
                st.markdown("### Understanding the counts")
                st.metric("Correctly identified 20+ sales", f"{tn:,}")
                st.metric("False low-sales alarms", f"{fp:,}")
                st.metric("Missed low-sales cases", f"{fn:,}")
                st.metric("Correctly identified low sales", f"{tp:,}")

                st.caption(
                    "This interpretation assumes the report's class order "
                    "is 20+ units first and low future sales second."
                )

        st.markdown(
            '<div class="pp-section"></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Saved evaluation visuals")

        visual_files = [
            ("ROC curve", CHART_DIR / "roc_curve.png"),
            (
                "Precision-recall curve",
                CHART_DIR / "precision_recall_curve.png",
            ),
            (
                "Feature importance",
                CHART_DIR / "feature_importance.png",
            ),
        ]

        found_visuals = [
            (label, path)
            for label, path in visual_files
            if path.exists()
        ]

        if found_visuals:
            tabs = st.tabs([label for label, _ in found_visuals])

            for tab, (label, path) in zip(tabs, found_visuals):
                with tab:
                    st.image(str(path), use_container_width=True)
        else:
            st.caption(
                "No saved ROC, precision-recall, or feature-importance "
                "images were found."
            )

        with st.expander("View original evaluation report"):
            st.code(report_text, language="text")

    else:
        st.warning("The final candidate evaluation report was not found.")
        st.caption(f"Expected location: {REPORT_PATH}")

    comparison_path = (
        TEMPORAL_REPORT
        if TEMPORAL_REPORT.exists()
        else MODEL_REPORT
    )

    if comparison_path.exists():
        st.markdown(
            '<div class="pp-section"></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Model comparison")

        try:
            comparison = pd.read_csv(comparison_path)

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download model comparison (CSV)",
                comparison.to_csv(index=False).encode("utf-8"),
                "productpulse_model_comparison.csv",
                "text/csv",
            )

        except Exception as exc:
            st.warning(f"Could not read model comparison file: {exc}")

    if FEATURE_IMPORTANCE_PATH.exists():
        with st.expander("Inspect feature importance data"):
            try:
                feature_importance = pd.read_csv(
                    FEATURE_IMPORTANCE_PATH
                )

                st.dataframe(
                    feature_importance,
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception as exc:
                st.warning(
                    f"Could not read feature importance file: {exc}"
                )

    if ABLATION_PATH.exists():
        with st.expander("Inspect feature ablation results"):
            try:
                ablation = pd.read_csv(ABLATION_PATH)

                st.dataframe(
                    ablation,
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception as exc:
                st.warning(
                    f"Could not read feature ablation file: {exc}"
                )

    st.markdown(
        '<div class="pp-section"></div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Precision measures how often low-sales predictions are correct. "
        "Recall measures how many actual low-sales cases the model catches. "
        "PR-AUC summarizes precision-recall performance. Use chronological "
        "evaluation results to assess performance on later dates."
    )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.markdown(
        '<div class="pp-eyebrow">Explore the dataset</div>',
        unsafe_allow_html=True,
    )

    st.title("Data Explorer")

    st.markdown(
        """
        <p class="pp-muted">
            Inspect the processed data behind ProductPulse and export
            the rows matching your selected filters.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Filter the displayed records")

    search = st.text_input(
        "Filter product IDs",
        placeholder="Enter part of a product ID",
    )

    view = filtered.copy()

    if search.strip():
        view = view[
            view[item_col].astype(str).str.contains(
                search.strip(),
                case=False,
                na=False,
            )
        ]

    st.markdown(
        '<div class="pp-section"></div>',
        unsafe_allow_html=True,
    )

    a, b = st.columns(2, gap="large")

    with a:
        st.metric("Matching rows", f"{len(view):,}")

    with b:
        st.metric("Columns", f"{len(view.columns):,}")

    st.markdown("### Dataset preview")

    if not view.empty:
        maximum = min(500, len(view))
        default = min(100, maximum)

        if maximum >= 10:
            limit = st.slider(
                "Rows to display",
                min_value=10,
                max_value=maximum,
                value=default,
                step=10,
            )
        else:
            limit = maximum

        st.dataframe(
            view.head(limit),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No rows match your search.")

    st.markdown(
        '<div class="pp-section"></div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download filtered data (CSV)",
        view.to_csv(index=False).encode("utf-8"),
        "productpulse_filtered_data.csv",
        "text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ProductPulse · Sales continuation and low-future-sales analytics. "
    "Predictions are estimates based on historical data and should support, "
    "not replace, business judgment."
)
