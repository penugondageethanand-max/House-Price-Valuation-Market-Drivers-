"""
House Price Valuation & Market Drivers — Interactive BI Dashboard
=================================================================
A single-file Streamlit application that transforms the Ames Housing
dataset into an actionable business-intelligence dashboard for
real-estate stakeholders.

Pages
-----
1. Executive Overview    – KPI cards, price distributions, neighbourhood analysis
2. Property Analytics    – Feature correlations, drivers, quality / area analysis
3. Valuation Calculator  – ML-powered price estimator with risk & opportunity insights

Dataset : Ames Housing (Kaggle)
Model   : Gradient Boosting Regressor (scikit-learn)
Author  : <Your Name>
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")

# ── Streamlit Page Configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Valuation Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for KPI Cards & Insight Boxes ─────────────────────────────────
st.markdown(
    """
<style>
.kpi-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px; border-radius: 12px; color: white;
    text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.kpi-card h2 { margin: 0; font-size: 28px; }
.kpi-card p  { margin: 5px 0 0 0; font-size: 14px; opacity: 0.85; }
.risk-box   { background:#FFF3E0; border-left:4px solid #FF9800; padding:15px; border-radius:8px; margin:10px 0; }
.opp-box    { background:#E8F5E9; border-left:4px solid #4CAF50; padding:15px; border-radius:8px; margin:10px 0; }
.action-box { background:#E3F2FD; border-left:4px solid #2196F3; padding:15px; border-radius:8px; margin:10px 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING & CLEANING
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """Load the Ames Housing dataset from a local CSV file."""
    try:
        df = pd.read_csv("train.csv")
    except FileNotFoundError:
        st.error(
            "Dataset file **train.csv** not found!  "
            "Place it in the same directory as this script."
        )
        st.info(
            "Download from: "
            "https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data"
        )
        st.stop()
    return df


@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and engineer useful features."""
    df = df.copy()

    # Numerical columns → fill missing with median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns → fill missing with mode
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # ── Feature Engineering ──────────────────────────────────────────────────
    df["PricePerSqFt"] = df["SalePrice"] / df["GrLivArea"]
    df["HouseAge"]     = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"]     = df["YrSold"] - df["YearRemodAdd"]
    df["TotalSF"]      = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"]    = (
        df["FullBath"]
        + 0.5 * df["HalfBath"]
        + df["BsmtFullBath"]
        + 0.5 * df["BsmtHalfBath"]
    )
    df["HasGarage"]    = (df["GarageArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "OverallQual", "GrLivArea", "GarageCars", "GarageArea",
    "TotalBsmtSF", "1stFlrSF", "FullBath", "TotRmsAbvGrd",
    "YearBuilt", "YearRemodAdd", "LotArea", "Fireplaces",
    "TotalSF", "TotalBath", "HouseAge", "OverallCond",
]

FEATURE_LABELS = {
    "OverallQual":  "Overall Quality (1-10)",
    "GrLivArea":    "Living Area (sq ft)",
    "GarageCars":   "Garage Capacity (cars)",
    "GarageArea":   "Garage Area (sq ft)",
    "TotalBsmtSF":  "Basement Area (sq ft)",
    "1stFlrSF":     "First Floor Area (sq ft)",
    "FullBath":     "Full Bathrooms",
    "TotRmsAbvGrd": "Total Rooms",
    "YearBuilt":    "Year Built",
    "YearRemodAdd": "Year Remodeled",
    "LotArea":      "Lot Area (sq ft)",
    "Fireplaces":   "Fireplaces",
    "TotalSF":      "Total Area (sq ft)",
    "TotalBath":    "Total Bathrooms",
    "HouseAge":     "House Age (years)",
    "OverallCond":  "Overall Condition (1-10)",
}


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train a Gradient Boosting Regressor and return model + diagnostics."""
    X = df[FEATURE_COLS].copy()
    y = df["SalePrice"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    residual_std = (y_test - y_pred).std()

    metrics = {
        "R2":           r2_score(y_test, y_pred),
        "MAE":          mean_absolute_error(y_test, y_pred),
        "RMSE":         np.sqrt(mean_squared_error(y_test, y_pred)),
        "Residual_Std": residual_std,
    }

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    metrics["CV_R2_Mean"] = cv_scores.mean()
    metrics["CV_R2_Std"]  = cv_scores.std()

    importances = (
        pd.DataFrame({
            "Feature":    FEATURE_COLS,
            "Label":      [FEATURE_LABELS[f] for f in FEATURE_COLS],
            "Importance": model.feature_importances_,
        })
        .sort_values("Importance", ascending=True)
    )

    return model, metrics, importances, X_test, y_test, y_pred


# ═════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def kpi_card(title: str, value: str, subtitle: str = ""):
    """Render a styled KPI metric card."""
    st.markdown(
        f'<div class="kpi-card"><p>{title}</p>'
        f"<h2>{value}</h2><p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


def insight_box(text: str, box_type: str = "risk"):
    """Render a coloured insight box (risk / opp / action)."""
    st.markdown(f'<div class="{box_type}-box">{text}</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

def main():
    raw_df = load_data()
    df     = clean_data(raw_df)
    model, metrics, importances, X_test, y_test, y_pred = train_model(df)

    # ── Sidebar Navigation ───────────────────────────────────────────────────
    st.sidebar.title("🏠 Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "📊 Executive Overview",
            "🔍 Property Analytics",
            "🧮 Valuation Calculator",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Model Performance")
    st.sidebar.metric("R² Score", f"{metrics['R2']:.3f}")
    st.sidebar.metric("MAE",     f"${metrics['MAE']:,.0f}")
    st.sidebar.metric(
        "CV R²",
        f"{metrics['CV_R2_Mean']:.3f} ± {metrics['CV_R2_Std']:.3f}",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Dataset: Ames Housing (Kaggle)")
    st.sidebar.caption(f"Records: {len(df):,}")

    # ── Page Routing ─────────────────────────────────────────────────────────
    if page == "📊 Executive Overview":
        page_overview(df)
    elif page == "🔍 Property Analytics":
        page_analytics(df, importances, metrics, X_test, y_test, y_pred)
    else:
        page_calculator(df, model, metrics)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def page_overview(df: pd.DataFrame):
    st.title("📊 Executive Overview")
    st.markdown(
        "Key performance indicators and market snapshot "
        "for the Ames, Iowa housing market."
    )

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card(
            "Median Sale Price",
            f"${df['SalePrice'].median():,.0f}",
            "Central tendency",
        )
    with k2:
        kpi_card(
            "Avg Price / Sq Ft",
            f"${df['PricePerSqFt'].mean():.2f}",
            "Value density",
        )
    with k3:
        kpi_card("Total Properties", f"{len(df):,}", "Dataset size")
    with k4:
        kpi_card(
            "Price Spread",
            f"${df['SalePrice'].max() - df['SalePrice'].min():,.0f}",
            f"${df['SalePrice'].min():,.0f} – ${df['SalePrice'].max():,.0f}",
        )

    st.markdown("---")

    # Row 1 — Price Distribution + Top Neighbourhoods
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Sale Price Distribution")
        fig = px.histogram(
            df, x="SalePrice", nbins=50,
            color_discrete_sequence=["#667eea"],
            template="plotly_white",
        )
        fig.update_layout(xaxis_title="Sale Price ($)", yaxis_title="Count", bargap=0.05)
        fig.add_vline(
            x=df["SalePrice"].median(), line_dash="dash", line_color="#F44336",
            annotation_text=f"Median ${df['SalePrice'].median():,.0f}",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Top 15 Neighbourhoods by Median Price")
        nbr = (
            df.groupby("Neighborhood")["SalePrice"]
            .median()
            .sort_values(ascending=True)
            .tail(15)
        )
        fig = px.bar(
            x=nbr.values, y=nbr.index, orientation="h",
            color=nbr.values, color_continuous_scale="Viridis",
            template="plotly_white",
        )
        fig.update_layout(
            xaxis_title="Median Sale Price ($)", yaxis_title="",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 — Year Built scatter + Quality box-plot
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Price by Year Built")
        fig = px.scatter(
            df, x="YearBuilt", y="SalePrice",
            color="OverallQual", color_continuous_scale="RdYlGn",
            opacity=0.6, template="plotly_white",
            hover_data=["Neighborhood", "GrLivArea"],
        )
        fig.update_layout(xaxis_title="Year Built", yaxis_title="Sale Price ($)")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Price by Overall Quality")
        fig = px.box(
            df, x="OverallQual", y="SalePrice",
            color="OverallQual",
            color_discrete_sequence=px.colors.sequential.Viridis,
            template="plotly_white",
        )
        fig.update_layout(
            xaxis_title="Overall Quality Rating",
            yaxis_title="Sale Price ($)", showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — PROPERTY ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

def page_analytics(df, importances, metrics, X_test, y_test, y_pred):
    st.title("🔍 Property Analytics")
    st.markdown(
        "Deep-dive into price drivers, feature correlations, "
        "and model diagnostics."
    )

    # Row 1 — Feature Importance + Correlation Heat-map
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("🎯 Top Price Drivers")
        fig = px.bar(
            importances.tail(12), x="Importance", y="Label",
            orientation="h", color="Importance",
            color_continuous_scale="Blues", template="plotly_white",
        )
        fig.update_layout(
            xaxis_title="Relative Importance", yaxis_title="",
            coloraxis_showscale=False, height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("📊 Feature Correlation Matrix")
        corr_cols = FEATURE_COLS + ["SalePrice"]
        corr = df[corr_cols].corr()
        top_corr = (
            corr["SalePrice"]
            .drop("SalePrice")
            .abs()
            .sort_values(ascending=False)
            .head(10)
        )
        fig = px.imshow(
            corr.loc[top_corr.index, top_corr.index],
            text_auto=".2f", color_continuous_scale="RdBu_r",
            template="plotly_white", aspect="auto",
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Row 2 — Living Area scatter + Actual vs Predicted
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Living Area vs Sale Price")
        fig = px.scatter(
            df, x="GrLivArea", y="SalePrice",
            color="OverallQual", color_continuous_scale="Turbo",
            opacity=0.6, trendline="ols", template="plotly_white",
        )
        fig.update_layout(
            xaxis_title="Above-Grade Living Area (sq ft)",
            yaxis_title="Sale Price ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Actual vs Predicted Prices")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test.values, y=y_pred, mode="markers",
            marker=dict(color="#667eea", opacity=0.5, size=6),
            name="Predictions",
        ))
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx], mode="lines",
            line=dict(color="#F44336", dash="dash"),
            name="Perfect Prediction",
        ))
        fig.update_layout(
            xaxis_title="Actual ($)", yaxis_title="Predicted ($)",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 3 — House Style violin + Bathrooms bar
    c5, c6 = st.columns(2)

    with c5:
        st.subheader("Price by House Style")
        top_styles = df["HouseStyle"].value_counts().head(6).index
        fig = px.violin(
            df[df["HouseStyle"].isin(top_styles)],
            x="HouseStyle", y="SalePrice", color="HouseStyle",
            template="plotly_white", box=True,
        )
        fig.update_layout(
            xaxis_title="House Style", yaxis_title="Sale Price ($)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        st.subheader("Impact of Total Bathrooms")
        bath = (
            df.groupby("TotalBath")["SalePrice"]
            .agg(["median", "count"])
            .reset_index()
        )
        bath = bath[bath["count"] >= 5]
        fig = px.bar(
            bath, x="TotalBath", y="median",
            color="median", color_continuous_scale="Greens",
            template="plotly_white",
            text=bath["median"].apply(lambda v: f"${v:,.0f}"),
        )
        fig.update_layout(
            xaxis_title="Total Bathrooms",
            yaxis_title="Median Sale Price ($)",
            coloraxis_showscale=False,
        )
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # Model metrics summary
    st.markdown("---")
    st.subheader("📋 Model Performance Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score",            f"{metrics['R2']:.4f}")
    m2.metric("Mean Absolute Error", f"${metrics['MAE']:,.0f}")
    m3.metric("Root Mean Sq Error",  f"${metrics['RMSE']:,.0f}")
    m4.metric(
        "Cross-Val R²",
        f"{metrics['CV_R2_Mean']:.4f} ± {metrics['CV_R2_Std']:.4f}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — VALUATION CALCULATOR & STRATEGIC INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def page_calculator(df, model, metrics):
    st.title("🧮 Property Valuation Calculator")
    st.markdown(
        "Estimate the market value of a property and uncover "
        "strategic risks, opportunities, and actions."
    )

    # ── Input Form ───────────────────────────────────────────────────────────
    st.subheader("🏡 Enter Property Details")
    c1, c2, c3 = st.columns(3)

    with c1:
        overall_qual  = st.slider("Overall Quality (1–10)", 1, 10, 6)
        overall_cond  = st.slider("Overall Condition (1–10)", 1, 10, 5)
        gr_liv_area   = st.number_input("Living Area (sq ft)", 300, 6000, 1500, step=50)
        first_flr_sf  = st.number_input("1st Floor Area (sq ft)", 300, 4000, 1000, step=50)

    with c2:
        total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 3000, 800, step=50)
        garage_cars   = st.slider("Garage Capacity (cars)", 0, 4, 2)
        garage_area   = st.number_input("Garage Area (sq ft)", 0, 1500, 480, step=50)
        lot_area      = st.number_input("Lot Area (sq ft)", 1000, 100000, 9000, step=500)

    with c3:
        year_built  = st.slider("Year Built", 1870, 2010, 1990)
        year_remod  = st.slider("Year Remodeled", 1950, 2010, 2000)
        full_bath   = st.slider("Full Bathrooms", 0, 4, 2)
        tot_rooms   = st.slider("Total Rooms Above Grade", 2, 14, 6)
        fireplaces  = st.slider("Fireplaces", 0, 3, 1)

    # Derived features
    total_sf   = total_bsmt_sf + first_flr_sf + max(0, gr_liv_area - first_flr_sf)
    total_bath = float(full_bath)
    house_age  = 2010 - year_built  # dataset max YrSold

    input_data = pd.DataFrame([{
        "OverallQual":  overall_qual,
        "GrLivArea":    gr_liv_area,
        "GarageCars":   garage_cars,
        "GarageArea":   garage_area,
        "TotalBsmtSF":  total_bsmt_sf,
        "1stFlrSF":     first_flr_sf,
        "FullBath":     full_bath,
        "TotRmsAbvGrd": tot_rooms,
        "YearBuilt":    year_built,
        "YearRemodAdd": year_remod,
        "LotArea":      lot_area,
        "Fireplaces":   fireplaces,
        "TotalSF":      total_sf,
        "TotalBath":    total_bath,
        "HouseAge":     house_age,
        "OverallCond":  overall_cond,
    }])

    # ── Prediction ───────────────────────────────────────────────────────────
    predicted_price = model.predict(input_data)[0]
    residual_std    = metrics["Residual_Std"]
    lower_bound     = max(0, predicted_price - 1.96 * residual_std)
    upper_bound     = predicted_price + 1.96 * residual_std

    st.markdown("---")
    st.subheader("💰 Estimated Property Value")

    r1, r2, r3 = st.columns(3)
    with r1:
        kpi_card("Predicted Price", f"${predicted_price:,.0f}", "Best estimate")
    with r2:
        kpi_card("Lower Bound (95%)", f"${lower_bound:,.0f}", "Conservative")
    with r3:
        kpi_card("Upper Bound (95%)", f"${upper_bound:,.0f}", "Optimistic")

    # Price-per-sqft comparison
    pred_ppsf = predicted_price / gr_liv_area if gr_liv_area > 0 else 0
    avg_ppsf  = df["PricePerSqFt"].mean()
    verdict   = (
        "🟢 Below market average — good value"
        if pred_ppsf < avg_ppsf
        else "🔴 Above market average — premium property"
    )
    st.info(
        f"**Price per Sq Ft:** {pred_ppsf:.2f} USD  |  "
        f"**Market avg:** {avg_ppsf:.2f} USD  —  {verdict}"
    )

    st.markdown("---")

    # ── Comparable Properties ────────────────────────────────────────────────
    st.subheader("🏘️ Comparable Properties in Dataset")
    margin = predicted_price * 0.15
    comps  = df[
        (df["SalePrice"].between(predicted_price - margin, predicted_price + margin))
        & (df["OverallQual"].between(overall_qual - 1, overall_qual + 1))
        & (df["GrLivArea"].between(gr_liv_area * 0.8, gr_liv_area * 1.2))
    ][
        ["Neighborhood", "SalePrice", "GrLivArea", "OverallQual",
         "YearBuilt", "GarageCars", "TotalBath"]
    ].head(10)

    if len(comps) > 0:
        st.dataframe(
            comps.style.format({
                "SalePrice": "${:,.0f}",
                "GrLivArea": "{:,.0f}",
            }),
            use_container_width=True,
        )
    else:
        st.warning("No close comparables found — try adjusting your inputs.")

    st.markdown("---")

    # ── Risk · Opportunity · Action ──────────────────────────────────────────
    st.subheader("⚡ Strategic Insights — Fact → Insight → Action")

    col_r, col_o, col_a = st.columns(3)

    # ── RISKS ────────────────────────────────────────────────────────────────
    with col_r:
        st.markdown("#### ⚠️ Risks")
        risk_count = 0
        if house_age > 50:
            insight_box(
                "🔧 <b>Aging Structure:</b> Built before 1960 — expect "
                "higher maintenance and possible structural upgrades.",
                "risk",
            )
            risk_count += 1
        if overall_cond < 5:
            insight_box(
                "📉 <b>Below-Avg Condition:</b> Faster depreciation and "
                "fewer interested buyers.",
                "risk",
            )
            risk_count += 1
        if garage_cars == 0:
            insight_box(
                "🚗 <b>No Garage:</b> Reduces appeal in suburban markets "
                "by an estimated 8-12%.",
                "risk",
            )
            risk_count += 1
        if overall_qual <= 4:
            insight_box(
                "⬇️ <b>Low Quality:</b> Strongly correlates with "
                "below-median pricing and slower sales.",
                "risk",
            )
            risk_count += 1
        if risk_count == 0:
            insight_box(
                "✅ <b>Low Risk Profile:</b> No major red flags detected.",
                "opp",
            )

    # ── OPPORTUNITIES ────────────────────────────────────────────────────────
    with col_o:
        st.markdown("#### 🌟 Opportunities")
        if overall_qual >= 7:
            insight_box(
                "⭐ <b>Premium Quality:</b> Top-quartile homes command "
                "a significant price premium.",
                "opp",
            )
        if year_built >= 2000:
            insight_box(
                "🏗️ <b>Modern Build:</b> Updated codes, energy efficiency, "
                "and lower near-term maintenance.",
                "opp",
            )
        if gr_liv_area > df["GrLivArea"].quantile(0.75):
            insight_box(
                f"📐 <b>Above-Avg Size:</b> {gr_liv_area:,} sq ft exceeds "
                "the 75th percentile — strong family appeal.",
                "opp",
            )
        if fireplaces >= 1:
            insight_box(
                "🔥 <b>Fireplace Premium:</b> Adds an estimated "
                "5K–12K USD in perceived value.",
                "opp",
            )
        remod_gap = year_remod - year_built
        if remod_gap > 10:
            insight_box(
                f"🔨 <b>Remodeled:</b> Updated {remod_gap} years after "
                "construction — modernised appeal.",
                "opp",
            )

    # ── RECOMMENDED ACTIONS ──────────────────────────────────────────────────
    with col_a:
        st.markdown("#### 🎯 Recommended Actions")
        if overall_cond < 5:
            insight_box(
                "🔨 <b>Invest in Repairs:</b> Raising condition to 5+ "
                "can lift value by 10-15%. Prioritise exterior & kitchen.",
                "action",
            )
        if garage_cars == 0:
            insight_box(
                "🏗️ <b>Add Garage:</b> A 2-car addition typically "
                "yields 60-80% ROI in this market.",
                "action",
            )
        if total_bath < 2:
            insight_box(
                "🚿 <b>Add Bathroom:</b> 2+ baths sell faster and "
                "attract a ~15K USD premium.",
                "action",
            )
        if year_remod < 1990:
            insight_box(
                "🔄 <b>Modernize:</b> Kitchen/bath renovation can "
                "boost value 15-20% for pre-1990 remodels.",
                "action",
            )
        if overall_qual >= 7 and house_age < 20:
            insight_box(
                "📈 <b>Price Competitively:</b> Strong demand profile — "
                "list at or slightly above estimated value.",
                "action",
            )
        if house_age > 30 and overall_qual >= 6:
            insight_box(
                "🏡 <b>Highlight Character:</b> Market vintage charm "
                "alongside modern upgrades.",
                "action",
            )

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "Model: Gradient Boosting Regressor · Dataset: Ames Housing (Kaggle) · "
        "Built with Streamlit + scikit-learn + Plotly"
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
