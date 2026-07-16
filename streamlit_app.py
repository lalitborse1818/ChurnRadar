"""
streamlit_app.py
-----------------
Interactive dashboard for ChurnRadar. Reads the artifacts produced by
src/pipeline.py (model scores, SHAP rankings, metrics); if they don't exist
yet (e.g. a fresh clone on Streamlit Community Cloud where large generated
files aren't committed), it runs the pipeline once on first load and caches
the result.

Run locally:
    streamlit run streamlit_app.py

Deploy free on Streamlit Community Cloud:
    https://share.streamlit.io -> New app -> point at this repo + this file.
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from config import (  # noqa: E402
    FEATURES_CSV,
    METRICS_PATH,
    REPORTS_DIR,
    POWERBI_EXPORT,
    POWERBI_DIR,
    SHAP_SUMMARY_PATH,
    SHAP_BAR_PATH,
)

st.set_page_config(
    page_title="ChurnRadar — Predictive Churn Dashboard",
    page_icon="📡",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading — run the pipeline once if artifacts are missing (fresh clone)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ensure_pipeline_has_run() -> bool:
    required = [FEATURES_CSV, METRICS_PATH, POWERBI_EXPORT, REPORTS_DIR / "top_churn_drivers.csv"]
    if all(p.exists() for p in required):
        return True

    with st.spinner("First run: generating data, training the model, and running SHAP (~10s)..."):
        result = subprocess.run(
            [sys.executable, "pipeline.py"],
            cwd=str(SRC),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            st.error("Pipeline failed to run. See details below.")
            st.code(result.stdout + "\n" + result.stderr)
            st.stop()
    return True


@st.cache_data(show_spinner=False)
def load_data():
    scored = pd.read_csv(POWERBI_EXPORT)
    metrics = json.loads(METRICS_PATH.read_text())
    drivers = pd.read_csv(REPORTS_DIR / "top_churn_drivers.csv")
    exec_summary = pd.read_csv(POWERBI_DIR / "executive_summary.csv").iloc[0]
    return scored, metrics, drivers, exec_summary


ensure_pipeline_has_run()
scored, metrics, drivers, exec_summary = load_data()


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title("📡 ChurnRadar")
st.sidebar.caption("Python · XGBoost · SHAP · SQL · Power BI")

st.sidebar.markdown("### Filters")
region_filter = st.sidebar.multiselect(
    "Region", sorted(scored["region"].unique()), default=sorted(scored["region"].unique())
)
tier_filter = st.sidebar.multiselect(
    "Plan tier", sorted(scored["plan_tier"].unique()), default=sorted(scored["plan_tier"].unique())
)
risk_filter = st.sidebar.multiselect(
    "Risk band", ["Low", "Medium", "High", "Critical"], default=["Low", "Medium", "High", "Critical"]
)

filtered = scored[
    scored["region"].isin(region_filter)
    & scored["plan_tier"].isin(tier_filter)
    & scored["risk_band"].isin(risk_filter)
]

st.sidebar.markdown("---")
st.sidebar.metric("Model AUC-ROC", f"{metrics['auc_roc']:.3f}")
st.sidebar.metric("Precision / Recall", f"{metrics['precision']:.2f} / {metrics['recall']:.2f}")
st.sidebar.caption(f"Trained on {metrics['n_train']:,} rows · scored {len(scored):,} customers")


# ---------------------------------------------------------------------------
# Header + KPI row
# ---------------------------------------------------------------------------
st.title("ChurnRadar — Predictive Churn Dashboard")
st.caption(
    "End-to-end churn prediction: SQL feature engineering → XGBoost "
    f"({metrics['auc_roc']:.1%} AUC-ROC) → SHAP explainability."
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customers (filtered)", f"{len(filtered):,}")
k2.metric("Actual churn rate", f"{filtered['churned'].mean():.1%}")
k3.metric("Predicted churn rate", f"{filtered['predicted_churn'].mean():.1%}")
k4.metric(
    "Monthly revenue at risk",
    f"${filtered['monthly_revenue_at_risk'].sum():,.0f}",
)
k5.metric(
    "Critical-risk customers",
    f"{(filtered['risk_band'] == 'Critical').sum():,}",
)

st.markdown("---")

tab_overview, tab_segments, tab_explain, tab_customers = st.tabs(
    ["📊 Overview", "🧩 Segments", "🔍 Explainability", "👤 Customer Explorer"]
)

# ---------------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Risk band distribution")
        risk_counts = (
            filtered["risk_band"]
            .value_counts()
            .reindex(["Low", "Medium", "High", "Critical"])
            .fillna(0)
            .reset_index()
        )
        risk_counts.columns = ["risk_band", "customers"]
        fig = px.bar(
            risk_counts,
            x="risk_band",
            y="customers",
            color="risk_band",
            color_discrete_map={
                "Low": "#4CAF50",
                "Medium": "#FFC107",
                "High": "#FF7043",
                "Critical": "#D32F2F",
            },
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Customers")
        st.plotly_chart(fig, width='stretch')

    with c2:
        st.subheader("Revenue at risk by risk band")
        rev_by_band = (
            filtered.groupby("risk_band", observed=True)["monthly_revenue_at_risk"]
            .sum()
            .reindex(["Low", "Medium", "High", "Critical"])
            .fillna(0)
            .reset_index()
        )
        fig2 = px.bar(
            rev_by_band,
            x="risk_band",
            y="monthly_revenue_at_risk",
            color="risk_band",
            color_discrete_map={
                "Low": "#4CAF50",
                "Medium": "#FFC107",
                "High": "#FF7043",
                "Critical": "#D32F2F",
            },
        )
        fig2.update_layout(showlegend=False, xaxis_title="", yaxis_title="Monthly $ at risk")
        st.plotly_chart(fig2, width='stretch')

    st.subheader("Model performance")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm, index=["Actual: Stayed", "Actual: Churned"], columns=["Pred: Stayed", "Pred: Churned"]
    )
    m1, m2 = st.columns([1, 1])
    with m1:
        st.dataframe(cm_df, width='stretch')
    with m2:
        roc = pd.read_csv(REPORTS_DIR / "roc_curve.csv")
        fig3 = px.line(roc, x="fpr", y="tpr", title=f"ROC Curve (AUC = {metrics['auc_roc']:.3f})")
        fig3.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="gray"))
        fig3.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig3, width='stretch')


# ---------------------------------------------------------------------------
# Segments tab
# ---------------------------------------------------------------------------
with tab_segments:
    st.subheader("Churn rate by tenure segment × contract risk")
    seg = (
        filtered.groupby(["tenure_segment", "contract_risk_tier"], observed=True)
        .agg(customers=("customer_id", "count"), churn_rate=("churned", "mean"))
        .reset_index()
    )
    pivot = seg.pivot(index="tenure_segment", columns="contract_risk_tier", values="churn_rate")
    st.dataframe(pivot.style.format("{:.1%}").background_gradient(cmap="Reds", axis=None), width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Churn rate by region")
        by_region = filtered.groupby("region", observed=True)["churned"].mean().sort_values(ascending=False)
        st.plotly_chart(
            px.bar(by_region.reset_index(), x="region", y="churned", labels={"churned": "Churn rate"}),
            width='stretch',
        )
    with c2:
        st.subheader("Churn rate by behavioral flags")
        flag_cols = [
            "high_support_flag",
            "late_payment_flag",
            "low_engagement_billing_flag",
            "bundled_customer_flag",
        ]
        flag_rates = {c: filtered.loc[filtered[c] == 1, "churned"].mean() for c in flag_cols}
        flag_df = pd.DataFrame({"flag": list(flag_rates.keys()), "churn_rate": list(flag_rates.values())})
        st.plotly_chart(
            px.bar(flag_df, x="flag", y="churn_rate", labels={"churn_rate": "Churn rate"}),
            width='stretch',
        )


# ---------------------------------------------------------------------------
# Explainability tab
# ---------------------------------------------------------------------------
with tab_explain:
    st.subheader("Top churn drivers (SHAP)")
    st.caption("Ranked by mean |SHAP value| across a representative customer sample.")
    st.dataframe(
        drivers[["rank", "feature_readable", "mean_abs_shap", "direction"]],
        width='stretch',
        hide_index=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if SHAP_BAR_PATH.exists():
            st.image(str(SHAP_BAR_PATH), caption="Global feature importance")
    with c2:
        if SHAP_SUMMARY_PATH.exists():
            st.image(str(SHAP_SUMMARY_PATH), caption="SHAP summary (direction + magnitude)")


# ---------------------------------------------------------------------------
# Customer Explorer tab
# ---------------------------------------------------------------------------
with tab_customers:
    st.subheader("At-risk customer table")
    sort_col = st.selectbox(
        "Sort by", ["churn_probability", "monthly_revenue_at_risk", "tenure_months"], index=0
    )
    top_n = st.slider("Show top N customers", 10, 500, 50, step=10)

    display_cols = [
        "customer_id",
        "region",
        "plan_tier",
        "tenure_segment",
        "contract_risk_tier",
        "churn_probability",
        "risk_band",
        "monthly_charges",
        "monthly_revenue_at_risk",
        "support_calls_last_90d",
        "late_payments_last_12m",
    ]
    table = filtered.sort_values(sort_col, ascending=False)[display_cols].head(top_n)
    st.dataframe(
        table.style.format(
            {
                "churn_probability": "{:.1%}",
                "monthly_charges": "${:.2f}",
                "monthly_revenue_at_risk": "${:.2f}",
            }
        ),
        width='stretch',
        hide_index=True,
    )

    st.download_button(
        "Download this table as CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="churnradar_at_risk_customers.csv",
        mime="text/csv",
    )

st.markdown("---")
st.caption(
    "ChurnRadar is a portfolio project trained on synthetic data "
    "(see `src/generate_data.py`). Swap in a real extract via `src/db_utils.py` "
    "to run against production data."
)
