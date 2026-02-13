import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import openai as OpenAI

from agents.orchestrator import (
    build_agent_table_for_week,
    generate_portfolio_summary,
    generate_customer_summary
)

api_key = None

if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=api_key)

st.set_page_config(layout="wide")
st.title("🚀 FusionX Agentic AI Command Center")

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_parquet(BASE_DIR / "data/master/final_predictions_enriched.parquet")
df = df.sort_values(["customer_id", "week_start"]).reset_index(drop=True)

# ==========================================================
# HELPERS
# ==========================================================

def format_week(date):
    return pd.to_datetime(date).strftime("%d-%b-%Y")

def calculate_portfolio_kpi(data):
    kpi = (
        data
        .groupby("week_start")
        .agg(
            total_customers=("customer_id", "nunique"),
            avg_usage_score=("usage_score", "mean"),
            avg_csat_score=("csat_score", "mean"),
            total_tickets=("tickets_created", "sum"),
            total_escalations=("escalations", "sum"),
            avg_first_response_hrs=("avg_first_response_hrs", "mean"),
            avg_resolution_hrs=("avg_resolution_hrs", "mean"),
            portfolio_sla_breaches=("sla_breaches", "sum"),
            portfolio_tickets_created=("tickets_created", "sum"),
            customers_at_churn_risk=("churn_flag", "sum"),
            customers_expansion_eligible=("expansion_flag", "sum"),
            customers_model_eligible=("churn_flag", "count")
        )
        .reset_index()
        .sort_values("week_start")
    )

    kpi["sla_breach_rate"] = (
        kpi["portfolio_sla_breaches"] / kpi["portfolio_tickets_created"]
    )

    kpi["churn_risk_rate_pct"] = (
        kpi["customers_at_churn_risk"] / kpi["customers_model_eligible"]
    ) * 100

    kpi["expansion_rate_pct"] = (
        kpi["customers_expansion_eligible"] / kpi["customers_model_eligible"]
    ) * 100

    kpi.drop(columns=["portfolio_sla_breaches", "portfolio_tickets_created"], inplace=True)

    return kpi


tab1, tab2 = st.tabs(["📊 Portfolio View", "👤 Customer View"])

# ==========================================================
# 📊 PORTFOLIO TAB
# ==========================================================

with tab1:

    st.header("📊 Portfolio Intelligence")

    col1, col2, col3 = st.columns(3)

    selected_week = col1.selectbox(
        "Select Week",
        sorted(df["week_start"].unique(), reverse=True),
        format_func=lambda x: format_week(x),
        key="pf_week"
    )

    selected_segment = col2.selectbox(
        "Select Segment",
        ["All"] + sorted(df["segment"].dropna().unique()),
        key="pf_segment"
    )

    selected_tier = col3.selectbox(
        "Select Tier",
        ["All"] + sorted(df["customer_tier"].dropna().unique()),
        key="pf_tier"
    )

    # ---- Apply Segment & Tier Filtering ----

    segment_df = df.copy()

    if selected_segment != "All":
        segment_df = segment_df[segment_df["segment"] == selected_segment]

    if selected_tier != "All":
        segment_df = segment_df[segment_df["customer_tier"] == selected_tier]

    # ---- KPI from FULL history of segment ----

    kpi_dynamic = calculate_portfolio_kpi(segment_df)

    # STRICT CAP to selected week
    kpi_dynamic = kpi_dynamic[kpi_dynamic["week_start"] <= selected_week]

    # Row for selected week
    kpi_row = kpi_dynamic[kpi_dynamic["week_start"] == selected_week]

    if kpi_row.empty:
        st.warning("No data available for selected filters.")
        st.stop()

    kpi_row = kpi_row.iloc[0]

    # ---- Agent Table for Selected Week ----

    week_df = segment_df[segment_df["week_start"] == selected_week]

    agent_week_df = build_agent_table_for_week(week_df, selected_week)

    # ======================================================
    # KPI CARDS
    # ======================================================

    total_arr = agent_week_df["arr_inr"].sum()
    arr_at_risk = agent_week_df.loc[
        agent_week_df["immediate_attention_flag"] == 1, "arr_inr"
    ].sum()

    immediate_count = agent_week_df["immediate_attention_flag"].sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Churn Risk Rate %",
              round(kpi_row["churn_risk_rate_pct"], 2))

    c2.metric("Expansion Rate %",
              round(kpi_row["expansion_rate_pct"], 2))

    c3.metric("Immediate Attention Accounts",
              int(immediate_count))

    c4.metric("ARR At Risk %",
              round((arr_at_risk / total_arr) * 100, 2) if total_arr > 0 else 0)

    st.divider()

    # ======================================================
    # PORTFOLIO TABLE
    # ======================================================

    st.subheader("📋 Portfolio Customer Table")

    st.dataframe(
        agent_week_df.sort_values("churn_risk_score", ascending=False),
        use_container_width=True,
        height=400
    )

    st.divider()

    # ======================================================
    # TOP DRIVERS
    # ======================================================

    st.subheader("🔥 Top Churn Drivers")

    driver_series = (
        agent_week_df["action_reason"]
        .str.split(", ")
        .explode()
        .value_counts()
        .head(5)
    )

    st.plotly_chart(
        px.bar(driver_series,
               title="Top Portfolio Risk Drivers"),
        use_container_width=True
    )

    # ======================================================
    # ACTION DISTRIBUTION
    # ======================================================

    st.subheader("📌 Action Distribution")

    action_counts = agent_week_df["next_best_action"].value_counts()

    st.plotly_chart(
        px.pie(values=action_counts.values,
               names=action_counts.index,
               title="Recommended Actions Distribution"),
        use_container_width=True
    )

    # ======================================================
    # CSM RISK LOAD
    # ======================================================

    st.subheader("👥 CSM Risk Load")

    csm_risk = (
        agent_week_df.groupby("csm_owner")["immediate_attention_flag"]
        .sum()
        .sort_values(ascending=False)
    )

    st.plotly_chart(
        px.bar(csm_risk,
               title="High-Risk Accounts per CSM"),
        use_container_width=True
    )

    # ======================================================
    # KPI TRENDS (STRICTLY CAPPED)
    # ======================================================

    st.subheader("📈 Portfolio KPI Trends")

    trend_pairs = [
        ("sla_breach_rate", "SLA Breach Rate"),
        ("avg_csat_score", "Avg CSAT"),
        ("churn_risk_rate_pct", "Churn Risk Rate %"),
        ("expansion_rate_pct", "Expansion Rate %"),
        ("avg_usage_score", "Avg Usage Score"),
        ("avg_resolution_hrs", "Avg Resolution Hrs"),
    ]

    for i in range(0, len(trend_pairs), 2):
        cols = st.columns(2)
        for col, (metric, title) in zip(cols, trend_pairs[i:i+2]):
            fig = px.line(kpi_dynamic, x="week_start", y=metric, title=title)
            fig.update_xaxes(range=[
                kpi_dynamic["week_start"].min(),
                selected_week
            ])
            col.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 AI Portfolio Report")

    if st.button("Generate Portfolio Report", key="pf_summary_btn"):
        summary = generate_portfolio_summary(kpi_row, agent_week_df)
        st.write(summary)


# ==========================================================
# 👤 CUSTOMER TAB
# ==========================================================

with tab2:

    st.header("👤 Customer Intelligence")

    customer_id = st.selectbox(
        "Select Customer",
        sorted(df["customer_id"].unique()),
        key="cust_id"
    )

    customer_weeks = sorted(
        df[df["customer_id"] == customer_id]["week_start"].unique(),
        reverse=True
    )

    selected_customer_week = st.selectbox(
        "Select Week",
        customer_weeks,
        format_func=lambda x: format_week(x),
        key="cust_week"
    )

    customer_trend_df = df[
        (df["customer_id"] == customer_id) &
        (df["week_start"] <= selected_customer_week)
    ]

    # Use orchestrator to get snapshot with action + drivers
    agent_snapshot_df = build_agent_table_for_week(df, selected_customer_week)

    snapshot_row = agent_snapshot_df[
        agent_snapshot_df["customer_id"] == customer_id
    ].iloc[0]

    # ======================================================
    # EXPANDED SNAPSHOT METRICS (4 PER ROW)
    # ======================================================

    st.subheader("Customer Snapshot")

    # Row 1
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("Churn Risk Score", round(snapshot_row["churn_risk_score"], 1))
    r1c2.metric("Expansion Score", round(snapshot_row["expansion_score"], 1))
    r1c3.metric("ARR (₹)", f"{int(snapshot_row['arr_inr']):,}")
    r1c4.metric("Days to Renewal", int(snapshot_row["days_to_renewal_capped"]))

    # Row 2
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    r2c1.metric(
        "Renewal Date",
        format_week(snapshot_row["renewal_date"])
        if pd.notna(snapshot_row["renewal_date"]) else "-"
    )
    r2c2.metric("Usage Score", round(snapshot_row.get("usage_score", 0), 1))
    r2c3.metric("Ticket Volume", int(snapshot_row.get("tickets_created", 0)))
    r2c4.metric("SLA Breaches", int(snapshot_row.get("sla_breaches", 0)))

    # Row 3
    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    r3c1.metric("CSAT", round(snapshot_row.get("csat_score", 0), 2))
    r3c2.metric("Overdue Payments", int(snapshot_row.get("days_overdue", 0)))
    r3c3.metric("Support Escalations", int(snapshot_row.get("escalations", 0)))
    r3c4.metric(
        "Renewal Approaching",
        "Yes" if snapshot_row.get("renewal_upcoming_30d", 0) == 1 else "No"
    )

    st.divider()

    # ======================================================
    # 📋 CUSTOMER 360 TABLE (RESTORED)
    # ======================================================

    st.subheader("📋 Customer 360 Overview")

    customer_cols = [
        "customer_name",
        "segment",
        "customer_tier",
        "region",
        "industry",
        "arr_inr",
        "renewal_date",
        "days_to_renewal_capped",
        "usage_score",
        "usage_score_mean_4w",
        "usage_score_mean_12w",
        "active_users",
        "tickets_created",
        "escalations",
        "csat_score",
        "days_overdue",
        "billing_risk_flag",
        "churn_flag",
        "expansion_flag"
    ]

    st.dataframe(
        customer_trend_df.tail(1)[customer_cols],
        use_container_width=True
    )

    st.divider()

    # ======================================================
    # CUSTOMER TRENDS (3 PER ROW + ADDITIONAL)
    # ======================================================
    st.subheader("Monitor (Health Signals)")
    
    trend_cols = [
        ("churn_risk_score", "Churn Risk Score"),
        ("expansion_score", "Expansion Score"),
        ("usage_score", "Usage"),
        ("csat_score", "CSAT"),
        ("tickets_created", "Ticket Volume"),
        ("sla_breaches", "SLA Breaches"),
        ("escalations", "Support Escalations"),
        ("days_overdue", "Overdue Invoice Count"),
        ("renewal_upcoming_30d", "Renewal Approaching")
    ]

    for i in range(0, len(trend_cols), 3):
        cols = st.columns(3)
        for col, (metric, title) in zip(cols, trend_cols[i:i+3]):
            if metric in customer_trend_df.columns:
                fig = px.line(
                    customer_trend_df,
                    x="week_start",
                    y=metric,
                    title=title
                )
                fig.update_xaxes(range=[
                    customer_trend_df["week_start"].min(),
                    selected_customer_week
                ])
                col.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ======================================================
    # NEXT BEST ACTION (RESTORED)
    # ======================================================
    st.subheader("Recommendations")
    st.markdown(f"**Next Best Action:** {snapshot_row['next_best_action']}")
    st.markdown(f"**Action Reason:** {snapshot_row['action_reason']}")
    st.markdown(f"**Action Owner:** {snapshot_row['action_owner']}")

    st.divider()

    # ======================================================
    # AI CUSTOMER SUMMARY
    # ======================================================

    st.subheader("🧠 AI Customer Report")

    if st.button("Generate Customer Report", key="cust_summary_btn"):
        summary = generate_customer_summary(snapshot_row)
        st.write(summary)
