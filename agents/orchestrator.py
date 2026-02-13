import pandas as pd
import numpy as np
import json
import os
from openai import OpenAI

import streamlit as st

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)


# ==========================================================
# ----------------- DETERMINISTIC ENGINE -------------------
# ==========================================================

def get_churn_drivers(row):
    drivers = []

    if row.get("usage_decline_12w", 0) == 1:
        drivers.append("Usage decline")

    if row.get("csat_12w_avg", 5) < 3.5:
        drivers.append("Low CSAT")

    if row.get("sla_breach_4w_flag", 0) == 1 or row.get("escalations_12w_sum", 0) > 0:
        drivers.append("Support issues")

    if row.get("billing_risk_flag", 0) == 1:
        drivers.append("Billing risk")

    if row.get("renewal_upcoming_30d", 0) == 1:
        drivers.append("Renewal imminent")

    return ", ".join(drivers) if drivers else "No major risk signals"


def recommend_action(row):

    if row.get("churn_risk_score", 0) >= 80 and row.get("renewal_upcoming_60d", 0) == 1:
        return "Immediate executive escalation and renewal rescue"

    if row.get("churn_risk_score", 0) >= 60:
        return "Proactive retention outreach and issue resolution"

    if row.get("usage_decline_12w", 0) == 1 and row.get("tickets_4w_sum", 0) == 0:
        return "Adoption coaching and success check-in"

    if row.get("sla_breach_4w_flag", 0) == 1 or row.get("escalations_12w_sum", 0) > 0:
        return "Resolve open escalations and service recovery"

    if row.get("billing_risk_flag", 0) == 1 and row.get("renewal_upcoming_60d", 0) == 1:
        return "Finance + CSM billing resolution outreach"

    if row.get("expansion_score", 0) >= 70 and row.get("churn_risk_score", 100) < 40:
        return "Initiate upsell / expansion motion"

    return "Monitor – no immediate action"


def assign_owner(action):
    action = action.lower()

    if "billing" in action:
        return "Finance"
    if "support" in action or "escalation" in action:
        return "Support"
    if "upsell" in action or "expansion" in action:
        return "Sales"
    if "executive" in action:
        return "Leadership"

    return "CSM"


def build_agent_table_for_week(df, selected_week):

    week_df = df[df["week_start"] == selected_week].copy()

    week_df["next_best_action"] = week_df.apply(recommend_action, axis=1)
    week_df["action_owner"] = week_df["next_best_action"].apply(assign_owner)
    week_df["action_reason"] = week_df.apply(get_churn_drivers, axis=1)

    week_df["immediate_attention_flag"] = (
        (week_df["churn_risk_score"] >= 60) |
        (week_df["billing_risk_flag"] == 1) |
        (week_df["renewal_upcoming_30d"] == 1)
    ).astype(int)

    return week_df.sort_values(
        ["immediate_attention_flag", "churn_risk_score"],
        ascending=[False, False]
    )


# ==========================================================
# ------------------ SAFE JSON HELPER ----------------------
# ==========================================================

def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if pd.isna(obj):
        return None
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj

import json
from openai import OpenAI

client = OpenAI()  # assumes API key is set via secrets/env


# ==========================================================
# ------------------ AI PORTFOLIO SUMMARY ------------------
# ==========================================================

def generate_portfolio_summary(kpi_row, risk_df):

    # --- Reduce columns to only important ones ---
    important_cols = [
        col for col in [
            "account_name",
            "churn_risk_score",
            "expansion_score",
            "usage_trend",
            "csat",
            "revenue"
        ] if col in risk_df.columns
    ]

    top_risk_accounts = (
        risk_df
        .sort_values("churn_risk_score", ascending=False)
        [important_cols]
        .head(5)
    )

    context = {
        "portfolio_kpis": make_json_safe(kpi_row.to_dict()),
        "top_5_risk_accounts": make_json_safe(
            top_risk_accounts.to_dict(orient="records")
        )
    }

    prompt = f"""
You are a VP-level AI Strategy Copilot.

Below is a summarized portfolio snapshot.

Portfolio Data:
{json.dumps(context, indent=2)}

Provide:
1. Portfolio health summary (executive level)
2. Key churn drivers
3. Expansion opportunities
4. Immediate strategic priorities (bullet format)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt[:12000]}],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI Summary Error: {str(e)}"


# ==========================================================
# ------------------ AI CUSTOMER SUMMARY -------------------
# ==========================================================

def generate_customer_summary(customer_row):

    # Reduce noise: keep only non-null & relevant fields
    customer_dict = {
        k: v for k, v in make_json_safe(customer_row.to_dict()).items()
        if v is not None
    }

    prompt = f"""
You are a Senior Customer Success AI Copilot.

Customer Snapshot:
{json.dumps(customer_dict, indent=2)}

Provide:
1. Executive summary
2. Key risks (if any)
3. Expansion potential
4. Recommended next action (clear & actionable)
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt[:12000]}],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI Summary Error: {str(e)}"

