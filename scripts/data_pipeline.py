import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================================
# CONFIG
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Problem_Statement_3.xlsx"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
MASTER_DIR = DATA_DIR / "master"
MODEL_DIR = BASE_DIR / "models"

# ==========================================================
# LOAD RAW DATA
# ==========================================================

def load_data(DATA_PATH):

    customer = pd.read_excel(DATA_PATH, sheet_name="Customer_Master")
    usage = pd.read_excel(DATA_PATH, sheet_name="Usage_Weekly")
    support = pd.read_excel(DATA_PATH, sheet_name="Support_Weekly")
    crm = pd.read_excel(DATA_PATH, sheet_name="CRM_Activity_Weekly")
    billing = pd.read_excel(DATA_PATH, sheet_name="Billing_Invoices")

    return customer, usage, support, crm, billing

# ==========================================================
# BUILD PORTFOLIO TABLE
# ==========================================================

def build_portfolio(customer, usage, support, crm, billing):

    usage["week_start"] = pd.to_datetime(usage["week_start"])
    support["week_start"] = pd.to_datetime(support["week_start"])
    crm["week_start"] = pd.to_datetime(crm["week_start"])

    customer["contract_start_date"] = pd.to_datetime(customer["contract_start_date"])
    customer["renewal_date"] = pd.to_datetime(customer["renewal_date"])

    billing["invoice_date"] = pd.to_datetime(billing["invoice_date"])
    billing["due_date"] = pd.to_datetime(billing["due_date"])
    billing["paid_date"] = pd.to_datetime(billing["paid_date"])

    portfolio = usage.copy()

    portfolio = portfolio.merge(customer, on="customer_id", how="left")
    portfolio = portfolio.merge(support, on=["customer_id", "week_start"], how="left", suffixes=("", "_support"))
    portfolio = portfolio.merge(crm, on=["customer_id", "week_start"], how="left", suffixes=("", "_crm"))

    billing = billing.sort_values(["customer_id", "invoice_date"])

    def latest_invoice_asof(portfolio_df, billing_df):
        out = []
        for _, r in portfolio_df[["customer_id", "week_start"]].iterrows():
            inv = billing_df[
                (billing_df["customer_id"] == r["customer_id"]) &
                (billing_df["invoice_date"] <= r["week_start"])
            ].tail(1)
            if len(inv) == 0:
                out.append(pd.Series())
            else:
                out.append(inv.iloc[0])
        return pd.DataFrame(out)

    billing_weekly = latest_invoice_asof(portfolio, billing)
    billing_weekly = billing_weekly.drop(columns=["customer_id"], errors="ignore")

    portfolio = pd.concat(
        [portfolio.reset_index(drop=True),
         billing_weekly.reset_index(drop=True)],
        axis=1
    )

    portfolio = portfolio.loc[:, ~portfolio.columns.duplicated()]
    portfolio = portfolio.sort_values(["customer_id", "week_start"]).reset_index(drop=True)

    return portfolio

# ==========================================================
# PREPROCESSING
# ==========================================================

def preprocess(portfolio):
    
    numeric_cols = portfolio.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = portfolio.select_dtypes(include="object").columns.tolist()
    categorical_cols = [c for c in categorical_cols if c not in ["customer_id"]]

    portfolio = portfolio.sort_values(["customer_id", "week_start"]).reset_index(drop=True)

    portfolio[numeric_cols] = (
        portfolio.groupby("customer_id")[numeric_cols].ffill()
    )

    portfolio[categorical_cols] = (
        portfolio.groupby("customer_id")[categorical_cols].ffill()
    )

    portfolio[numeric_cols] = (
        portfolio.groupby("customer_id")[numeric_cols].bfill()
    )

    portfolio[categorical_cols] = (
        portfolio.groupby("customer_id")[categorical_cols].bfill()
    )

    for col in numeric_cols:
        portfolio[col] = portfolio[col].fillna(portfolio[col].mean())

    for col in categorical_cols:
        if portfolio[col].isna().any():
            portfolio[col].fillna(portfolio[col].mode().iloc[0],inplace=True)
    
    return portfolio

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def feature_engineering(portfolio):
    
    portfolio["is_overdue_flag"] = (portfolio["days_overdue"] > 0).astype(int)
    portfolio["is_severely_overdue_flag"] = (portfolio["days_overdue"] > 30).astype(int)
    portfolio["unresolved_overdue_flag"] = ((portfolio["days_overdue"] > 0) & (portfolio["paid_date"].isna())).astype(int)
    portfolio["overdue_severity"] = (portfolio["days_overdue"].clip(lower=0, upper=90))
    portfolio["billing_risk_flag"] = ((portfolio["unresolved_overdue_flag"] == 1) | (portfolio["billing_issue_flag"] == 1)).astype(int)

    def billing_state(row):
        if row["days_overdue"] <= 0:
            return "Not Due"
        if row["days_overdue"] > 0 and not pd.isna(row["paid_date"]):
            return "Paid Late"
        if row["days_overdue"] > 0 and pd.isna(row["paid_date"]):
            return "Overdue Unpaid"
        return "Unknown"

    portfolio["billing_state"] = portfolio.apply(billing_state, axis=1)

    portfolio = portfolio.sort_values(["customer_id", "week_start"])

    portfolio["ever_overdue_4w"] = (portfolio.groupby("customer_id")["is_overdue_flag"].rolling(4).max().reset_index(level=0, drop=True))
    portfolio["ever_unresolved_overdue_12w"] = (portfolio.groupby("customer_id")["unresolved_overdue_flag"].rolling(12).max().reset_index(level=0, drop=True))
    portfolio["max_overdue_12w"] = (portfolio.groupby("customer_id")["overdue_severity"].rolling(12).max().reset_index(level=0, drop=True))

    portfolio = portfolio.sort_values(["customer_id", "week_start"]).reset_index(drop=True)

    def rolling_stats(df, group_col, target_col, window, prefix):
        rolled = (
            df
            .groupby(group_col)[target_col]
            .rolling(window, min_periods=window)
            .agg(["mean", "std", "min", "max"])
            .reset_index(level=0, drop=True)
        )
        rolled.columns = [f"{prefix}_{c}_{window}w" for c in rolled.columns]
        return rolled

    portfolio = pd.concat([
        portfolio,
        rolling_stats(portfolio, "customer_id", "usage_score", 4, "usage_score"),
        rolling_stats(portfolio, "customer_id", "usage_score", 12, "usage_score")
    ], axis=1)

    portfolio["active_users_4w_avg"] = (
        portfolio.groupby("customer_id")["active_users"]
        .rolling(4, min_periods=4)
        .mean()
        .reset_index(level=0, drop=True)
    )

    portfolio["sessions_4w_avg"] = (
        portfolio.groupby("customer_id")["sessions"]
        .rolling(4, min_periods=4)
        .mean()
        .reset_index(level=0, drop=True)
    )

    feature_cols = [
        "feature_adoption_core",
        "feature_adoption_analytics",
        "feature_adoption_automation",
        "feature_adoption_integrations",
        "feature_adoption_ai_assist"
    ]

    portfolio["avg_feature_adoption"] = portfolio[feature_cols].mean(axis=1)
    portfolio["min_feature_adoption"] = portfolio[feature_cols].min(axis=1)

    portfolio["avg_feature_adoption_4w"] = (
        portfolio.groupby("customer_id")["avg_feature_adoption"]
        .rolling(4, min_periods=4)
        .mean()
        .reset_index(level=0, drop=True)
    )

    portfolio["min_feature_adoption_4w"] = (
        portfolio.groupby("customer_id")["min_feature_adoption"]
        .rolling(4, min_periods=4)
        .mean()
        .reset_index(level=0, drop=True)
    )

    portfolio["num_features_above_30pct"] = (
        (portfolio[feature_cols] > 0.30).sum(axis=1)
    )

    portfolio["tickets_4w_sum"] = (
        portfolio.groupby("customer_id")["tickets_created"]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )

    portfolio["tickets_12w_avg"] = (
        portfolio.groupby("customer_id")["tickets_created"]
        .rolling(12, min_periods=12)
        .mean()
        .reset_index(level=0, drop=True)
    )

    portfolio["escalations_12w_sum"] = (
        portfolio.groupby("customer_id")["escalations"]
        .rolling(12, min_periods=12)
        .sum()
        .reset_index(level=0, drop=True)
    )

    portfolio["sla_breach_4w_flag"] = (
        portfolio.groupby("customer_id")["sla_breaches"]
        .rolling(4, min_periods=4)
        .max()
        .reset_index(level=0, drop=True)
    )

    portfolio["csat_4w_avg"] = (
        portfolio.groupby("customer_id")["csat_score"]
        .rolling(4, min_periods=4)
        .mean()
        .reset_index(level=0, drop=True)
    )

    portfolio["csat_12w_avg"] = (
        portfolio.groupby("customer_id")["csat_score"]
        .rolling(12, min_periods=12)
        .mean()
        .reset_index(level=0, drop=True)
    )

    portfolio["csat_12w_std"] = (
        portfolio.groupby("customer_id")["csat_score"]
        .rolling(12, min_periods=12)
        .std()
        .reset_index(level=0, drop=True)
    )

    portfolio["reopen_rate_12w_avg"] = (
        portfolio.groupby("customer_id")["reopen_rate"]
        .rolling(12, min_periods=12)
        .mean()
        .reset_index(level=0, drop=True)
    )

    portfolio["avg_resolution_12w_avg"] = (
        portfolio.groupby("customer_id")["avg_resolution_hrs"]
        .rolling(12, min_periods=12)
        .mean()
        .reset_index(level=0, drop=True)
    )

    portfolio["emails_4w_sum"] = (
        portfolio.groupby("customer_id")["emails_sent"]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )

    portfolio["meetings_4w_sum"] = (
        portfolio.groupby("customer_id")["meetings_held"]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )

    portfolio["calls_4w_sum"] = (
        portfolio.groupby("customer_id")["calls_made"]
        .rolling(4, min_periods=4)
        .sum()
        .reset_index(level=0, drop=True)
    )

    portfolio["crm_touch"] = (
        portfolio["emails_sent"]
        + portfolio["meetings_held"]
        + portfolio["calls_made"]
    )

    portfolio = portfolio.sort_values(["customer_id", "week_start"])

    portfolio["crm_touch_12w_sum"] = (
        portfolio
        .groupby("customer_id")["crm_touch"]
        .rolling(12, min_periods=12)
        .sum()
        .reset_index(level=0, drop=True)
    )

    portfolio["had_meeting"] = (portfolio["meetings_held"] > 0).astype(int)

    portfolio["week_idx"] = portfolio.groupby("customer_id").cumcount()

    portfolio["last_meeting_week_idx"] = (
        portfolio["week_idx"]
        .where(portfolio["had_meeting"] == 1)
        .groupby(portfolio["customer_id"])
        .ffill()
    )

    portfolio["weeks_since_last_meeting"] = (
        portfolio["week_idx"] - portfolio["last_meeting_week_idx"]
    )

    portfolio.loc[
        portfolio["last_meeting_week_idx"].isna(),
        "weeks_since_last_meeting"
    ] = np.nan

    portfolio["qbr_completed_last_12w"] = (
        portfolio
        .groupby("customer_id")["qbr_completed"]
        .rolling(12, min_periods=12)
        .max()
        .reset_index(level=0, drop=True)
    )

    portfolio.drop(
        columns=["had_meeting", "week_idx", "last_meeting_week_idx"],
        inplace=True
    )

    portfolio["raw_days_to_renewal"] = (
        portfolio["renewal_date"] - portfolio["week_start"]
    ).dt.days

    portfolio["days_to_renewal_capped"] = (
        portfolio["raw_days_to_renewal"]
        .clip(lower=-365, upper=365)
    )

    def renewal_state(row):
        if row["raw_days_to_renewal"] >= 0:
            return "Upcoming"
        if row["raw_days_to_renewal"] < 0 and row["billing_risk_flag"] == 0:
            return "Past_Renewal_But_Active"
        if row["raw_days_to_renewal"] < 0 and row["billing_risk_flag"] == 1:
            return "Past_Renewal_At_Risk"
        return "Unknown"

    portfolio["renewal_state"] = portfolio.apply(renewal_state, axis=1)

    portfolio["renewal_upcoming_30d"] = (
        (portfolio["raw_days_to_renewal"] >= 0) &
        (portfolio["raw_days_to_renewal"] <= 30)
    ).astype(int)

    portfolio["renewal_upcoming_60d"] = (
        (portfolio["raw_days_to_renewal"] >= 0) &
        (portfolio["raw_days_to_renewal"] <= 60)
    ).astype(int)

    portfolio["past_renewal_with_billing_risk"] = (
        (portfolio["raw_days_to_renewal"] < 0) &
        (portfolio["billing_risk_flag"] == 1)
    ).astype(int)

    portfolio["contract_age_days"] = (
        portfolio["week_start"] - portfolio["contract_start_date"]
    ).dt.days.clip(lower=0)

    portfolio = portfolio.loc[:, ~portfolio.columns.duplicated()]

    return portfolio

# ==========================================================
# MODEL ELIGIBILITY
# ==========================================================

def get_eligible(portfolio):

    required_12w_cols = [c for c in portfolio.columns if "_12w" in c]
    eligibility_mask = portfolio[required_12w_cols].notna().all(axis=1)

    portfolio_eligible = portfolio.loc[eligibility_mask].copy()

    portfolio_eligible["usage_decline_12w"] = (
        portfolio_eligible["usage_score_mean_12w"]
        - portfolio_eligible["usage_score_mean_4w"]
    )

    portfolio_eligible["usage_volatility"] = (
        portfolio_eligible["usage_score_std_12w"] /
        (portfolio_eligible["usage_score_mean_12w"] + 1)
    )

    portfolio_eligible["usage_floor_breach"] = (
    portfolio_eligible["usage_score_min_4w"] < 0.5 * portfolio_eligible["usage_score_mean_12w"]
    ).astype(int)

    return portfolio_eligible

# ==========================================================
# RUN DATA PROCESSING PIPELINE
# ==========================================================

def run_data_pipeline():

    customer, usage, support, crm, billing = load_data(RAW_DATA_PATH)

    portfolio = build_portfolio(customer, usage, support, crm, billing)
    portfolio = preprocess(portfolio)
    portfolio = feature_engineering(portfolio)

    # Save portfolio (intermediate)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

    portfolio_path = INTERMEDIATE_DIR / "portfolio.parquet"
    portfolio.to_parquet(portfolio_path, index=False)
    print(f"Saved portfolio → {portfolio_path}")

    portfolio_eligible = get_eligible(portfolio)
    eligible_path = INTERMEDIATE_DIR / "portfolio_eligible.parquet"
    portfolio_eligible.to_parquet(eligible_path, index=False)

    print(f"Saved portfolio_eligible → {eligible_path}")

    return portfolio, portfolio_eligible


if __name__ == "__main__":
    run_data_pipeline()
