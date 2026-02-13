import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import RobustScaler

# ==========================================================
# CONFIG
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Problem_Statement_3.xlsx"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
MASTER_DIR = DATA_DIR / "master"
MODEL_DIR = BASE_DIR / "models"

CHURN_MODEL_PATH = MODEL_DIR / "churn_model_final.joblib"
EXPANSION_MODEL_PATH = MODEL_DIR / "expansion_model_final.joblib"

# ==========================================================
# CHURN PREDICTION
# ==========================================================

def run_churn_model(portfolio_eligible):
    portfolio_eligible_copy = portfolio_eligible.copy()
   
    artifacts = joblib.load(CHURN_MODEL_PATH)

    model = artifacts["model"]
    scaler = artifacts["scaler"]
    encoder = artifacts["encoder"]
    categorical_cols = artifacts["categorical_cols"]
    feature_columns = artifacts["feature_columns"]
    min_s = artifacts["anomaly_min"]
    max_s = artifacts["anomaly_max"]
    max_usage_12w = artifacts["max_usage_12w"]
    alpha = artifacts["alpha"]
    beta = artifacts["beta"]
    
    drop_cols = [
        "customer_id",
        "customer_name",
        "invoice_id",
        "csm_owner",
        "account_owner",

        "contract_start_date",
        "renewal_date",
        "invoice_date",
        "due_date",
        "paid_date",

        "invoice_status",
        "raw_days_to_renewal"
    ]

    portfolio_eligible = portfolio_eligible.drop(
    columns=[c for c in drop_cols if c in portfolio_eligible.columns],
    errors="ignore"
)
    
    numeric_cols = portfolio_eligible.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    encoded_cat = encoder.transform(portfolio_eligible[categorical_cols])

    encoded_cat_df = pd.DataFrame(
        encoded_cat,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=portfolio_eligible.index
    )

    X = pd.concat(
        [portfolio_eligible[numeric_cols], encoded_cat_df],
        axis=1
    )

    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )

    support_cols = [
    c for c in X_scaled.columns
    if "tickets" in c or "escalations" in c]

    X_test = X_scaled.copy()
    X_test[support_cols] = np.log1p(X_test[support_cols])

    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    anomaly_score = -model.score_samples(X_test)

    churn_risk_score_v1 = ((anomaly_score - min_s) / (max_s - min_s)) * 100

    portfolio_eligible["churn_risk_score_v1"] = churn_risk_score_v1

    portfolio_eligible["usage_risk"] = 1 - (
    portfolio_eligible["usage_score_mean_12w"] /
    max_usage_12w
)
    portfolio_eligible["usage_risk"] = portfolio_eligible["usage_risk"].clip(0, 1)

    portfolio_eligible["churn_risk_score_v2"] = (
        alpha * portfolio_eligible["churn_risk_score_v1"] +
        beta  * (portfolio_eligible["usage_risk"] * 100)
    )
    churn_prediction = pd.DataFrame({
        "week_start": portfolio_eligible_copy["week_start"],
        "customer_id": portfolio_eligible_copy["customer_id"],
        "churn_risk_score": portfolio_eligible["churn_risk_score_v2"],
    })
    
    return churn_prediction, portfolio_eligible

# ==========================================================
# EXPANSION PREDICTION
# ==========================================================

def run_expansion_model(portfolio_eligible):

    artifacts = joblib.load(EXPANSION_MODEL_PATH)

    readiness_features = artifacts["readiness_features"]
    scaler = artifacts["scaler"]
    pca = artifacts["pca"]
    raw_min = artifacts["raw_min"]
    raw_max = artifacts["raw_max"]
    anchor_feature = artifacts["anchor_feature"]

    X_new = portfolio_eligible[readiness_features].copy()

    for c in ["sessions_4w_avg", "active_users_4w_avg"]:
        if c in X_new.columns:
            X_new[c] = np.log1p(X_new[c])

    invert_cols = [
        "csat_12w_std",
        "reopen_rate_12w_avg",
        "avg_resolution_12w_avg",
        "billing_risk_flag",
        "sla_breach_4w_flag",
        "escalations_12w_sum"
    ]

    for c in invert_cols:
        if c in X_new.columns:
            X_new[c] = -1 * X_new[c]

    X_scaled = scaler.transform(X_new)
    
    X_scaled = pd.DataFrame(
        X_scaled,
        columns=readiness_features,
        index=portfolio_eligible.index
    )

    raw = pca.transform(X_scaled).ravel()

    if np.corrcoef(raw, portfolio_eligible[anchor_feature])[0, 1] < 0:
        raw *= -1

    portfolio_eligible["expansion_score"] = (
        (raw - raw_min) / (raw_max - raw_min) * 100
    ).clip(0, 100)

    expansion_prediction = pd.DataFrame({
    "week_start": portfolio_eligible["week_start"],
    "customer_id": portfolio_eligible["customer_id"],
    "expansion_score": portfolio_eligible["expansion_score"]
})

    return expansion_prediction, portfolio_eligible

# =======================================================================
# FLAG PREDICTION, PORTFOLIO KPI and INCLUDING ADDITIONAL MODELLING FEATURES
# =====================================================================

def flags_kpi_portfolio_enriched(
    churn_prediction,
    expansion_prediction,
    portfolio,
    portfolio_eligible,
    portfolio_churn,
    portfolio_expansion
):

    CHURN_SCORE_TH = 65
    EXPANSION_SCORE_TH = 65
    USAGE_BAD_TH = 42
    USAGE_GOOD_TH = 45
    CSAT_BAD_TH = 3.10
    CSAT_GOOD_TH = 3.15
    RESOLUTION_BAD_TH = 52

    final_predictions = churn_prediction.merge(
        expansion_prediction[
            ["customer_id", "week_start", "expansion_score"]
        ],
        on=["customer_id", "week_start"],
        how="left",
        validate="one_to_one"
    )
    portfolio_new = portfolio.merge(
        final_predictions,
        on=["week_start", "customer_id"],
        how="left",
        validate="one_to_one"
    )

    portfolio_new["churn_flag"] = np.where(
        portfolio_new["churn_risk_score"].notna(),
        (
            (portfolio_new["churn_risk_score"] >= CHURN_SCORE_TH) &
            (
                (portfolio_new["usage_score"] <= USAGE_BAD_TH) |
                (portfolio_new["csat_score"] <= CSAT_BAD_TH) |
                (portfolio_new["avg_resolution_hrs"] >= RESOLUTION_BAD_TH) |
                (portfolio_new["sla_breaches"] >= 1) |
                (portfolio_new["escalations"] >= 1)
            )
        ).astype(int),
        np.nan
    )

    portfolio_new["expansion_flag"] = np.where(
        portfolio_new["expansion_score"].notna(),
        (
            (portfolio_new["expansion_score"] >= EXPANSION_SCORE_TH) &
            (portfolio_new["usage_score"] >= USAGE_GOOD_TH) &
            (portfolio_new["csat_score"] >= CSAT_GOOD_TH) &
            (portfolio_new["avg_resolution_hrs"] <= RESOLUTION_BAD_TH - 1)
        ).astype(int),
        np.nan
    )

    final_predictions = portfolio_new.copy()

    portfolio_kpi_weekly = (
        final_predictions
        .groupby("week_start")
        .agg(
            total_customers=("customer_id", "nunique"),

            # Usage & CX
            avg_usage_score=("usage_score", "mean"),
            avg_csat_score=("csat_score", "mean"),

            # Support load
            total_tickets=("tickets_created", "sum"),
            total_escalations=("escalations", "sum"),

            avg_first_response_hrs=("avg_first_response_hrs", "mean"),
            avg_resolution_hrs=("avg_resolution_hrs", "mean"),

            # Engagement
            portfolio_emails_sent=("emails_sent", "sum"),
            portfolio_meetings_held=("meetings_held", "sum"),

            # SLA
            portfolio_sla_breaches=("sla_breaches", "sum"),
            portfolio_tickets_created=("tickets_created", "sum"),

            # Model-driven counts
            customers_at_churn_risk=("churn_flag", "sum"),
            customers_expansion_eligible=("expansion_flag", "sum"),

            # Model eligible count (non-null scores)
            customers_model_eligible=("churn_flag", "count")
        )
        .reset_index()
        .sort_values("week_start")
        .reset_index(drop=True)
    )

    portfolio_kpi_weekly["sla_breach_rate"] = (
        portfolio_kpi_weekly["portfolio_sla_breaches"]
        / portfolio_kpi_weekly["portfolio_tickets_created"]
    )

    portfolio_kpi_weekly["churn_risk_rate_pct"] = (
        portfolio_kpi_weekly["customers_at_churn_risk"]
        / portfolio_kpi_weekly["customers_model_eligible"]
    ) * 100

    portfolio_kpi_weekly["expansion_rate_pct"] = (
        portfolio_kpi_weekly["customers_expansion_eligible"]
        / portfolio_kpi_weekly["customers_model_eligible"]
    ) * 100

    portfolio_kpi_weekly.drop(
        columns=[
            "portfolio_sla_breaches",
            "portfolio_tickets_created"
        ],
        inplace=True
    )
    portfolio_churn["customer_id"] = portfolio_eligible["customer_id"]
    
    def safe_enrich_merge(
        base_df: pd.DataFrame,
        enrich_df: pd.DataFrame,
        on_cols=("customer_id", "week_start"),
    ):
        base_cols = set(base_df.columns)

        enrich_cols = [
            c for c in enrich_df.columns
            if c not in base_cols and c not in on_cols
        ]

        if not enrich_cols:
            return base_df

        return base_df.merge(
            enrich_df[list(on_cols) + enrich_cols],
            on=list(on_cols),
            how="left",
            validate="one_to_one"
        )

    final_predictions_enriched = final_predictions.copy()

    final_predictions_enriched = safe_enrich_merge(
        final_predictions_enriched,
        portfolio_eligible
    )

    final_predictions_enriched = safe_enrich_merge(
        final_predictions_enriched,
        portfolio_churn
    )

    final_predictions_enriched = safe_enrich_merge(
        final_predictions_enriched,
        portfolio_expansion
    )

    cols_to_drop = [
        c for c in ["churn_risk_score_v1", "churn_risk_score_v2"]
        if c in final_predictions_enriched.columns
    ]

    final_predictions_enriched.drop(columns=cols_to_drop, inplace=True)

    return portfolio_kpi_weekly, final_predictions_enriched


# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main():
    portfolio_path = INTERMEDIATE_DIR / "portfolio.parquet"
    portfolio = pd.read_parquet(portfolio_path)

    eligible_path = INTERMEDIATE_DIR / "portfolio_eligible.parquet"
    portfolio_eligible = pd.read_parquet(eligible_path)

    churn_prediction, portfolio_churn = run_churn_model(portfolio_eligible)
    expansion_prediction, portfolio_expansion = run_expansion_model(portfolio_eligible)
    portfolio_kpi_weekly, final_predictions_enriched = flags_kpi_portfolio_enriched(churn_prediction, expansion_prediction, portfolio, portfolio_eligible, portfolio_churn, portfolio_expansion)

    output_path = MASTER_DIR / "portfolio_kpi_weekly.parquet"
    portfolio_kpi_weekly.to_parquet(output_path, index=False)

    output_path = MASTER_DIR / "final_predictions_enriched.parquet"
    final_predictions_enriched.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
