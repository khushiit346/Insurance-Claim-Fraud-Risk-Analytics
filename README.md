"""
scoring.py
----------
Turns two independent anomaly scores + one hard rule into a single,
business-usable output: a confidence tier per claim, plus portfolio-level
financial exposure metrics that translate the model's output into numbers
a risk/audit team actually cares about.

Confidence tiers (used instead of a single boolean is_anomaly flag):
  - CRITICAL : hard collusion rule fired (shared bank account, shared
               address, or shared last name between agent and customer).
               Always investigate regardless of model scores.
  - HIGH     : both the autoencoder AND the isolation forest flag the
               claim in their respective top percentile.
  - MEDIUM   : exactly one of the two models flags the claim.
  - LOW      : neither model flags the claim.
"""

import numpy as np
import pandas as pd

DEFAULT_PERCENTILE = 95


def add_percentile_flags(
    df: pd.DataFrame,
    ae_score: np.ndarray,
    iso_score: np.ndarray,
    percentile: int = DEFAULT_PERCENTILE,
) -> pd.DataFrame:
    df = df.copy()
    df["autoencoder_score"] = ae_score
    df["isolation_forest_score"] = iso_score

    ae_threshold = np.percentile(ae_score, percentile)
    iso_threshold = np.percentile(iso_score, percentile)

    df["ae_flag"] = df["autoencoder_score"] > ae_threshold
    df["iso_flag"] = df["isolation_forest_score"] > iso_threshold

    model_agreement = df["ae_flag"].astype(int) + df["iso_flag"].astype(int)

    conditions = [
        df["HARD_COLLUSION_FLAG"] == 1,
        model_agreement == 2,
        model_agreement == 1,
    ]
    choices = ["CRITICAL", "HIGH", "MEDIUM"]
    df["confidence_tier"] = np.select(conditions, choices, default="LOW")

    df["is_flagged"] = df["confidence_tier"] != "LOW"

    # A single normalized 0-1 composite score, handy for sorting/reporting.
    ae_norm = (df["autoencoder_score"] - ae_score.min()) / (
        ae_score.max() - ae_score.min()
    )
    iso_norm = (df["isolation_forest_score"] - iso_score.min()) / (
        iso_score.max() - iso_score.min()
    )
    df["composite_score"] = (ae_norm + iso_norm) / 2

    return df


def business_impact_summary(df: pd.DataFrame) -> dict:
    total_claims = len(df)
    total_value = df["CLAIM_AMOUNT"].sum()

    flagged = df[df["is_flagged"]]
    flagged_value = flagged["CLAIM_AMOUNT"].sum()

    tier_counts = df["confidence_tier"].value_counts().to_dict()
    tier_value = flagged.groupby("confidence_tier")["CLAIM_AMOUNT"].sum().to_dict()

    by_risk_segment = (
        df.groupby("RISK_SEGMENTATION")["is_flagged"].mean().sort_values(ascending=False)
    )
    by_insurance_type = (
        df.groupby("INSURANCE_TYPE")["is_flagged"].mean().sort_values(ascending=False)
    )

    summary = {
        "total_claims": total_claims,
        "total_portfolio_value": total_value,
        "flagged_claim_count": len(flagged),
        "flagged_claim_pct": len(flagged) / total_claims,
        "flagged_dollar_value": flagged_value,
        "flagged_dollar_pct": flagged_value / total_value,
        "tier_counts": tier_counts,
        "tier_dollar_value": tier_value,
        "flag_rate_by_risk_segment": by_risk_segment.to_dict(),
        "flag_rate_by_insurance_type": by_insurance_type.to_dict(),
    }
    return summary


def agent_risk_scorecard(df: pd.DataFrame, min_claims: int = 5) -> pd.DataFrame:
    """
    Aggregates claim-level flags up to the agent level. This is the view an
    internal investigations team would actually act on: which agents show a
    disproportionate rate of flagged claims, not just which individual
    claims look odd.
    """
    grouped = df.groupby("AGENT_ID").agg(
        total_claims=("TRANSACTION_ID", "count"),
        flagged_claims=("is_flagged", "sum"),
        critical_flags=("confidence_tier", lambda s: (s == "CRITICAL").sum()),
        total_claim_value=("CLAIM_AMOUNT", "sum"),
        flagged_claim_value=(
            "CLAIM_AMOUNT",
            lambda s: s[df.loc[s.index, "is_flagged"]].sum(),
        ),
        avg_composite_score=("composite_score", "mean"),
    )
    grouped["flag_rate"] = grouped["flagged_claims"] / grouped["total_claims"]
    grouped = grouped[grouped["total_claims"] >= min_claims]
    return grouped.sort_values(
        ["critical_flags", "flag_rate"], ascending=False
    )
