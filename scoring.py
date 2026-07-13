"""
visualize.py
------------
Generates the chart set used in the executive summary / README. Kept as
plain matplotlib (no seaborn dependency) so the project has a minimal
dependency footprint.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["font.size"] = 11

TIER_COLORS = {
    "CRITICAL": "#B3261E",
    "HIGH": "#E8A33D",
    "MEDIUM": "#4C78A8",
    "LOW": "#B0B7C0",
}


def plot_score_distribution(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    scores = df["autoencoder_score"]
    threshold = df.loc[df["ae_flag"], "autoencoder_score"].min()
    # A handful of extreme outliers can compress the whole histogram into
    # one bar. Clip the display range to the 99.5th percentile so the bulk
    # of the distribution (and the threshold line) is actually visible;
    # the clipped count is annotated rather than silently hidden.
    display_cap = np.percentile(scores, 99.5)
    clipped = scores[scores <= display_cap]
    n_hidden = (scores > display_cap).sum()

    ax.hist(clipped, bins=60, color="#4C78A8", alpha=0.85)
    ax.axvline(threshold, color="#B3261E", linestyle="--", linewidth=1.5,
               label=f"95th percentile threshold ({threshold:.3f})")
    title = "Autoencoder Reconstruction Error Distribution"
    if n_hidden:
        title += f"\n({n_hidden} extreme outliers beyond the 99.5th pct not shown, for readability)"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Reconstruction error (anomaly score)")
    ax.set_ylabel("Number of claims")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confidence_tiers(df: pd.DataFrame, out_path: Path):
    order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    counts = df["confidence_tier"].value_counts().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(order, counts.values, color=[TIER_COLORS[t] for t in order])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(counts.values) * 0.01,
                f"{int(v):,}", ha="center", fontsize=10)
    ax.set_title("Claims by Investigation Priority Tier")
    ax.set_ylabel("Number of claims")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_flag_rate_by_group(df: pd.DataFrame, group_col: str, title: str, out_path: Path):
    rates = (
        df.groupby(group_col)["is_flagged"].mean().sort_values(ascending=False) * 100
    )
    overall = df["is_flagged"].mean() * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(rates.index.astype(str), rates.values, color="#4C78A8")
    ax.axhline(overall, color="#B3261E", linestyle="--", linewidth=1.5,
               label=f"Portfolio average ({overall:.1f}%)")
    for b, v in zip(bars, rates.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel("% of claims flagged")
    ax.legend()
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dollar_pareto(df: pd.DataFrame, out_path: Path):
    sorted_df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    cum_value_pct = sorted_df["CLAIM_AMOUNT"].cumsum() / sorted_df["CLAIM_AMOUNT"].sum() * 100
    claim_pct = (np.arange(1, len(sorted_df) + 1) / len(sorted_df)) * 100

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(claim_pct, cum_value_pct, color="#4C78A8", linewidth=2)
    ax.axvline(5, color="#B3261E", linestyle="--", linewidth=1.2, label="Top 5% of claims")
    value_at_5pct = cum_value_pct.iloc[int(len(sorted_df) * 0.05)]
    ax.axhline(value_at_5pct, color="#B3261E", linestyle=":", linewidth=1)
    ax.set_title("Cumulative Claim Dollar Value Captured, Ranked by Risk Score")
    ax.set_xlabel("% of claims reviewed (ranked highest-risk first)")
    ax.set_ylabel("% of total claim dollar value captured")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_agent_scorecard(scorecard: pd.DataFrame, out_path: Path, top_n: int = 10):
    top = scorecard.head(top_n).sort_values("flag_rate")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#B3261E" if c > 0 else "#4C78A8" for c in top["critical_flags"]]
    ax.barh(top.index, top["flag_rate"] * 100, color=colors)
    for i, (idx, row) in enumerate(top.iterrows()):
        ax.text(row["flag_rate"] * 100 + 0.5, i,
                f"{int(row['flagged_claims'])}/{int(row['total_claims'])} claims",
                va="center", fontsize=9)
    ax.set_title(f"Top {top_n} Agents by Flagged-Claim Rate (min. 5 claims)")
    ax.set_xlabel("% of this agent's claims flagged")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_all_charts(df: pd.DataFrame, scorecard: pd.DataFrame, charts_dir: Path):
    charts_dir.mkdir(parents=True, exist_ok=True)
    plot_score_distribution(df, charts_dir / "01_score_distribution.png")
    plot_confidence_tiers(df, charts_dir / "02_confidence_tiers.png")
    plot_flag_rate_by_group(
        df, "RISK_SEGMENTATION", "Flag Rate by Risk Segmentation",
        charts_dir / "03_flag_rate_by_risk_segment.png",
    )
    plot_flag_rate_by_group(
        df, "INSURANCE_TYPE", "Flag Rate by Insurance Type",
        charts_dir / "04_flag_rate_by_insurance_type.png",
    )
    plot_dollar_pareto(df, charts_dir / "05_dollar_exposure_pareto.png")
    plot_agent_scorecard(scorecard, charts_dir / "06_top_agents_scorecard.png")
