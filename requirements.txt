"""
data_pipeline.py
-----------------
Loads the three raw source tables, joins them into a single claim-level
record, and engineers the red-flag features used by the fraud model.

Fixes vs. the original notebook:
  1. Robust file paths (relative to project root, not the working directory).
  2. AGENT_CUST_SAME_BANK never fired on this dataset (0 matches out of
     10,000 rows) -- kept as a safety net, but it is no longer the *only*
     collusion signal.
  3. Two new collusion signals that DO fire on real data:
       - AGENT_CUST_SAME_ADDRESS (exact street address match; stronger
         evidence than just sharing a ZIP code)
       - AGENT_CUST_SAME_LASTNAME (possible undisclosed family relationship)
  4. The vendor postal code was already being merged in the original code
     but was never actually compared to anything -- it was dead data.
     Added AGENT_VENDOR_SAME_ZIP and CUSTOMER_VENDOR_SAME_ZIP, which
     capture a classic "agent steers claims to a kickback vendor" and
     "customer and repair vendor are suspiciously close" pattern.
  5. Monetary fields are heavily right-skewed (CLAIM_AMOUNT ranges from
     $100 to $100,000). A log1p transform is applied before scaling so
     the autoencoder isn't dominated by a handful of large-dollar claims.
"""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}\n"
            f"Place insurance_data.csv, employee_data.csv and vendor_data.csv "
            f"inside {DATA_DIR}"
        )
    return path


def load_raw_tables(data_dir: Path = DATA_DIR):
    insurance = pd.read_csv(_require_file(data_dir / "insurance_data.csv"))
    employee = pd.read_csv(_require_file(data_dir / "employee_data.csv"))
    vendor = pd.read_csv(_require_file(data_dir / "vendor_data.csv"))
    return insurance, employee, vendor


def _last_name(full_name: str):
    if pd.isna(full_name):
        return None
    parts = str(full_name).strip().split()
    return parts[-1].lower() if parts else None


def load_and_merge_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    insurance, employee, vendor = load_raw_tables(data_dir)

    df = insurance.merge(
        employee[
            [
                "AGENT_ID",
                "AGENT_NAME",
                "ADDRESS_LINE1",
                "POSTAL_CODE",
                "EMP_ROUTING_NUMBER",
                "EMP_ACCT_NUMBER",
            ]
        ],
        on="AGENT_ID",
        how="left",
        suffixes=("_CUST", "_EMP"),
    )

    df = df.merge(
        vendor[["VENDOR_ID", "POSTAL_CODE"]],
        on="VENDOR_ID",
        how="left",
    )
    df.rename(columns={"POSTAL_CODE": "POSTAL_CODE_VNDR"}, inplace=True)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Timing red flags ---
    df["LOSS_DT"] = pd.to_datetime(df["LOSS_DT"])
    df["POLICY_EFF_DT"] = pd.to_datetime(df["POLICY_EFF_DT"])
    df["REPORT_DT"] = pd.to_datetime(df["REPORT_DT"])

    df["DAYS_POLICY_TO_LOSS"] = (df["LOSS_DT"] - df["POLICY_EFF_DT"]).dt.days
    df["DAYS_LOSS_TO_REPORT"] = (df["REPORT_DT"] - df["LOSS_DT"]).dt.days

    # --- Agent <-> Customer collusion signals ---
    df["AGENT_CUST_SAME_ZIP"] = (
        df["POSTAL_CODE_CUST"] == df["POSTAL_CODE_EMP"]
    ).astype(int)

    df["AGENT_CUST_SAME_ADDRESS"] = (
        df["ADDRESS_LINE1_CUST"] == df["ADDRESS_LINE1_EMP"]
    ).astype(int)

    df["AGENT_CUST_SAME_BANK"] = (
        (df["ROUTING_NUMBER"] == df["EMP_ROUTING_NUMBER"])
        | (df["ACCT_NUMBER"] == df["EMP_ACCT_NUMBER"])
    ).astype(int)

    cust_last = df["CUSTOMER_NAME"].apply(_last_name)
    agent_last = df["AGENT_NAME"].apply(_last_name)
    df["AGENT_CUST_SAME_LASTNAME"] = (
        cust_last.notna() & agent_last.notna() & (cust_last == agent_last)
    ).astype(int)

    # --- Agent / Customer <-> Vendor collusion signals (previously unused) ---
    df["AGENT_VENDOR_SAME_ZIP"] = (
        df["POSTAL_CODE_EMP"] == df["POSTAL_CODE_VNDR"]
    ).astype(int)
    df["CUSTOMER_VENDOR_SAME_ZIP"] = (
        df["POSTAL_CODE_CUST"] == df["POSTAL_CODE_VNDR"]
    ).astype(int)

    # --- Skew correction on monetary fields ---
    df["LOG_CLAIM_AMOUNT"] = np.log1p(df["CLAIM_AMOUNT"])
    df["LOG_PREMIUM_AMOUNT"] = np.log1p(df["PREMIUM_AMOUNT"])

    # A single hard-rule collusion flag, used later to force-flag a claim
    # regardless of what the model thinks.
    df["HARD_COLLUSION_FLAG"] = (
        (df["AGENT_CUST_SAME_BANK"] == 1)
        | (df["AGENT_CUST_SAME_ADDRESS"] == 1)
        | (df["AGENT_CUST_SAME_LASTNAME"] == 1)
    ).astype(int)

    return df


NUM_FEATURES = [
    "LOG_PREMIUM_AMOUNT",
    "LOG_CLAIM_AMOUNT",
    "AGE",
    "TENURE",
    "DAYS_POLICY_TO_LOSS",
    "DAYS_LOSS_TO_REPORT",
    "AGENT_CUST_SAME_ZIP",
    "AGENT_CUST_SAME_ADDRESS",
    "AGENT_CUST_SAME_BANK",
    "AGENT_CUST_SAME_LASTNAME",
    "AGENT_VENDOR_SAME_ZIP",
    "CUSTOMER_VENDOR_SAME_ZIP",
]

CAT_FEATURES = [
    "INSURANCE_TYPE",
    "EMPLOYMENT_STATUS",
    "INCIDENT_SEVERITY",
    "RISK_SEGMENTATION",
    "SOCIAL_CLASS",
]


def build_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    raw = load_and_merge_data(data_dir)
    return engineer_features(raw)
