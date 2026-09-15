import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_previous_applications(prev_app: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates previous_application.csv into one row per applicant using
    compact, interpretable previous-application history features.
    """

    prev_app = prev_app.copy()

    # 1. Drop high-missingness and less relevant columns
    drop_cols = [
        "RATE_INTEREST_PRIMARY",
        "RATE_INTEREST_PRIVILEGED",
        "AMT_DOWN_PAYMENT",
        "RATE_DOWN_PAYMENT",
        "NAME_TYPE_SUITE",
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
        "NFLAG_INSURED_ON_APPROVAL",
    ]

    drop_cols = [col for col in drop_cols if col in prev_app.columns]
    prev_app = prev_app.drop(columns=drop_cols)

    # 2. Convert DAYS_DECISION into years since previous application decision
    prev_app["YEARS_SINCE_PREVIOUS_APPLICATION"] = (
        -prev_app["DAYS_DECISION"] / 365
    )

    # 3. Create previous application outcome indicators
    prev_app["IS_APPROVED_APPLICATION"] = (
        prev_app["NAME_CONTRACT_STATUS"] == "Approved"
    ).astype(int)

    prev_app["IS_REFUSED_APPLICATION"] = (
        prev_app["NAME_CONTRACT_STATUS"] == "Refused"
    ).astype(int)

    # 4. Aggregate previous applications into one row per applicant
    prev_agg = prev_app.groupby("SK_ID_CURR").agg(
        TOTAL_PREVIOUS_APPLICATIONS=("SK_ID_PREV", "count"),
        APPROVED_PREVIOUS_APPLICATIONS=("IS_APPROVED_APPLICATION", "sum"),
        REFUSED_PREVIOUS_APPLICATIONS=("IS_REFUSED_APPLICATION", "sum"),
        AVG_PREVIOUS_APPLICATION_AMOUNT=("AMT_APPLICATION", "mean"),
        AVG_PREVIOUS_CREDIT_AMOUNT=("AMT_CREDIT", "mean"),
        MAX_PREVIOUS_CREDIT_AMOUNT=("AMT_CREDIT", "max"),
        MOST_RECENT_PREVIOUS_APPLICATION_YEARS=(
            "YEARS_SINCE_PREVIOUS_APPLICATION",
            "min"
        )
    ).reset_index()

    # 5. Create approval and refusal rates
    prev_agg["PREVIOUS_APPROVAL_RATE"] = (
        prev_agg["APPROVED_PREVIOUS_APPLICATIONS"] /
        prev_agg["TOTAL_PREVIOUS_APPLICATIONS"]
    )

    prev_agg["PREVIOUS_REFUSAL_RATE"] = (
        prev_agg["REFUSED_PREVIOUS_APPLICATIONS"] /
        prev_agg["TOTAL_PREVIOUS_APPLICATIONS"]
    )

    # 6. Fill missing values after aggregation
    prev_agg = prev_agg.fillna(0)

    # 7. Round selected features for readability
    prev_agg["AVG_PREVIOUS_APPLICATION_AMOUNT"] = (
        prev_agg["AVG_PREVIOUS_APPLICATION_AMOUNT"].round(2)
    )

    prev_agg["AVG_PREVIOUS_CREDIT_AMOUNT"] = (
        prev_agg["AVG_PREVIOUS_CREDIT_AMOUNT"].round(2)
    )

    prev_agg["MAX_PREVIOUS_CREDIT_AMOUNT"] = (
        prev_agg["MAX_PREVIOUS_CREDIT_AMOUNT"].round(2)
    )

    prev_agg["MOST_RECENT_PREVIOUS_APPLICATION_YEARS"] = (
        prev_agg["MOST_RECENT_PREVIOUS_APPLICATION_YEARS"].round(1)
    )

    prev_agg["PREVIOUS_APPROVAL_RATE"] = (
        prev_agg["PREVIOUS_APPROVAL_RATE"].round(3)
    )

    prev_agg["PREVIOUS_REFUSAL_RATE"] = (
        prev_agg["PREVIOUS_REFUSAL_RATE"].round(3)
    )

    # 8. Reorder columns for readability
    final_cols = [
        "SK_ID_CURR",
        "TOTAL_PREVIOUS_APPLICATIONS",
        "APPROVED_PREVIOUS_APPLICATIONS",
        "REFUSED_PREVIOUS_APPLICATIONS",
        "PREVIOUS_APPROVAL_RATE",
        "PREVIOUS_REFUSAL_RATE",
        "AVG_PREVIOUS_APPLICATION_AMOUNT",
        "AVG_PREVIOUS_CREDIT_AMOUNT",
        "MAX_PREVIOUS_CREDIT_AMOUNT",
        "MOST_RECENT_PREVIOUS_APPLICATION_YEARS",
    ]

    prev_agg = prev_agg[final_cols]

    return prev_agg


def main():
    prev_path = RAW_DIR / "previous_application.csv"

    previous_applications = pd.read_csv(prev_path)

    previous_applications_processed = preprocess_previous_applications(
        previous_applications
    )

    output_path = PROCESSED_DIR / "previous_applications_processed.csv"
    previous_applications_processed.to_csv(output_path, index=False)

    print("Previous applications preprocessing complete.")
    print(f"Raw previous applications shape: {previous_applications.shape}")
    print(
        "Processed previous applications shape:",
        previous_applications_processed.shape
    )
    print(f"Saved to: {output_path}")
    print("\nProcessed previous applications columns:")
    print(previous_applications_processed.columns.tolist())


if __name__ == "__main__":
    main()