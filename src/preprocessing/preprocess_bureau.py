import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates bureau.csv into one row per applicant using a compact set
    of direct, interpretable credit-history features.
    """

    bureau = bureau.copy()

    # Drop low-value categorical column
    if "CREDIT_CURRENCY" in bureau.columns:
        bureau = bureau.drop(columns=["CREDIT_CURRENCY"])

    # Convert DAYS_CREDIT into years since previous credit was taken
    bureau["YEARS_SINCE_PREVIOUS_CREDIT"] = (-bureau["DAYS_CREDIT"]) / 365

    # Indicator for active previous credits
    bureau["IS_ACTIVE_CREDIT"] = (
        bureau["CREDIT_ACTIVE"] == "Active"
    ).astype(int)

    # Treat negative debt values as 0 because outstanding debt should not be negative
    bureau["AMT_CREDIT_SUM_DEBT"] = bureau["AMT_CREDIT_SUM_DEBT"].clip(lower=0)

    # Aggregate bureau records into one row per applicant
    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        TOTAL_PREVIOUS_LOANS=("SK_ID_BUREAU", "count"),
        ACTIVE_PREVIOUS_LOANS=("IS_ACTIVE_CREDIT", "sum"),
        TOTAL_BUREAU_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
        TOTAL_OVERDUE_DEBT=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        MAX_DAYS_OVERDUE=("CREDIT_DAY_OVERDUE", "max"),
        MOST_RECENT_PREVIOUS_CREDIT_YEARS=("YEARS_SINCE_PREVIOUS_CREDIT", "min")
    ).reset_index()

    # Create selected credit type count features
    credit_type_counts = pd.crosstab(
        bureau["SK_ID_CURR"],
        bureau["CREDIT_TYPE"]
    )

    credit_type_counts["COUNT_CREDIT_CARDS"] = (
        credit_type_counts["Credit card"]
        if "Credit card" in credit_type_counts.columns
        else 0
    )

    credit_type_counts["COUNT_CONSUMER_CREDITS"] = (
        credit_type_counts["Consumer credit"]
        if "Consumer credit" in credit_type_counts.columns
        else 0
    )

    credit_type_counts = credit_type_counts[
        [
            "COUNT_CREDIT_CARDS",
            "COUNT_CONSUMER_CREDITS"
        ]
    ].reset_index()

    # Merge credit type counts into bureau aggregation
    bureau_agg = bureau_agg.merge(
        credit_type_counts,
        on="SK_ID_CURR",
        how="left"
    )

    # Fill missing aggregate values with 0
    bureau_agg = bureau_agg.fillna(0)

    # Round year-based feature for readability
    bureau_agg["MOST_RECENT_PREVIOUS_CREDIT_YEARS"] = (
        bureau_agg["MOST_RECENT_PREVIOUS_CREDIT_YEARS"].round(1)
    )

    return bureau_agg


def main():
    bureau_path = RAW_DIR / "bureau.csv"

    bureau = pd.read_csv(bureau_path)

    bureau_processed = preprocess_bureau(bureau)

    output_path = PROCESSED_DIR / "bureau_processed.csv"
    bureau_processed.to_csv(output_path, index=False)

    print("Bureau preprocessing complete.")
    print(f"Raw bureau shape: {bureau.shape}")
    print(f"Processed bureau shape: {bureau_processed.shape}")
    print(f"Saved to: {output_path}")
    print("\nProcessed bureau columns:")
    print(bureau_processed.columns.tolist())


if __name__ == "__main__":
    main()