import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_installments(installments: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates installments_payments.csv into one row per applicant using
    compact, interpretable repayment-behaviour features.
    """

    installments = installments.copy()

    # 1. Create raw payment delay
    # Positive = paid late
    # Zero = paid on time
    # Negative = paid early
    installments["PAYMENT_DELAY_DAYS"] = (
        installments["DAYS_ENTRY_PAYMENT"] - installments["DAYS_INSTALMENT"]
    )

    # 2. Convert delay into late-payment-only delay
    # Early and on-time payments become 0
    # Late payments keep their positive delay value
    installments["LATE_PAYMENT_DAYS"] = installments["PAYMENT_DELAY_DAYS"].clip(lower=0)

    # 3. Create late payment indicator
    installments["IS_LATE_PAYMENT"] = (
        installments["LATE_PAYMENT_DAYS"] > 0
    ).astype(int)

    # 4. Aggregate installment records into one row per applicant
    installments_agg = installments.groupby("SK_ID_CURR").agg(
        TOTAL_INSTALLMENT_RECORDS=("SK_ID_PREV", "count"),
        LATE_PAYMENT_COUNT=("IS_LATE_PAYMENT", "sum"),
        AVG_LATE_PAYMENT_DAYS=("LATE_PAYMENT_DAYS", "mean"),
        MAX_LATE_PAYMENT_DAYS=("LATE_PAYMENT_DAYS", "max")
    ).reset_index()

    # 5. Create late payment rate
    installments_agg["LATE_PAYMENT_RATE"] = (
        installments_agg["LATE_PAYMENT_COUNT"] /
        installments_agg["TOTAL_INSTALLMENT_RECORDS"]
    )

    # 6. Fill missing values after aggregation
    installments_agg = installments_agg.fillna(0)

    # 7. Round selected features for readability
    installments_agg["LATE_PAYMENT_RATE"] = (
        installments_agg["LATE_PAYMENT_RATE"].round(3)
    )

    installments_agg["AVG_LATE_PAYMENT_DAYS"] = (
        installments_agg["AVG_LATE_PAYMENT_DAYS"].round(2)
    )

    installments_agg["MAX_LATE_PAYMENT_DAYS"] = (
        installments_agg["MAX_LATE_PAYMENT_DAYS"].round(2)
    )

    # 8. Reorder columns
    final_cols = [
        "SK_ID_CURR",
        "TOTAL_INSTALLMENT_RECORDS",
        "LATE_PAYMENT_COUNT",
        "LATE_PAYMENT_RATE",
        "AVG_LATE_PAYMENT_DAYS",
        "MAX_LATE_PAYMENT_DAYS"
    ]

    installments_agg = installments_agg[final_cols]

    return installments_agg


def main():
    installments_path = RAW_DIR / "installments_payments.csv"

    installments = pd.read_csv(installments_path)

    installments_processed = preprocess_installments(installments)

    output_path = PROCESSED_DIR / "installments_processed.csv"
    installments_processed.to_csv(output_path, index=False)

    print("Installments preprocessing complete.")
    print(f"Raw installments shape: {installments.shape}")
    print(f"Processed installments shape: {installments_processed.shape}")
    print(f"Saved to: {output_path}")
    print("\nProcessed installments columns:")
    print(installments_processed.columns.tolist())


if __name__ == "__main__":
    main()