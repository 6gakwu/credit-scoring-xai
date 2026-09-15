import pandas as pd
from pathlib import Path


PROCESSED_DIR = Path("data/processed")


def merge_supporting_tables(application_df, bureau, previous_apps, installments):
    """
    Merges processed bureau, previous application, and installment features
    into the processed application dataset using SK_ID_CURR.
    """

    df = application_df.copy()

    # merge each table in one at a time, then fill 0 only for the columns
    # that table just brought in. no bureau/previous app/installment record
    # for someone genuinely means zero there, so that's fine to fill.
    # important: don't do one big fillna(0) at the end - that would also
    # wipe out real NaNs already in the application data (EXT_SOURCE_1/2/3,
    # OWN_CAR_AGE etc), which mean "unknown"/"not applicable" and get
    # handled properly later on, not just zeroed out
    for supporting_table in (bureau, previous_apps, installments):
        new_cols = [c for c in supporting_table.columns if c != "SK_ID_CURR"]
        df = df.merge(supporting_table, on="SK_ID_CURR", how="left")
        df[new_cols] = df[new_cols].fillna(0)

    return df


def main():
    train_path = PROCESSED_DIR / "application_train_processed.csv"
    test_path = PROCESSED_DIR / "application_test_processed.csv"
    bureau_path = PROCESSED_DIR / "bureau_processed.csv"
    previous_apps_path = PROCESSED_DIR / "previous_applications_processed.csv"
    installments_path = PROCESSED_DIR / "installments_processed.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    bureau = pd.read_csv(bureau_path)
    previous_apps = pd.read_csv(previous_apps_path)
    installments = pd.read_csv(installments_path)

    train_merged = merge_supporting_tables(
        train,
        bureau,
        previous_apps,
        installments
    )

    test_merged = merge_supporting_tables(
        test,
        bureau,
        previous_apps,
        installments
    )

    train_output = PROCESSED_DIR / "train_merged.csv"
    test_output = PROCESSED_DIR / "test_merged.csv"

    train_merged.to_csv(train_output, index=False)
    test_merged.to_csv(test_output, index=False)

    print("Merging complete.")
    print(f"Train merged shape: {train_merged.shape}")
    print(f"Test merged shape: {test_merged.shape}")
    print(f"Train saved to: {train_output}")
    print(f"Test saved to: {test_output}")

    print("\nTARGET in train:", "TARGET" in train_merged.columns)
    print("TARGET in test:", "TARGET" in test_merged.columns)

    train_features = set(train_merged.columns) - {"TARGET"}
    test_features = set(test_merged.columns)

    print("\nTrain/test feature alignment except TARGET:")
    print(train_features == test_features)


if __name__ == "__main__":
    main()