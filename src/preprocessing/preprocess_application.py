import pandas as pd
import numpy as np
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_application(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()

    # DAYS_EMPLOYED anomaly (365243) - mostly pensioners/unemployed people.
    # grab a mask before we wipe it to NaN so we can go back and set these
    # rows properly once EMPLOYMENT_YEARS exists
    if "DAYS_EMPLOYED" in df.columns:
        employment_anomaly_mask = df["DAYS_EMPLOYED"] == 365243
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # turn the day-offset columns into something readable (years instead of
    # negative day counts)
    date_conversions = {
        "DAYS_BIRTH": "AGE_YEARS",
        "DAYS_EMPLOYED": "EMPLOYMENT_YEARS",
        "DAYS_REGISTRATION": "REGISTRATION_YEARS",
        "DAYS_ID_PUBLISH": "YEARS_SINCE_ID_UPDATE",
        "DAYS_LAST_PHONE_CHANGE": "YEARS_SINCE_PHONE_CHANGE",
    }

    for old_col, new_col in date_conversions.items():
        if old_col in df.columns:
            df[new_col] = (-df[old_col]) / 365

    # for the anomaly rows, EMPLOYMENT_YEARS isn't really "missing" - it just
    # doesn't apply, since they're not currently employed. 0 is the honest
    # value here, not a guess, and NAME_INCOME_TYPE already flags who these
    # people are so there's no need for a separate missing-indicator column
    if "EMPLOYMENT_YEARS" in df.columns:
        df.loc[employment_anomaly_mask, "EMPLOYMENT_YEARS"] = 0

    # a couple of simple ratio features - affordability relative to income
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]

    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_TO_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    # just in case dividing by a zero income slipped through
    df = df.replace([np.inf, -np.inf], np.nan)

    # don't need the raw day-offset columns anymore now that we have the
    # year versions
    drop_cols = [
        "DAYS_BIRTH",
        "DAYS_EMPLOYED",
        "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH",
        "DAYS_LAST_PHONE_CHANGE",
    ]

    # these contact flags are basically constant or redundant, not worth keeping
    drop_cols.extend([
        "FLAG_MOBIL",
        "FLAG_CONT_MOBILE",
        "FLAG_WORK_PHONE",
        "REG_REGION_NOT_LIVE_REGION",
        "REG_CITY_NOT_LIVE_CITY",
    ])

    # document flag columns are very sparse and don't add much
    document_cols = [
        col for col in df.columns
        if col.startswith("FLAG_DOCUMENT_")
    ]
    drop_cols.extend(document_cols)

    # building/housing metadata - lots of missingness, low interpretability,
    # cutting these to keep the feature set manageable
    building_keywords = [
        "APARTMENTS",
        "BASEMENTAREA",
        "YEARS_BEGINEXPLUATATION",
        "YEARS_BUILD",
        "COMMONAREA",
        "ELEVATORS",
        "ENTRANCES",
        "FLOORSMAX",
        "FLOORSMIN",
        "LANDAREA",
        "LIVINGAPARTMENTS",
        "LIVINGAREA",
        "NONLIVINGAPARTMENTS",
        "NONLIVINGAREA",
        "FONDKAPREMONT",
        "HOUSETYPE",
        "TOTALAREA",
        "WALLSMATERIAL",
        "EMERGENCYSTATE",
    ]

    building_cols = [
        col for col in df.columns
        if any(keyword in col for keyword in building_keywords)
    ]
    drop_cols.extend(building_cols)

    # only drop what's actually there (test set won't have TARGET etc.)
    drop_cols = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=drop_cols)

    # sanity check - train should always have TARGET, fail loudly if not
    if is_train and "TARGET" not in df.columns:
        raise ValueError("TARGET column is missing from processed training data.")

    # round for readability, no real precision lost here
    year_cols = [
        "AGE_YEARS",
        "EMPLOYMENT_YEARS",
        "REGISTRATION_YEARS",
        "YEARS_SINCE_ID_UPDATE",
        "YEARS_SINCE_PHONE_CHANGE"
]

    existing_year_cols = [col for col in year_cols if col in df.columns]
    df[existing_year_cols] = df[existing_year_cols].round(1)

    ratio_cols = [
        "CREDIT_TO_INCOME_RATIO",
        "ANNUITY_TO_INCOME_RATIO"
    ]

    existing_ratio_cols = [col for col in ratio_cols if col in df.columns]
    df[existing_ratio_cols] = df[existing_ratio_cols].round(3)

    # OCCUPATION_TYPE has genuine blanks (nobody entered a job) - label them
    # rather than leaving NaN. ORGANIZATION_TYPE already uses "XNA" for the
    # same idea so this mostly just covers OCCUPATION_TYPE in practice
    categorical_fill_cols = [
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE"
    ]

    for col in categorical_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # belt and braces
    df = df.drop_duplicates()

    return df


def main():
    train_path = RAW_DIR / "application_train.csv"
    test_path = RAW_DIR / "application_test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_processed = preprocess_application(train, is_train=True)
    test_processed = preprocess_application(test, is_train=False)

    train_processed.to_csv(
        PROCESSED_DIR / "application_train_processed.csv",
        index=False
    )

    test_processed.to_csv(
        PROCESSED_DIR / "application_test_processed.csv",
        index=False
    )

    print("Application preprocessing complete.")
    print(f"Train processed shape: {train_processed.shape}")
    print(f"Test processed shape: {test_processed.shape}")
    print(f"Train saved to: {PROCESSED_DIR / 'application_train_processed.csv'}")
    print(f"Test saved to: {PROCESSED_DIR / 'application_test_processed.csv'}")


if __name__ == "__main__":
    main()