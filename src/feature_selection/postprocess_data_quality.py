import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression


PROCESSED_DIR = Path("data/processed")

# predictors for the EXT_SOURCE imputation models - all fully populated by
# this point in the pipeline
EXT_SOURCE_PREDICTORS = [
    "AGE_YEARS",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "EMPLOYMENT_YEARS",
]

# these get collapsed into a single smoothed-target-encoded column each
TARGET_ENCODE_COLS = [
    "ORGANIZATION_TYPE",
    "OCCUPATION_TYPE",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
]

# straightforward two-category columns, just map to 0/1
BINARY_MAP_COLS = {
    "NAME_CONTRACT_TYPE": {"Cash loans": 0, "Revolving loans": 1},
    "FLAG_OWN_CAR": {"N": 0, "Y": 1},
    "FLAG_OWN_REALTY": {"N": 0, "Y": 1},
}

# education has a real order to it, so this becomes one ordinal column
# rather than a pile of one-hot columns
EDUCATION_ORDER = {
    "Lower secondary": 0,
    "Secondary / secondary special": 1,
    "Incomplete higher": 2,
    "Higher education": 3,
    "Academic degree": 4,
}

# checked this one - default rate only moves about half a point across all
# 7 days, not worth keeping
COLS_TO_DROP = ["WEEKDAY_APPR_PROCESS_START"]

# these 6 go missing together, always for applicants with zero prior
# bureau loans - so it's a true zero (no history to have enquiries
# against), not something we need to guess at
BUREAU_ENQUIRY_COLS = [
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]

# also missing together, but nothing else in the data explains why - so
# this one's a genuine unknown, gets a median fill plus a shared flag
SOCIAL_CIRCLE_COLS = [
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
]

# tiny amounts of missingness (a handful of rows each), median fill is
# plenty here, not worth a flag column
MINOR_MISSING_COLS = ["AMT_ANNUITY", "CNT_FAM_MEMBERS", "YEARS_SINCE_PHONE_CHANGE", "ANNUITY_TO_INCOME_RATIO"]

# how much weight to give the overall average when smoothing a category's
# rate - a category this size gets pulled about 50/50 toward the average.
# tested against 20 and 100, barely moved validation AUC, so this isn't a
# sensitive choice
SMOOTHING_WEIGHT = 50

# the 117,000,000 income row is >6x the next highest value and breaks the
# otherwise smooth income distribution - almost certainly a typo, not a
# real income
INCOME_OUTLIER_THRESHOLD = 50_000_000


def impute_ext_sources(train: pd.DataFrame, test: pd.DataFrame):
    """
    Fills in EXT_SOURCE_1/2/3 where they're genuinely missing (these come
    back as real NaN now that the merge step isn't zero-filling everything).
    Everything gets fit on the training set only and then applied to both
    train and test, so test never leaks into the fitting process.

    Order matters here - EXT_SOURCE_2 has hardly any gaps so it goes first
    (median fill), then that gets used to help predict EXT_SOURCE_3, and
    finally both of those help predict EXT_SOURCE_1, which is missing the
    most and is the hardest of the three.
    """
    train = train.copy()
    test = test.copy()

    # EXT_SOURCE_2 - barely any missing, median fill is enough
    train["EXT_SOURCE_2_MISSING"] = train["EXT_SOURCE_2"].isna().astype(int)
    test["EXT_SOURCE_2_MISSING"] = test["EXT_SOURCE_2"].isna().astype(int)

    ext2_median = train["EXT_SOURCE_2"].median()  # train only
    train["EXT_SOURCE_2"] = train["EXT_SOURCE_2"].fillna(ext2_median)
    test["EXT_SOURCE_2"] = test["EXT_SOURCE_2"].fillna(ext2_median)

    # EXT_SOURCE_3 next, using regression
    train, test = _predictive_impute(
        train, test, target_col="EXT_SOURCE_3",
        predictor_cols=EXT_SOURCE_PREDICTORS + ["EXT_SOURCE_2"],
    )

    # then EXT_SOURCE_1, which can now lean on the completed EXT_SOURCE_3 too
    train, test = _predictive_impute(
        train, test, target_col="EXT_SOURCE_1",
        predictor_cols=EXT_SOURCE_PREDICTORS + ["EXT_SOURCE_2", "EXT_SOURCE_3"],
    )

    return train, test


def _predictive_impute(train, test, target_col, predictor_cols):
    """Trains a plain linear regression on the rows where target_col is
    known, then uses it to fill the gaps in both train and test. Also
    drops a *_MISSING flag column first so we don't lose track of which
    rows were originally blank."""

    train[f"{target_col}_MISSING"] = train[target_col].isna().astype(int)
    test[f"{target_col}_MISSING"] = test[target_col].isna().astype(int)

    known_mask = train[target_col].notna()
    model = LinearRegression()
    model.fit(train.loc[known_mask, predictor_cols], train.loc[known_mask, target_col])

    train_missing_mask = train[target_col].isna()
    test_missing_mask = test[target_col].isna()

    train.loc[train_missing_mask, target_col] = model.predict(
        train.loc[train_missing_mask, predictor_cols]
    )
    test.loc[test_missing_mask, target_col] = model.predict(
        test.loc[test_missing_mask, predictor_cols]
    )

    return train, test


def drop_income_outliers(train: pd.DataFrame) -> pd.DataFrame:
    """Drops the one absurd income value from training data. Test doesn't
    have anything like it so nothing to do there."""
    before = len(train)
    train = train[train["AMT_INCOME_TOTAL"] < INCOME_OUTLIER_THRESHOLD].copy()
    dropped = before - len(train)
    print(f"Dropped {dropped} income outlier row(s) from training data.")
    return train


def add_credit_goods_ratio(train: pd.DataFrame, test: pd.DataFrame):
    """AMT_CREDIT and AMT_GOODS_PRICE were sitting at a 0.987 correlation -
    basically saying the same thing twice. Swapping AMT_GOODS_PRICE out for
    a ratio keeps the useful part of the relationship without the
    duplication."""
    train = train.copy()
    test = test.copy()

    train["CREDIT_GOODS_RATIO"] = train["AMT_CREDIT"] / train["AMT_GOODS_PRICE"]
    test["CREDIT_GOODS_RATIO"] = test["AMT_CREDIT"] / test["AMT_GOODS_PRICE"]

    fallback_ratio = train["CREDIT_GOODS_RATIO"].median()  # train only
    train["CREDIT_GOODS_RATIO"] = train["CREDIT_GOODS_RATIO"].fillna(fallback_ratio)
    test["CREDIT_GOODS_RATIO"] = test["CREDIT_GOODS_RATIO"].fillna(fallback_ratio)

    train = train.drop(columns=["AMT_GOODS_PRICE"])
    test = test.drop(columns=["AMT_GOODS_PRICE"])

    return train, test


def smoothed_target_encode(train: pd.DataFrame, test: pd.DataFrame, col: str, weight: int = SMOOTHING_WEIGHT):
    """Swaps a category for its average TARGET rate, but pulls small
    categories back toward the overall average so a category with just a
    handful of rows can't produce some wild, overconfident number. Rates
    come from training data only; anything in test that wasn't seen in
    training just gets the overall average."""
    overall_rate = train["TARGET"].mean()

    stats = train.groupby(col)["TARGET"].agg(["mean", "count"])
    stats["smoothed"] = (
        stats["count"] * stats["mean"] + weight * overall_rate
    ) / (stats["count"] + weight)

    encoding_map = stats["smoothed"].to_dict()

    train[col] = train[col].map(encoding_map)
    test[col] = test[col].map(encoding_map).fillna(overall_rate)

    return train, test


def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame):
    # NAME_TYPE_SUITE has some genuine blanks - label them instead of
    # guessing, then it just becomes another category to encode
    train["NAME_TYPE_SUITE"] = train["NAME_TYPE_SUITE"].fillna("Unknown")
    test["NAME_TYPE_SUITE"] = test["NAME_TYPE_SUITE"].fillna("Unknown")

    for col in TARGET_ENCODE_COLS:
        train, test = smoothed_target_encode(train, test, col)
    return train, test


def encode_binary_columns(train: pd.DataFrame, test: pd.DataFrame):
    """Two-category columns just become 0/1 - a second one-hot column
    would only repeat the same information."""
    for col, mapping in BINARY_MAP_COLS.items():
        train[col] = train[col].map(mapping)
        test[col] = test[col].map(mapping)
    return train, test


def encode_education_ordinal(train: pd.DataFrame, test: pd.DataFrame):
    """One column, 0 through 4, following the natural order of education
    level rather than treating the categories as unrelated."""
    train["NAME_EDUCATION_TYPE"] = train["NAME_EDUCATION_TYPE"].map(EDUCATION_ORDER)
    test["NAME_EDUCATION_TYPE"] = test["NAME_EDUCATION_TYPE"].map(EDUCATION_ORDER)
    return train, test


def encode_gender_onehot(train: pd.DataFrame, test: pd.DataFrame):
    """Only 3 categories here (and XNA is just 4 rows), so plain one-hot is
    fine. Categories are locked in from training data so train and test end
    up with the same columns even though test has zero XNA rows."""
    categories = sorted(train["CODE_GENDER"].unique())  # train only

    for category in categories:
        col_name = f"CODE_GENDER_{category}"
        train[col_name] = (train["CODE_GENDER"] == category).astype(int)
        test[col_name] = (test["CODE_GENDER"] == category).astype(int)

    train = train.drop(columns=["CODE_GENDER"])
    test = test.drop(columns=["CODE_GENDER"])
    return train, test


def drop_low_signal_columns(train: pd.DataFrame, test: pd.DataFrame):
    train = train.drop(columns=COLS_TO_DROP)
    test = test.drop(columns=COLS_TO_DROP)
    return train, test


def fix_remaining_missing_values(train: pd.DataFrame, test: pd.DataFrame):
    """A couple more missing-value patterns that only showed up once the
    merge script was fixed and stopped zero-filling everything - bureau
    enquiry counts, social circle counts, and a few odds and ends with
    barely any gaps at all."""
    train = train.copy()
    test = test.copy()

    # bureau enquiries - real zero, no flag needed since TOTAL_PREVIOUS_LOANS
    # already tells us these applicants have no bureau history
    train[BUREAU_ENQUIRY_COLS] = train[BUREAU_ENQUIRY_COLS].fillna(0)
    test[BUREAU_ENQUIRY_COLS] = test[BUREAU_ENQUIRY_COLS].fillna(0)

    # social circle counts - genuinely unknown, median fill plus one shared
    # flag (all four go missing together so one flag covers it)
    train["SOCIAL_CIRCLE_MISSING"] = train["OBS_30_CNT_SOCIAL_CIRCLE"].isna().astype(int)
    test["SOCIAL_CIRCLE_MISSING"] = test["OBS_30_CNT_SOCIAL_CIRCLE"].isna().astype(int)
    for col in SOCIAL_CIRCLE_COLS:
        median_val = train[col].median()  # train only
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)

    # everything else with just a few missing rows - median fill, done
    for col in MINOR_MISSING_COLS:
        median_val = train[col].median()  # train only
        train[col] = train[col].fillna(median_val)
        test[col] = test[col].fillna(median_val)

    return train, test


def reorder_columns(train: pd.DataFrame, test: pd.DataFrame):
    """Moves each *_MISSING flag next to the column it belongs to instead
    of wherever it happened to land during processing - just makes the
    table easier to skim."""

    def reorder(cols):
        cols = list(cols)
        for flag in ["EXT_SOURCE_1_MISSING", "EXT_SOURCE_2_MISSING", "EXT_SOURCE_3_MISSING", "SOCIAL_CIRCLE_MISSING"]:
            if flag in cols:
                cols.remove(flag)

        def insert_after(cols, anchor, new_col):
            idx = cols.index(anchor)
            cols.insert(idx + 1, new_col)
            return cols

        cols = insert_after(cols, "EXT_SOURCE_1", "EXT_SOURCE_1_MISSING")
        cols = insert_after(cols, "EXT_SOURCE_2", "EXT_SOURCE_2_MISSING")
        cols = insert_after(cols, "EXT_SOURCE_3", "EXT_SOURCE_3_MISSING")
        cols = insert_after(cols, "DEF_60_CNT_SOCIAL_CIRCLE", "SOCIAL_CIRCLE_MISSING")
        return cols

    train = train[reorder(train.columns)]
    test = test[reorder(test.columns)]
    return train, test


def main():
    train = pd.read_csv(PROCESSED_DIR / "train_merged.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_merged.csv")

    train, test = impute_ext_sources(train, test)
    train = drop_income_outliers(train)
    train, test = add_credit_goods_ratio(train, test)
    train, test = fix_remaining_missing_values(train, test)
    train, test = encode_categoricals(train, test)
    train, test = encode_binary_columns(train, test)
    train, test = encode_education_ordinal(train, test)
    train, test = encode_gender_onehot(train, test)
    train, test = drop_low_signal_columns(train, test)
    train, test = reorder_columns(train, test)

    train.to_csv(PROCESSED_DIR / "train_merged.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test_merged.csv", index=False)

    print("Post-processing complete.")
    print(f"Final train shape: {train.shape}")
    print(f"Final test shape: {test.shape}")


if __name__ == "__main__":
    main()
