"""
RQ1 step 1, generating the nudged applicant data.

Takes 50 real applicants (a fixed subset of our original 200) and creates
5 slightly perturbed versions of each one - same person, but with their
genuine numeric features (income, age, EXT_SOURCE scores, etc.) nudged up
or down by a small, random amount, up to 5% each way. Flags, categories,
and small integer counts are left completely untouched, since 5% more
doesn't mean apply to those.

Two ratio columns (CREDIT_TO_INCOME_RATIO, ANNUITY_TO_INCOME_RATIO) get
recalculated afterward from their own newly-nudged raw amounts, rather
than nudged directly - otherwise we'd end up with an applicant whose
numbers contradict each other (income changed, but the ratio still
reflects the old income).

This script only creates the nudged data - it doesn't touch the models
or generate any explanations, the stability_* scripts in this folder will do that
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\data\final datasets")
EXPLAIN_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability")
OUTPUT_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability\perturbation_analysis")

RANDOM_SEED = 42
VALIDATION_SIZE = 0.2
N_APPLICANTS = 50
N_PERTURBATIONS = 5
NUDGE_RANGE = 0.05  # +/- 5%

# genuine continuous measurements get independently nudged
NUDGE_COLS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "REGION_POPULATION_RELATIVE",
    "OWN_CAR_AGE", "AGE_YEARS", "EMPLOYMENT_YEARS", "REGISTRATION_YEARS",
    "YEARS_SINCE_ID_UPDATE", "YEARS_SINCE_PHONE_CHANGE",
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "CREDIT_GOODS_RATIO",
    "TOTAL_BUREAU_DEBT", "TOTAL_OVERDUE_DEBT", "MOST_RECENT_PREVIOUS_CREDIT_YEARS",
    "AVG_PREVIOUS_APPLICATION_AMOUNT", "MAX_PREVIOUS_CREDIT_AMOUNT",
    "MOST_RECENT_PREVIOUS_APPLICATION_YEARS", "PREVIOUS_APPROVAL_RATE",
    "PREVIOUS_REFUSAL_RATE", "LATE_PAYMENT_RATE", "AVG_LATE_PAYMENT_DAYS",
    "MAX_LATE_PAYMENT_DAYS",
]

# rates and scores need to stay within their real, valid bounds after nudging
CLIP_0_1 = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "PREVIOUS_APPROVAL_RATE",
            "PREVIOUS_REFUSAL_RATE", "LATE_PAYMENT_RATE"]
CLIP_NONNEGATIVE = ["OWN_CAR_AGE", "AGE_YEARS", "EMPLOYMENT_YEARS", "REGISTRATION_YEARS",
                     "YEARS_SINCE_ID_UPDATE", "YEARS_SINCE_PHONE_CHANGE", "TOTAL_BUREAU_DEBT",
                     "TOTAL_OVERDUE_DEBT", "AVG_LATE_PAYMENT_DAYS", "MAX_LATE_PAYMENT_DAYS",
                     "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]


def make_perturbed_version(row, rng):
    row = row.copy()
    for col in NUDGE_COLS:
        original = row[col]
        if original == 0:
            continue  # nothing to nudge - 0% of anything is still 0 (e.g. no-car, pensioners)
        pct_change = rng.uniform(-NUDGE_RANGE, NUDGE_RANGE)
        row[col] = original * (1 + pct_change)

    for col in CLIP_0_1:
        row[col] = np.clip(row[col], 0, 1)
    for col in CLIP_NONNEGATIVE:
        row[col] = max(row[col], 0)

    # recompute these from the now-nudged raw amounts, so the applicant
    # stays internally consistent rather than contradicting itself
    row["CREDIT_TO_INCOME_RATIO"] = row["AMT_CREDIT"] / row["AMT_INCOME_TOTAL"]
    row["ANNUITY_TO_INCOME_RATIO"] = row["AMT_ANNUITY"] / row["AMT_INCOME_TOTAL"]

    return row


def main():
    train = pd.read_csv(DATA_DIR / "train_selected.csv")
    feature_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]

    # same split as every other script in this project .these are the
    # same validation applicants everything else was built on
    train_split, val_split = train_test_split(
        train, test_size=VALIDATION_SIZE, stratify=train["TARGET"], random_state=RANDOM_SEED
    )

    sample_ids = pd.read_csv(EXPLAIN_DIR / "explanation_sample_applicants.csv")["SK_ID_CURR"]
    perturb_ids = sample_ids.sample(n=N_APPLICANTS, random_state=RANDOM_SEED)

    val_indexed = val_split.set_index("SK_ID_CURR")
    rng = np.random.default_rng(RANDOM_SEED)

    all_perturbed_rows = []
    for sk_id in perturb_ids:
        original_row = val_indexed.loc[sk_id, feature_cols]
        for p in range(N_PERTURBATIONS):
            perturbed_row = make_perturbed_version(original_row, rng)
            record = {"SK_ID_CURR": sk_id, "perturbation_number": p + 1}
            record.update(perturbed_row.to_dict())
            all_perturbed_rows.append(record)

    perturbed_df = pd.DataFrame(all_perturbed_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "perturbed_applicants.csv"
    perturbed_df.to_csv(output_path, index=False)

    print(f"Perturbed data shape: {perturbed_df.shape} (expecting ({N_APPLICANTS * N_PERTURBATIONS}, {len(feature_cols) + 2}))")
    print(f"Saved to {output_path}")

    # sanity check on one real applicant across its 5 perturbed versions -
    # values should be close to the original but not identical, and the
    # ratio should track the nudged income/credit correctly
    example_id = perturb_ids.iloc[0]
    print(f"\nExample applicant {example_id} across its {N_PERTURBATIONS} perturbed versions:")
    cols_to_show = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "CREDIT_TO_INCOME_RATIO", "EXT_SOURCE_1", "OWN_CAR_AGE"]
    print(perturbed_df[perturbed_df["SK_ID_CURR"] == example_id][cols_to_show].to_string(index=False))
    print("\nOriginal values for comparison:")
    print(val_indexed.loc[example_id, cols_to_show])


if __name__ == "__main__":
    main()