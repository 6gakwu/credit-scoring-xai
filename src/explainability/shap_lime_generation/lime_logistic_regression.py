"""
Step 8 - LIME explanations for Logistic Regression.

Unlike SHAP, LIME explains one applicant at a time - for each person, it
builds hundreds of slightly randomized fake versions of them, asks the
real model what it thinks of each one, then fits a tiny local model just
to explain that one person. This makes it noticeably slower than SHAP,
since it has to repeat that whole process 200 separate times.

discretize_continuous=False keeps every feature as one clean number and
weight, rather than LIME's default of grouping continuous features into
ranges - we need this to match the SHAP output format for later stability calculations.


No probability-wrapper formatting needed here unlike the SHAP scripts 
LIME is always built around the model's actual probability output
so every model's LIME explanation is already in comparable units.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib
from lime.lime_tabular import LimeTabularExplainer

DATA_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\data\final datasets")
MODEL_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\models\saved_models")
EXPLAIN_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability")

RANDOM_SEED = 42
VALIDATION_SIZE = 0.2


def get_validation_split():
    train = pd.read_csv(DATA_DIR / "train_selected.csv")
    feature_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    train_split, val_split = train_test_split(
        train, test_size=VALIDATION_SIZE, stratify=train["TARGET"], random_state=RANDOM_SEED
    )
    return train_split, val_split, feature_cols


def get_explanation_sample():
    sample_path = EXPLAIN_DIR / "explanation_sample_applicants.csv"
    sample_ids = pd.read_csv(sample_path)["SK_ID_CURR"]
    print(f"Reusing existing sample of {len(sample_ids)} applicants.")
    return sample_ids


def main():
    train_split, val_split, feature_cols = get_validation_split()
    sample_ids = get_explanation_sample()

    X_sample = val_split.set_index("SK_ID_CURR").loc[sample_ids, feature_cols]
    X_train = train_split[feature_cols]

    model = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
    scaler = joblib.load(MODEL_DIR / "logistic_regression_scaler.pkl")

    X_train_scaled = scaler.transform(X_train)
    X_sample_scaled = scaler.transform(X_sample)

    print("Building LIME explainer...")
    explainer = LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_cols,
        class_names=["No Default", "Default"],
        mode="classification",
        discretize_continuous=False,
        random_state=RANDOM_SEED,
    )

    # LIME wants a function that takes applicant rows and returns
    # probabilities for both classes - predict_proba already does exactly
    # this, no wrapper trick needed this time
    predict_fn = model.predict_proba

    print(f"Generating LIME explanations for {len(X_sample)} applicants...")
    print("(this will take noticeably longer than SHAP did - LIME explains one applicant at a time)")

    all_rows = []
    for i, (sk_id, row) in enumerate(zip(sample_ids.values, X_sample_scaled)):
        explanation = explainer.explain_instance(row, predict_fn, num_features=len(feature_cols))

        # as_map() gives weights keyed by feature index rather than text,
        # which sidesteps any risk of string-parsing issues
        weights_by_index = dict(explanation.as_map()[1])  # label 1 = "Default"
        row_weights = [weights_by_index.get(idx, 0.0) for idx in range(len(feature_cols))]
        all_rows.append(row_weights)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1} / {len(sample_ids)} done")

    lime_df = pd.DataFrame(all_rows, columns=feature_cols)
    lime_df.insert(0, "SK_ID_CURR", sample_ids.values)

    output_path = EXPLAIN_DIR / "lime_logistic_regression.csv"
    lime_df.to_csv(output_path, index=False)
    print(f"\nSaved LIME explanations to {output_path}")

    print(f"LIME values shape: {lime_df.shape} (expecting (200, {len(feature_cols) + 1}))")

    avg_importance = lime_df[feature_cols].abs().mean().sort_values(ascending=False)
    print("\nTop 10 features by average LIME contribution (in probability points):")
    print(avg_importance.head(10).to_string())

    print(f"\nLargest single contribution seen: {lime_df[feature_cols].abs().values.max():.4f}")


if __name__ == "__main__":
    main()