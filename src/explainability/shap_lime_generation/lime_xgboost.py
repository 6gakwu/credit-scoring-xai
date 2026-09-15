"""
LIME explanations for XGBoost.
"""

import pandas as pd
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

    model = joblib.load(MODEL_DIR / "xgboost_model.pkl")

    print("Building LIME explainer...")
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_cols,
        class_names=["No Default", "Default"],
        mode="classification",
        discretize_continuous=False,
        random_state=RANDOM_SEED,
    )

    predict_fn = model.predict_proba

    print(f"Generating LIME explanations for {len(X_sample)} applicants...")

    all_rows = []
    X_sample_values = X_sample.values
    for i, (sk_id, row) in enumerate(zip(sample_ids.values, X_sample_values)):
        explanation = explainer.explain_instance(row, predict_fn, num_features=len(feature_cols))

        weights_by_index = dict(explanation.as_map()[1])
        row_weights = [weights_by_index.get(idx, 0.0) for idx in range(len(feature_cols))]
        all_rows.append(row_weights)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1} / {len(sample_ids)} done")

    lime_df = pd.DataFrame(all_rows, columns=feature_cols)
    lime_df.insert(0, "SK_ID_CURR", sample_ids.values)

    output_path = EXPLAIN_DIR / "lime_xgboost.csv"
    lime_df.to_csv(output_path, index=False)
    print(f"\nSaved LIME explanations to {output_path}")

    print(f"LIME values shape: {lime_df.shape} (expecting (200, {len(feature_cols) + 1}))")

    avg_importance = lime_df[feature_cols].abs().mean().sort_values(ascending=False)
    print("\nTop 10 features by average LIME contribution (in probability points):")
    print(avg_importance.head(10).to_string())

    print(f"\nLargest single contribution seen: {lime_df[feature_cols].abs().values.max():.4f}")


if __name__ == "__main__":
    main()