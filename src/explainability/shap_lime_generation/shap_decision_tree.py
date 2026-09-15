"""
SHAP explanations for Decision Tree.

Same probability-based approach as Logistic Regression, reused on
purpose - every model in this project needs its SHAP contributions
measured in the same units (percentage points of predicted risk) for
accurate cross-model comparison

No scaler used 
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib
import shap

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
    # reusing the exact same 200 applicants saved during the Logistic
    # Regression run - never regenerated, so every model explains the
    # same people
    sample_path = EXPLAIN_DIR / "explanation_sample_applicants.csv"
    sample_ids = pd.read_csv(sample_path)["SK_ID_CURR"]
    print(f"Reusing existing sample of {len(sample_ids)} applicants.")
    return sample_ids


def main():
    train_split, val_split, feature_cols = get_validation_split()
    sample_ids = get_explanation_sample()

    X_sample = val_split.set_index("SK_ID_CURR").loc[sample_ids, feature_cols]
    X_train = train_split[feature_cols]

    model = joblib.load(MODEL_DIR / "decision_tree_model.pkl")

    # no scaling - straight from the source, same as how this model was trained
    background = shap.sample(X_train, 100, random_state=RANDOM_SEED)

    print("Building SHAP explainer...")
    predict_fn = lambda x: model.predict_proba(x)[:, 1]
    explainer = shap.Explainer(predict_fn, background, feature_names=feature_cols)

    print(f"Generating SHAP explanations for {len(X_sample)} applicants...")
    shap_values = explainer(X_sample)

    print(f"\nSHAP values shape: {shap_values.values.shape} (expecting (200, {len(feature_cols)}))")

    shap_df = pd.DataFrame(shap_values.values, columns=feature_cols)
    shap_df.insert(0, "SK_ID_CURR", sample_ids.values)

    output_path = EXPLAIN_DIR / "shap_decision_tree.csv"
    shap_df.to_csv(output_path, index=False)
    print(f"Saved SHAP explanations to {output_path}")

    avg_importance = shap_df[feature_cols].abs().mean().sort_values(ascending=False)
    print("\nTop 10 features by average SHAP contribution (in probability points):")
    print(avg_importance.head(10).to_string())

    print(f"\nLargest single contribution seen: {shap_df[feature_cols].abs().values.max():.4f}")


if __name__ == "__main__":
    main()