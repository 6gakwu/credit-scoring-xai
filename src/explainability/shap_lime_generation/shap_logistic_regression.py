"""
SHAP explanations for Logistic Regression.

Rescaled every feature onto a comparable range before
feeding to the model, since Logistic Regression is sensitive to feature
scale and was trained on scaled data

SHAP to report contributions in probability space, so that all models
can be compared on the same scale 

Results are saved to shap_logistic_regression.csv in the explainability folder.



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
SAMPLE_SIZE = 200


def get_validation_split():
    train = pd.read_csv(DATA_DIR / "train_selected.csv")
    feature_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    train_split, val_split = train_test_split(
        train, test_size=VALIDATION_SIZE, stratify=train["TARGET"], random_state=RANDOM_SEED
    )
    return train_split, val_split, feature_cols


def get_explanation_sample(val_split):
    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = EXPLAIN_DIR / "explanation_sample_applicants.csv"
    sample_ids = pd.read_csv(sample_path)["SK_ID_CURR"]
    print(f"Reusing existing sample of {len(sample_ids)} applicants.")
    return sample_ids


def main():
    train_split, val_split, feature_cols = get_validation_split()
    sample_ids = get_explanation_sample(val_split)

    X_sample = val_split.set_index("SK_ID_CURR").loc[sample_ids, feature_cols]
    X_train = train_split[feature_cols]

    model = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
    scaler = joblib.load(MODEL_DIR / "logistic_regression_scaler.pkl")

    X_train_scaled = scaler.transform(X_train)
    X_sample_scaled = scaler.transform(X_sample)

    background = shap.sample(X_train_scaled, 100, random_state=RANDOM_SEED)

    print("Building SHAP explainer...")
    # the key change: passing model.predict_proba (the probability output)
    # instead of the model itself, and telling SHAP explicitly this is
    # probability space. This is what guarantees every model in this
    # project reports contributions in the same, comparable units.
    predict_fn = lambda x: model.predict_proba(x)[:, 1]
    explainer = shap.Explainer(predict_fn, background, feature_names=feature_cols)

    print(f"Generating SHAP explanations for {len(X_sample)} applicants...")
    shap_values = explainer(X_sample_scaled)

    print(f"\nSHAP values shape: {shap_values.values.shape} (expecting (200, {len(feature_cols)}))")

    shap_df = pd.DataFrame(shap_values.values, columns=feature_cols)
    shap_df.insert(0, "SK_ID_CURR", sample_ids.values)

    output_path = EXPLAIN_DIR / "shap_logistic_regression.csv"
    shap_df.to_csv(output_path, index=False)
    print(f"Saved SHAP explanations to {output_path}")

    avg_importance = shap_df[feature_cols].abs().mean().sort_values(ascending=False)
    print("\nTop 10 features by average SHAP contribution (in probability points):")
    print(avg_importance.head(10).to_string())

    # sanity check specific to this fix - contributions should now be
    # small, sensible percentage-point numbers (e.g. 0.01-0.15), not
    # anything wildly large, since risk itself only ranges from 0 to 1
    print(f"\nLargest single contribution seen: {shap_df[feature_cols].abs().values.max():.4f}")


if __name__ == "__main__":
    main()