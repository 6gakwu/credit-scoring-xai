"""
Generates a SHAP summary plot for each of the 4 models, using the
explanations already saved to explainability/shap_*.csv. No new
explanations are generated here, this just visualises what already
exists.

A summary plot shows every one of the 200 sampled applicants as a dot
for each feature, positioned by how much that feature pushed their risk
up or down, and coloured by whether that applicant's actual value for
that feature was high or low. This makes it possible to see both which
features matter most, and the direction each one tends to push risk.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

DATA_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\data\final datasets")
EXPLAIN_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability")
OUTPUT_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\outputs\figures")

RANDOM_SEED = 42
VALIDATION_SIZE = 0.2

MODELS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
MODEL_TITLES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def get_split_and_sample():
    train = pd.read_csv(DATA_DIR / "train_selected.csv")
    feature_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    _, val_split = train_test_split(
        train, test_size=VALIDATION_SIZE, stratify=train["TARGET"], random_state=RANDOM_SEED
    )
    sample_ids = pd.read_csv(EXPLAIN_DIR / "explanation_sample_applicants.csv")["SK_ID_CURR"]
    val_indexed = val_split.set_index("SK_ID_CURR")
    X_sample = val_indexed.loc[sample_ids, feature_cols]
    return X_sample, feature_cols


def main():
    X_sample, feature_cols = get_split_and_sample()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        print(f"Building SHAP summary plot for {model}...")

        shap_df = pd.read_csv(EXPLAIN_DIR / f"shap_{model}.csv")
        shap_df = shap_df.set_index("SK_ID_CURR").loc[X_sample.index]
        shap_values = shap_df[feature_cols].values

        plt.figure()
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=feature_cols,
            show=False,
            max_display=15,
        )
        plt.title(f"SHAP Summary - {MODEL_TITLES[model]}", fontsize=12, fontweight="bold")
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"shap_summary_{model}.png"
        plt.savefig(output_path, dpi=170, bbox_inches="tight")
        plt.close()
        print(f"  Saved to {output_path}")

    print("\nAll 4 SHAP summary plots generated.")


if __name__ == "__main__":
    main()