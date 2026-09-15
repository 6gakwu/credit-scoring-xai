"""
Builds a single figure comparing how all 4 models explain the same
applicant, using LIME. Reads the existing lime_*.csv files in
explainability/, no new explanations generated here.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

EXPLAIN_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability")
OUTPUT_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\outputs\figures")

MODELS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
MODEL_TITLES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

APPLICANT_ID = None  # leave as None to just use the first applicant in the sample
N_FEATURES_SHOWN = 10


def main():
    sample_ids = pd.read_csv(EXPLAIN_DIR / "explanation_sample_applicants.csv")["SK_ID_CURR"]
    applicant_id = APPLICANT_ID if APPLICANT_ID is not None else sample_ids.iloc[0]
    print(f"Building comparison plot for applicant {applicant_id}...")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for ax, model in zip(axes, MODELS):
        df = pd.read_csv(EXPLAIN_DIR / f"lime_{model}.csv").set_index("SK_ID_CURR")
        feature_cols = [c for c in df.columns]
        row = df.loc[applicant_id, feature_cols]

        top = row.reindex(row.abs().sort_values(ascending=False).index).head(N_FEATURES_SHOWN)

        colors = ["#E45756" if v > 0 else "#4C78A8" for v in top.values]
        ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(MODEL_TITLES[model], fontsize=12, fontweight="bold")
        ax.set_xlabel("Contribution to predicted risk")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"LIME Explanation for Applicant {applicant_id}, Across All 4 Models\n"
        "(red = pushes risk up, blue = pushes risk down)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"lime_comparison_applicant_{applicant_id}.png"
    plt.savefig(output_path, dpi=170, bbox_inches="tight")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()