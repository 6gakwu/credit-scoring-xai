"""
RQ2, part 2 - Logistic Regression, SHAP similarity stability.

Pulls each anchor's EXISTING SHAP explanation from shap_logistic_regression.csv
and generates FRESH SHAP explanations for
the 150 neighbor applicants, since they're new people outside our
original 200. Same probability-based approach as every other SHAP
script in this project.

Then computes two things per anchor group (the anchor + its 5 neighbors):
- Variance in feature importance: how much each feature's actual signed
  contribution wobbles across the 6 people in the group, averaged across
  all 71 features. Lower = more stable.
- Rank consistency: Spearman rank correlation between the anchor's
  feature-importance ranking (by absolute size) and each neighbor's
  ranking, averaged across the 5 neighbors. Closer to 1 = more
  consistent about what matters most.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import joblib
import shap

DATA_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\data\final datasets")
MODEL_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\models\saved_models")
EXPLAIN_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability")
SIMILARITY_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability\similarity_analysis")

RANDOM_SEED = 42
VALIDATION_SIZE = 0.2


def get_split():
    train = pd.read_csv(DATA_DIR / "train_selected.csv")
    feature_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    train_split, val_split = train_test_split(
        train, test_size=VALIDATION_SIZE, stratify=train["TARGET"], random_state=RANDOM_SEED
    )
    return train_split, val_split, feature_cols


def main():
    train_split, val_split, feature_cols = get_split()
    groups = pd.read_csv(SIMILARITY_DIR / "similar_applicant_groups.csv")
    anchor_explanations = pd.read_csv(EXPLAIN_DIR / "shap_logistic_regression.csv").set_index("SK_ID_CURR")

    neighbor_ids = groups["neighbor_id"].unique()
    val_indexed = val_split.set_index("SK_ID_CURR")
    X_neighbors = val_indexed.loc[neighbor_ids, feature_cols]
    X_train = train_split[feature_cols]

    model = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
    scaler = joblib.load(MODEL_DIR / "logistic_regression_scaler.pkl")

    X_train_scaled = scaler.transform(X_train)
    X_neighbors_scaled = scaler.transform(X_neighbors)

    background = shap.sample(X_train_scaled, 100, random_state=RANDOM_SEED)

    print("Building SHAP explainer...")
    predict_fn = lambda x: model.predict_proba(x)[:, 1]
    explainer = shap.Explainer(predict_fn, background, feature_names=feature_cols)

    print(f"Generating SHAP explanations for {len(neighbor_ids)} neighbor applicants...")
    neighbor_shap_values = explainer(X_neighbors_scaled)

    print(f"Neighbor SHAP values shape: {neighbor_shap_values.values.shape} (expecting ({len(neighbor_ids)}, {len(feature_cols)}))")

    neighbor_explanations = pd.DataFrame(neighbor_shap_values.values, columns=feature_cols, index=neighbor_ids)
    neighbor_explanations.to_csv(SIMILARITY_DIR / "shap_logistic_regression_neighbors.csv")
    print(f"Saved neighbor explanations to {SIMILARITY_DIR / 'shap_logistic_regression_neighbors.csv'}")

    print("\nComputing variance and rank consistency per anchor group...")
    results = []
    for anchor_id in groups["anchor_id"].unique():
        anchor_vector = anchor_explanations.loc[anchor_id, feature_cols].values.astype(float)
        group_neighbor_ids = groups[groups["anchor_id"] == anchor_id]["neighbor_id"].values
        neighbor_vectors = neighbor_explanations.loc[group_neighbor_ids, feature_cols].values.astype(float)

        all_vectors = np.vstack([anchor_vector.reshape(1, -1), neighbor_vectors])
        avg_variance = all_vectors.var(axis=0, ddof=1).mean()

        anchor_rank = pd.Series(np.abs(anchor_vector)).rank()
        rank_corrs = [spearmanr(anchor_rank, pd.Series(np.abs(nv)).rank())[0] for nv in neighbor_vectors]
        avg_rank_corr = np.mean(rank_corrs)

        results.append({"anchor_id": anchor_id, "avg_feature_variance": avg_variance, "avg_rank_correlation": avg_rank_corr})

    results_df = pd.DataFrame(results)
    scores_path = SIMILARITY_DIR / "stability_scores_shap_logistic_regression.csv"
    results_df.to_csv(scores_path, index=False)
    print(f"Saved per-anchor stability scores to {scores_path}")

    print(f"\nLogistic Regression - SHAP similarity stability")
    print(f"  Avg feature variance: {results_df['avg_feature_variance'].mean():.6f}")
    print(f"  Avg rank correlation: {results_df['avg_rank_correlation'].mean():.4f}")


if __name__ == "__main__":
    main()