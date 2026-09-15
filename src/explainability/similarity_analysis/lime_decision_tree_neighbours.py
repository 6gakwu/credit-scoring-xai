"""
RQ2, part 2 - Decision Tree, LIME similarity stability. Same approach as
the other LIME versions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import joblib
from lime.lime_tabular import LimeTabularExplainer

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
    anchor_explanations = pd.read_csv(EXPLAIN_DIR / "lime_decision_tree.csv").set_index("SK_ID_CURR")

    neighbor_ids = groups["neighbor_id"].unique()
    val_indexed = val_split.set_index("SK_ID_CURR")
    X_neighbors = val_indexed.loc[neighbor_ids, feature_cols]
    X_train = train_split[feature_cols]

    model = joblib.load(MODEL_DIR / "decision_tree_model.pkl")

    print("Building LIME explainer...")
    explainer = LimeTabularExplainer(
        training_data=X_train.values, feature_names=feature_cols,
        class_names=["No Default", "Default"], mode="classification",
        discretize_continuous=False, random_state=RANDOM_SEED,
    )
    predict_fn = model.predict_proba

    print(f"Generating LIME explanations for {len(neighbor_ids)} neighbor applicants...")
    all_rows = []
    X_neighbors_values = X_neighbors.values
    for i, row in enumerate(X_neighbors_values):
        explanation = explainer.explain_instance(row, predict_fn, num_features=len(feature_cols))
        weights_by_index = dict(explanation.as_map()[1])
        row_weights = [weights_by_index.get(idx, 0.0) for idx in range(len(feature_cols))]
        all_rows.append(row_weights)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1} / {len(neighbor_ids)} done")

    neighbor_explanations = pd.DataFrame(all_rows, columns=feature_cols, index=neighbor_ids)
    neighbor_explanations.to_csv(SIMILARITY_DIR / "lime_decision_tree_neighbors.csv")
    print(f"\nSaved neighbor explanations to {SIMILARITY_DIR / 'lime_decision_tree_neighbors.csv'}")

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
    scores_path = SIMILARITY_DIR / "stability_scores_lime_decision_tree.csv"
    results_df.to_csv(scores_path, index=False)
    print(f"Saved per-anchor stability scores to {scores_path}")

    print(f"\nDecision Tree - LIME similarity stability")
    print(f"  Avg feature variance: {results_df['avg_feature_variance'].mean():.6f}")
    print(f"  Avg rank correlation: {results_df['avg_rank_correlation'].mean():.4f}")


if __name__ == "__main__":
    main()