"""
RQ1, part 2 - Random Forest LIME perturbation stability. Same approach
as the other LIME stability scripts. 
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from lime.lime_tabular import LimeTabularExplainer

DATA_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\data\final datasets")
MODEL_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\models\saved_models")
EXPLAIN_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability")
PERTURB_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability\perturbation_analysis")

RANDOM_SEED = 42
VALIDATION_SIZE = 0.2


def get_train_split():
    train = pd.read_csv(DATA_DIR / "train_selected.csv")
    feature_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    train_split, _ = train_test_split(
        train, test_size=VALIDATION_SIZE, stratify=train["TARGET"], random_state=RANDOM_SEED
    )
    return train_split, feature_cols


def main():
    train_split, feature_cols = get_train_split()
    X_train = train_split[feature_cols]

    perturbed = pd.read_csv(PERTURB_DIR / "perturbed_applicants.csv")
    original_explanations = pd.read_csv(EXPLAIN_DIR / "lime_random_forest.csv")

    model = joblib.load(MODEL_DIR / "random_forest_model.pkl")

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

    print(f"Generating LIME explanations for {len(perturbed)} perturbed applicant versions...")

    all_rows = []
    perturbed_values = perturbed[feature_cols].values
    for i, row in enumerate(perturbed_values):
        explanation = explainer.explain_instance(row, predict_fn, num_features=len(feature_cols))
        weights_by_index = dict(explanation.as_map()[1])
        row_weights = [weights_by_index.get(idx, 0.0) for idx in range(len(feature_cols))]
        all_rows.append(row_weights)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1} / {len(perturbed)} done")

    perturbed_lime_df = pd.DataFrame(all_rows, columns=feature_cols)
    perturbed_lime_df.insert(0, "perturbation_number", perturbed["perturbation_number"].values)
    perturbed_lime_df.insert(0, "SK_ID_CURR", perturbed["SK_ID_CURR"].values)

    raw_output_path = PERTURB_DIR / "lime_random_forest_perturbed.csv"
    perturbed_lime_df.to_csv(raw_output_path, index=False)
    print(f"\nSaved raw perturbed explanations to {raw_output_path}")
    print(f"Perturbed LIME values shape: {perturbed_lime_df.shape} (expecting (250, {len(feature_cols) + 2}))")

    print("\nComputing cosine similarity between original and perturbed explanations...")
    original_indexed = original_explanations.set_index("SK_ID_CURR")

    applicant_scores = []
    for sk_id in perturbed["SK_ID_CURR"].unique():
        original_vector = original_indexed.loc[sk_id, feature_cols].values.reshape(1, -1)
        perturbed_vectors = perturbed_lime_df[perturbed_lime_df["SK_ID_CURR"] == sk_id][feature_cols].values

        similarities = cosine_similarity(original_vector, perturbed_vectors)[0]
        avg_similarity = similarities.mean()

        applicant_scores.append({
            "SK_ID_CURR": sk_id,
            "avg_cosine_similarity": avg_similarity,
            "min_cosine_similarity": similarities.min(),
            "max_cosine_similarity": similarities.max(),
        })

    scores_df = pd.DataFrame(applicant_scores)
    scores_path = PERTURB_DIR / "stability_scores_lime_random_forest.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"Saved per-applicant stability scores to {scores_path}")

    print(f"\nRandom Forest - LIME perturbation stability")
    print(f"  Overall average cosine similarity: {scores_df['avg_cosine_similarity'].mean():.4f}")
    print(f"  Lowest single applicant score: {scores_df['avg_cosine_similarity'].min():.4f}")
    print(f"  Highest single applicant score: {scores_df['avg_cosine_similarity'].max():.4f}")


if __name__ == "__main__":
    main()