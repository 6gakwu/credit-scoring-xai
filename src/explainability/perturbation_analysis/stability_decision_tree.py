"""
RQ1, part 2 - Decision Tree perturbation stability.

Same approach as Logistic Regression - generate fresh SHAP explanations
for the same 250 nudged applicant versions, then compare each one back
against that applicant's original explanation using cosine similarity.

"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import shap

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
    original_explanations = pd.read_csv(EXPLAIN_DIR / "shap_decision_tree.csv")

    model = joblib.load(MODEL_DIR / "decision_tree_model.pkl")

    background = shap.sample(X_train, 100, random_state=RANDOM_SEED)

    print("Building SHAP explainer...")
    predict_fn = lambda x: model.predict_proba(x)[:, 1]
    explainer = shap.Explainer(predict_fn, background, feature_names=feature_cols)

    print(f"Generating SHAP explanations for {len(perturbed)} perturbed applicant versions...")
    perturbed_shap_values = explainer(perturbed[feature_cols])

    print(f"\nPerturbed SHAP values shape: {perturbed_shap_values.values.shape} (expecting (250, {len(feature_cols)}))")

    perturbed_shap_df = pd.DataFrame(perturbed_shap_values.values, columns=feature_cols)
    perturbed_shap_df.insert(0, "perturbation_number", perturbed["perturbation_number"].values)
    perturbed_shap_df.insert(0, "SK_ID_CURR", perturbed["SK_ID_CURR"].values)

    raw_output_path = PERTURB_DIR / "shap_decision_tree_perturbed.csv"
    perturbed_shap_df.to_csv(raw_output_path, index=False)
    print(f"Saved raw perturbed explanations to {raw_output_path}")

    print("\nComputing cosine similarity between original and perturbed explanations...")
    original_indexed = original_explanations.set_index("SK_ID_CURR")

    applicant_scores = []
    for sk_id in perturbed["SK_ID_CURR"].unique():
        original_vector = original_indexed.loc[sk_id, feature_cols].values.reshape(1, -1)
        perturbed_vectors = perturbed_shap_df[perturbed_shap_df["SK_ID_CURR"] == sk_id][feature_cols].values

        similarities = cosine_similarity(original_vector, perturbed_vectors)[0]
        avg_similarity = similarities.mean()

        applicant_scores.append({
            "SK_ID_CURR": sk_id,
            "avg_cosine_similarity": avg_similarity,
            "min_cosine_similarity": similarities.min(),
            "max_cosine_similarity": similarities.max(),
        })

    scores_df = pd.DataFrame(applicant_scores)
    scores_path = PERTURB_DIR / "stability_scores_shap_decision_tree.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"Saved per-applicant stability scores to {scores_path}")

    print(f"\nDecision Tree - SHAP perturbation stability")
    print(f"  Overall average cosine similarity: {scores_df['avg_cosine_similarity'].mean():.4f}")
    print(f"  Lowest single applicant score: {scores_df['avg_cosine_similarity'].min():.4f}")
    print(f"  Highest single applicant score: {scores_df['avg_cosine_similarity'].max():.4f}")


if __name__ == "__main__":
    main()