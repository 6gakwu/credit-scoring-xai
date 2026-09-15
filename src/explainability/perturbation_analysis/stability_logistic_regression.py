"""
RQ1, part 2 - Logistic Regression perturbation stability.

Takes the 250 nudged applicant versions from the generate_perturbations.py script, generates
a fresh SHAP explanation for each one (same probability-based approach
as the original explanations, so everything stays comparable), then
compares each nudged explanation back against that same applicant's
ORIGINAL explanation (already sitting in shap_logistic_regression.csv
from earlier).

The comparison uses cosine similarity - a score from 0 to 1 answering
"how similar are these two explanations," matching the formula in the
methodology: e(x) . e(x') / (||e(x)|| * ||e(x')||). 1 means identical,
0 means totally unrelated. We get 5 similarity scores per applicant (one
per nudge), average those into one score per applicant, then average
across all 50 applicants into one overall stability number for this
model.
"""

import pandas as pd
import numpy as np
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
    original_explanations = pd.read_csv(EXPLAIN_DIR / "shap_logistic_regression.csv")

    model = joblib.load(MODEL_DIR / "logistic_regression_model.pkl")
    scaler = joblib.load(MODEL_DIR / "logistic_regression_scaler.pkl")

    X_train_scaled = scaler.transform(X_train)
    X_perturbed_scaled = scaler.transform(perturbed[feature_cols])

    # same seed as the original explanation script, so this background
    # sample is identical to the one already used - keeps every comparison
    # on equal footing
    background = shap.sample(X_train_scaled, 100, random_state=RANDOM_SEED)

    print("Building SHAP explainer...")
    predict_fn = lambda x: model.predict_proba(x)[:, 1]
    explainer = shap.Explainer(predict_fn, background, feature_names=feature_cols)

    print(f"Generating SHAP explanations for {len(perturbed)} perturbed applicant versions...")
    perturbed_shap_values = explainer(X_perturbed_scaled)

    print(f"\nPerturbed SHAP values shape: {perturbed_shap_values.values.shape} (expecting (250, {len(feature_cols)}))")

    # attach applicant/perturbation identifiers back onto the raw SHAP output
    perturbed_shap_df = pd.DataFrame(perturbed_shap_values.values, columns=feature_cols)
    perturbed_shap_df.insert(0, "perturbation_number", perturbed["perturbation_number"].values)
    perturbed_shap_df.insert(0, "SK_ID_CURR", perturbed["SK_ID_CURR"].values)

    raw_output_path = PERTURB_DIR / "shap_logistic_regression_perturbed.csv"
    perturbed_shap_df.to_csv(raw_output_path, index=False)
    print(f"Saved raw perturbed explanations to {raw_output_path}")

    # now compare each perturbed explanation back to that applicant's
    # original one, applicant by applicant
    print("\nComputing cosine similarity between original and perturbed explanations...")
    original_indexed = original_explanations.set_index("SK_ID_CURR")

    applicant_scores = []
    for sk_id in perturbed["SK_ID_CURR"].unique():
        original_vector = original_indexed.loc[sk_id, feature_cols].values.reshape(1, -1)
        perturbed_vectors = perturbed_shap_df[perturbed_shap_df["SK_ID_CURR"] == sk_id][feature_cols].values

        similarities = cosine_similarity(original_vector, perturbed_vectors)[0]  # one similarity per nudge
        avg_similarity = similarities.mean()

        applicant_scores.append({
            "SK_ID_CURR": sk_id,
            "avg_cosine_similarity": avg_similarity,
            "min_cosine_similarity": similarities.min(),
            "max_cosine_similarity": similarities.max(),
        })

    scores_df = pd.DataFrame(applicant_scores)
    scores_path = PERTURB_DIR / "stability_scores_shap_logistic_regression.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"Saved per-applicant stability scores to {scores_path}")

    print(f"\nLogistic Regression - SHAP perturbation stability")
    print(f"  Overall average cosine similarity: {scores_df['avg_cosine_similarity'].mean():.4f}")
    print(f"  Lowest single applicant score: {scores_df['avg_cosine_similarity'].min():.4f}")
    print(f"  Highest single applicant score: {scores_df['avg_cosine_similarity'].max():.4f}")


if __name__ == "__main__":
    main()