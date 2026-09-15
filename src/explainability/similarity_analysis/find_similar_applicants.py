"""
RQ2, part 1 - finding each anchor applicant's closest real look-alikes.

Picks 30 applicants (a fixed subset of our original 200) as "anchors,"
then searches the ENTIRE validation set (61,502 applicants, not just
our small 200 applicant sample) to find each anchor's 5 nearest real neighbors.

"Closeness" is measured using scaled Euclidean distance across all 71
features 

This script only finds the groups - it doesn't generate any explanations yet.
That happens in the next 8 scripts in this folder, one per model/method.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import joblib

DATA_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\data\final datasets")
MODEL_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\models\saved_models")
EXPLAIN_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability")
OUTPUT_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability\similarity_analysis")

RANDOM_SEED = 42
VALIDATION_SIZE = 0.2
N_ANCHORS = 30
N_NEIGHBORS = 5


def main():
    train = pd.read_csv(DATA_DIR / "train_selected.csv")
    feature_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    train_split, val_split = train_test_split(
        train, test_size=VALIDATION_SIZE, stratify=train["TARGET"], random_state=RANDOM_SEED
    )

    sample_ids = pd.read_csv(EXPLAIN_DIR / "explanation_sample_applicants.csv")["SK_ID_CURR"]
    anchor_ids = sample_ids.sample(n=N_ANCHORS, random_state=RANDOM_SEED)

   
    scaler = joblib.load(MODEL_DIR / "logistic_regression_scaler.pkl")

    val_indexed = val_split.set_index("SK_ID_CURR")
    X_val_scaled = scaler.transform(val_indexed[feature_cols])
    val_ids_array = val_indexed.index.values

    print(f"Searching {len(val_indexed):,} validation applicants for each anchor's nearest neighbors...")

    groups = []
    for anchor_id in anchor_ids:
        anchor_pos = np.where(val_ids_array == anchor_id)[0][0]
        anchor_vector = X_val_scaled[anchor_pos].reshape(1, -1)

        distances = pairwise_distances(anchor_vector, X_val_scaled, metric="euclidean")[0]
        distances[anchor_pos] = np.inf  # an applicant can't be their own neighbor

        nearest_positions = np.argsort(distances)[:N_NEIGHBORS]
        neighbor_ids = val_ids_array[nearest_positions]
        neighbor_distances = distances[nearest_positions]

        for rank, (nid, dist) in enumerate(zip(neighbor_ids, neighbor_distances), start=1):
            groups.append({
                "anchor_id": anchor_id, "neighbor_id": nid,
                "neighbor_rank": rank, "distance": dist,
            })

    groups_df = pd.DataFrame(groups)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "similar_applicant_groups.csv"
    groups_df.to_csv(output_path, index=False)

    print(f"Groups shape: {groups_df.shape} (expecting ({N_ANCHORS * N_NEIGHBORS}, 4))")
    print(f"Saved to {output_path}")
    print(f"\nUnique neighbor applicants found: {groups_df['neighbor_id'].nunique()}")
    print(f"Number of neighbors already in our 200-applicant sample: {groups_df['neighbor_id'].isin(sample_ids).sum()}")


if __name__ == "__main__":
    main()