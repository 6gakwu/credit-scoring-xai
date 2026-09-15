"""
Random Forest.

"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

DATA_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\data\final datasets")
RESULTS_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\outputs\results")
MODEL_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\models\saved_models")

RANDOM_SEED = 42
VALIDATION_SIZE = 0.2
N_ESTIMATORS = 100  # standard default - how many trees to grow and average


def main():
    train = pd.read_csv(DATA_DIR / "train_selected.csv")

    feature_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    X = train[feature_cols]
    y = train["TARGET"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # the real-world default rate, taken from training data only - 8.07%
    decision_threshold = y_train.mean()
    y_pred = (y_pred_proba >= decision_threshold).astype(int)

    results = {
        "model": "Random Forest",
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_pred_proba),  # ROC-AUC doesn't depend on the cutoff at all
        "decision_threshold_used": decision_threshold,
    }

    print("Random Forest - validation results")
    for key, value in results.items():
        if key != "model":
            print(f"  {key}: {value:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(RESULTS_DIR / "random_forest_results.csv", index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "random_forest_model.pkl")
    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()