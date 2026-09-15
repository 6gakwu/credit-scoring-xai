"""
Decision Tree.

No scaling used here unlike logistion regression, as trees are not sensitive to feature scale.
Max tree depth capped at 10 to avoid overfitting, 
as unrestricted trees can memorize the training data and perform poorly on validation data.

"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

DATA_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\data\final datasets")
RESULTS_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\outputs\results")
MODEL_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\models\saved_models")

RANDOM_SEED = 42
VALIDATION_SIZE = 0.2
MAX_DEPTH = 10  # standard baseline value 


def main():
    train = pd.read_csv(DATA_DIR / "train_selected.csv")

    feature_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    X = train[feature_cols]
    y = train["TARGET"]

    # identical split settings to every other model in this project 
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    # class_weight="balanced" does the same job here as it did for Logistic
    # Regression - weights each defaulter row roughly 11.4x a non-defaulter
    # row, so the tree can't just ignore the minority class
    model = DecisionTreeClassifier(
        max_depth=MAX_DEPTH, class_weight="balanced", random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    results = {
        "model": "Decision Tree",
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
    }

    print("Decision Tree - validation results")
    for key, value in results.items():
        if key != "model":
            print(f"  {key}: {value:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(RESULTS_DIR / "decision_tree_results.csv", index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "decision_tree_model.pkl")
    print(f"Model saved to {MODEL_DIR}")
   


if __name__ == "__main__":
    main()