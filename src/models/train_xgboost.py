"""
XGBoost.

Same data, same validation split as the other three models. No feature
scaling needed cos trees are unaffected by feature scale.

Class weighting here is called scale_pos_weight - the ratio of
non-defaulters to defaulters, calculated straight from the training
data. Comes out to about 11.4, matching the weighting used throughout
this project.

"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib

DATA_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\data\final datasets")
RESULTS_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\outputs\results")
MODEL_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\models\saved_models")

RANDOM_SEED = 42
VALIDATION_SIZE = 0.2


def main():
    train = pd.read_csv(DATA_DIR / "train_selected.csv")

    feature_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    X = train[feature_cols]
    y = train["TARGET"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.3,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    results = {
        "model": "XGBoost",
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
        "decision_threshold_used": 0.5,
    }

    print("XGBoost - validation results")
    for key, value in results.items():
        if key != "model":
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(RESULTS_DIR / "xgboost_results.csv", index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "xgboost_model.pkl")
    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()