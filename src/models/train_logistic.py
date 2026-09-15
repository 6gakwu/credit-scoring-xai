"""
Logistic Regression baseline.

Reads the finished feature set, splits off a validation set, Scales the
features (Logistic Regression needs this - more on why below), trains
a class-weighted model, and checks how well it does on data it's never
seen.
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

DATA_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\data\final datasets")
RESULTS_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\outputs\results")
MODEL_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\models\saved_models")

RANDOM_SEED = 42
VALIDATION_SIZE = 0.2  # same 80/20 split used throughout this project


def main():
    train = pd.read_csv(DATA_DIR / "train_selected.csv")

    # SK_ID_CURR is just an applicant ID.
    # TARGET is the answer we're trying to predict, so it comes out of
    # the feature set too - everything else (73 columns) goes in.
    feature_cols = [c for c in train.columns if c not in ("SK_ID_CURR", "TARGET")]
    X = train[feature_cols]
    y = train["TARGET"]

    # same random seed and split size as feature selection, so this
    # validation set matches exactly what we'll use for the other 3
    # models too 
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VALIDATION_SIZE, stratify=y, random_state=RANDOM_SEED
    )

    # Logistic Regression is sensitive to features being on very
    # different scales (income in the hundreds of thousands vs.
    # EXT_SOURCE scores between 0 and 1) - rescaling everything onto a
    # comparable range first makes a real difference,
    # makes it easier for it to interpret
    # fit the scaler on training data only, then reuse it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # class_weight="balanced" automatically weights each defaulter row
    # roughly 11.4x a non-defaulter row, based on the real class counts
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    model.fit(X_train_scaled, y_train)

    # check how it does on applicants it's never seen
    y_pred = model.predict(X_val_scaled)
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]  # probability of default, needed for ROC-AUC

    results = {
        "model": "Logistic Regression",
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1_score": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_pred_proba),
    }

    print("Logistic Regression - validation results")
    for key, value in results.items():
        if key != "model":
            print(f"  {key}: {value:.4f}")

    # save the scores so all 4 models' results can be pulled into one
    # comparison table later
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(RESULTS_DIR / "logistic_regression_results.csv", index=False)

    # save the trained model AND the scaler - SHAP/LIME later need the
    # exact same model, and anything fed into it later needs the exact
    # same scaling applied first
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "logistic_regression_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "logistic_regression_scaler.pkl")
    print(f"Model and scaler saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()