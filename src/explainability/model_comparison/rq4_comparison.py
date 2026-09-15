"""
RQ4 - does higher predictive performance correspond to more stable
explanations, or is there a trade-off?

Combines Model performance results with the RQ1/RQ2 stability results
already sitting in perturbation_analysis/ and similarity_analysis/.
this just pulls together work that's already done.

Uses ROC-AUC specifically for performance, not accuracy/precision/recall/
F1 - those depend on the decision cutoff each model used, and Random
Forest used a different cutoff (8.07%) than the other three (0.50), so
they aren't directly comparable. ROC-AUC doesn't have that problem.

Feature variance is left out of this comparison, same reason as RQ3 -
it's not comparable across models due to Random Forest's naturally
compressed contribution values.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.stats import pearsonr

RESULTS_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\outputs\results")
PERTURB_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability\perturbation_analysis")
SIMILARITY_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability\similarity_analysis")
OUTPUT_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability\model_comparison")

MODELS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
LABELS = {"logistic_regression": "Logistic Regression", "decision_tree": "Decision Tree",
          "random_forest": "Random Forest", "xgboost": "XGBoost"}
COLORS = {"logistic_regression": "#4C78A8", "decision_tree": "#F58518",
          "random_forest": "#54A24B", "xgboost": "#E45756"}

# result CSV filenames from Step 7, one per model
RESULTS_FILES = {
    "logistic_regression": "logistic_regression_results.csv",
    "decision_tree": "decision_tree_results.csv",
    "random_forest": "random_forest_results.csv",
    "xgboost": "xgboost_results.csv",
}


def load_performance():
    roc_auc = {}
    for model, filename in RESULTS_FILES.items():
        df = pd.read_csv(RESULTS_DIR / filename)
        roc_auc[model] = df["roc_auc"].iloc[0]
    return roc_auc


def build_table(roc_auc):
    rows = []
    for model in MODELS:
        for method in ["shap", "lime"]:
            pert = pd.read_csv(PERTURB_DIR / f"stability_scores_{method}_{model}.csv")["avg_cosine_similarity"].mean()
            rank_file = SIMILARITY_DIR / f"stability_scores_{method}_{model}_RQ2.csv"
            rank = pd.read_csv(rank_file)["avg_rank_correlation"].mean()
            rows.append({
                "model": model, "method": method, "roc_auc": roc_auc[model],
                "perturbation_stability": pert, "rank_consistency": rank,
            })
    return pd.DataFrame(rows)


def make_scatter(df, ycol, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for _, row in df.iterrows():
        marker = 'o' if row['method'] == 'shap' else '^'
        # nudge SHAP labels slightly left/up and LIME slightly right/down,
        # so overlapping points (e.g. Logistic Regression's two very
        # close scores) don't print on top of each other
        offset = (-45, 8) if row['method'] == 'shap' else (8, -12)
        ax.scatter(row['roc_auc'], row[ycol], s=140, color=COLORS[row['model']], marker=marker,
                   edgecolors='black', linewidth=0.8, alpha=0.85, zorder=3)
        ax.annotate(f"{LABELS[row['model']]} ({row['method'].upper()})", (row['roc_auc'], row[ycol]),
                    textcoords="offset points", xytext=offset, fontsize=7.5)
    ax.set_xlabel("Predictive performance (ROC-AUC)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.2)
    legend_elems = [Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='SHAP'),
                     Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='LIME')]
    ax.legend(handles=legend_elems, loc='lower right', frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170, facecolor='white')
    plt.close()
    print(f"Saved {filename}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Model performance results...")
    roc_auc = load_performance()
    for model, val in roc_auc.items():
        print(f"  {model}: ROC-AUC = {val:.4f}")

    print("\nBuilding combined performance/stability table...")
    df = build_table(roc_auc)
    df.to_csv(OUTPUT_DIR / "rq4_performance_vs_stability.csv", index=False)
    print(df.round(4).to_string(index=False))

    print("\nCalculating correlations (descriptive only - see note in the script header)...")
    r1, p1 = pearsonr(df["roc_auc"], df["perturbation_stability"])
    r2, p2 = pearsonr(df["roc_auc"], df["rank_consistency"])
    log_lines = [
        f"Performance vs. perturbation stability: r={r1:.3f}, p={p1:.3f}",
        f"Performance vs. rank consistency: r={r2:.3f}, p={p2:.3f}",
        "",
        "Note: with only 4 models (8 points including both explanation methods),",
        "these correlations are descriptive of the observed pattern, not a",
        "statistically rigorous test of a performance-stability trade-off.",
    ]
    log_text = "\n".join(log_lines)
    print(log_text)
    (OUTPUT_DIR / "rq4_correlation_notes.txt").write_text(log_text)

    print("\nBuilding charts...")
    make_scatter(df, "perturbation_stability", "Perturbation stability (avg. cosine similarity)",
                 "Performance vs. Perturbation Stability (RQ4)", "rq4_performance_vs_perturbation.png")
    make_scatter(df, "rank_consistency", "Rank consistency (avg. Spearman correlation)",
                 "Performance vs. Rank Consistency (RQ4)", "rq4_performance_vs_rank_consistency.png")

    print(f"\nAll RQ4 outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()