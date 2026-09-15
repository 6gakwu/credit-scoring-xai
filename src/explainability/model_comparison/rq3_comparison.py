"""
RQ3 - pulling RQ1 and RQ2 results together into one comparison across
all 4 models, testing whether the differences are statistically real,
and building the comparison charts.

Reads the 8 files already sitting in perturbation_analysis/ (RQ1) and
the 8 already sitting in similarity_analysis/ (RQ2)this just combines and analyzes what those two steps
already produced.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations

PERTURB_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability\perturbation_analysis")
SIMILARITY_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability\similarity_analysis")
OUTPUT_DIR = Path(r"C:\Users\jeffo\OneDrive\Dokumenter\HomeCredit_XAI_Project\explainability\model_comparison")

MODELS = ["logistic_regression", "decision_tree", "random_forest", "xgboost"]
MODEL_LABELS = ["Logistic\nRegression", "Decision\nTree", "Random\nForest", "XGBoost"]
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]


def load_rq1(method, model):
    return pd.read_csv(PERTURB_DIR / f"stability_scores_{method}_{model}.csv")["avg_cosine_similarity"].values


def load_rq2(method, model):
    df = pd.read_csv(SIMILARITY_DIR / f"stability_scores_{method}_{model}_RQ2.csv")
    return df["avg_feature_variance"].values, df["avg_rank_correlation"].values


def build_summary_table():
    rows = []
    for model in MODELS:
        for method in ["shap", "lime"]:
            pert = load_rq1(method, model)
            variance, rank_corr = load_rq2(method, model)
            rows.append({
                "model": model, "method": method,
                "perturbation_stability_avg": pert.mean(),
                "feature_variance_avg": variance.mean(),
                "rank_consistency_avg": rank_corr.mean(),
            })
    return pd.DataFrame(rows)


def run_statistical_tests(log_lines):
    """Kruskal-Wallis across all 4 models, then pairwise Mann-Whitney U
    with Bonferroni correction wherever the overall test is significant."""

    def test_metric(data_dict, metric_name):
        log_lines.append(f"\n{'='*60}\n{metric_name}\n{'='*60}")
        groups = [data_dict[m] for m in MODELS]
        stat, p = kruskal(*groups)
        log_lines.append(f"Kruskal-Wallis across all 4 models: H={stat:.3f}, p={p:.6f}")

        if p < 0.05:
            log_lines.append("-> Significant overall difference exists. Pairwise comparisons:")
            pairs = list(combinations(MODELS, 2))
            for m1, m2 in pairs:
                stat2, p2 = mannwhitneyu(data_dict[m1], data_dict[m2])
                p2_corrected = min(p2 * len(pairs), 1.0)  # Bonferroni correction
                sig = "SIGNIFICANT" if p2_corrected < 0.05 else "not significant"
                log_lines.append(f"   {m1} vs {m2}: p={p2:.4f}, corrected p={p2_corrected:.4f} -> {sig}")
        else:
            log_lines.append("-> No significant overall difference found.")

    for method in ["shap", "lime"]:
        data = {m: load_rq1(method, m) for m in MODELS}
        test_metric(data, f"Perturbation Stability ({method.upper()})")

    for method in ["shap", "lime"]:
        data = {m: load_rq2(method, m)[1] for m in MODELS}  # rank correlation only - variance isn't comparable across models
        test_metric(data, f"Rank Consistency ({method.upper()})")


def make_chart(shap_data, lime_data, ylabel, title, filename, ylim=None, seed=42):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharey=True)
    for ax, data, method_label in zip(axes, [shap_data, lime_data], ["SHAP", "LIME"]):
        box_data = [data[m] for m in MODELS]
        bp = ax.boxplot(box_data, tick_labels=MODEL_LABELS, patch_artist=True, showfliers=False, widths=0.5)
        for patch, color in zip(bp['boxes'], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for i, (m, color) in enumerate(zip(MODELS, COLORS)):
            y = data[m]
            x = rng.normal(i + 1, 0.05, size=len(y))
            ax.scatter(x, y, alpha=0.6, s=18, color=color, edgecolors='none')
        ax.set_title(method_label, fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        if ylim:
            ax.set_ylim(ylim)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=170, facecolor='white')
    plt.close()
    print(f"Saved {filename}")


def build_charts():
    shap_pert = {m: load_rq1("shap", m) for m in MODELS}
    lime_pert = {m: load_rq1("lime", m) for m in MODELS}
    make_chart(shap_pert, lime_pert, "Cosine similarity (original vs. nudged)",
               "Perturbation Stability Across Models (RQ1)", "rq3_perturbation_stability.png", ylim=(0.4, 1.02))

    shap_rank = {m: load_rq2("shap", m)[1] for m in MODELS}
    lime_rank = {m: load_rq2("lime", m)[1] for m in MODELS}
    make_chart(shap_rank, lime_rank, "Spearman rank correlation (anchor vs. neighbors)",
               "Rank Consistency Across Similar Applicants (RQ2)", "rq3_rank_consistency.png", ylim=(0.0, 1.05))

    shap_var = {m: load_rq2("shap", m)[0] for m in MODELS}
    lime_var = {m: load_rq2("lime", m)[0] for m in MODELS}
    make_chart(shap_var, lime_var, "Avg. feature-importance variance",
               "Feature Variance Across Similar Applicants (RQ2)\n(not directly comparable across models - see note in log)",
               "rq3_feature_variance.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building summary table...")
    summary = build_summary_table()
    summary.to_csv(OUTPUT_DIR / "rq3_model_comparison.csv", index=False)
    print(summary.round(4).to_string(index=False))

    print("\nRunning statistical tests...")
    log_lines = []
    run_statistical_tests(log_lines)
    log_text = "\n".join(log_lines)
    print(log_text)
    (OUTPUT_DIR / "rq3_statistical_tests.txt").write_text(log_text)
    print(f"\nSaved statistical test results to {OUTPUT_DIR / 'rq3_statistical_tests.txt'}")

    print("\nBuilding charts...")
    build_charts()

    print(f"\nAll RQ3 outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()