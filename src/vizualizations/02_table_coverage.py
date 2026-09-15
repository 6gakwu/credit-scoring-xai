"""
Figure 2 - how many applicants actually have a record in each supporting
table (bureau, previous applications, installments), since the merge is
a left join and not everyone has history everywhere.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\data\processed")
OUTPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\outputs\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(INPUT_DIR / "train_merged.csv")

has_bureau = (train["TOTAL_PREVIOUS_LOANS"] > 0).mean() * 100
has_prev = (train["TOTAL_PREVIOUS_APPLICATIONS"] > 0).mean() * 100
has_inst = (train["TOTAL_INSTALLMENT_RECORDS"] > 0).mean() * 100
has_none = (
    (train["TOTAL_PREVIOUS_LOANS"] == 0)
    & (train["TOTAL_PREVIOUS_APPLICATIONS"] == 0)
    & (train["TOTAL_INSTALLMENT_RECORDS"] == 0)
).mean() * 100
has_all = (
    (train["TOTAL_PREVIOUS_LOANS"] > 0)
    & (train["TOTAL_PREVIOUS_APPLICATIONS"] > 0)
    & (train["TOTAL_INSTALLMENT_RECORDS"] > 0)
).mean() * 100

categories = ["Has bureau\nhistory", "Has previous\napplication history", "Has installment\nhistory", "Has all\nthree", "Has none\nof the three"]
values = [has_bureau, has_prev, has_inst, has_all, has_none]
colors = ["#4C78A8", "#4C78A8", "#4C78A8", "#54A24B", "#B54A4A"]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(categories, values, color=colors, width=0.6)
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel("% of applicants (training set)")
ax.set_ylim(0, 105)
ax.set_title("Applicant Coverage Across Supporting Tables", fontsize=12, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_table_coverage.png", dpi=180, facecolor='white')
print("Saved:", OUTPUT_DIR / "02_table_coverage.png")
