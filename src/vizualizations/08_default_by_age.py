"""
Figure 8 - default rate by age bracket.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\data\processed")
OUTPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\outputs\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(INPUT_DIR / "train_merged.csv")

train['AGE_BRACKET'] = pd.cut(train['AGE_YEARS'], bins=[20, 30, 40, 50, 60, 70],
                               labels=['20-30', '30-40', '40-50', '50-60', '60-70'])
rates = train.groupby('AGE_BRACKET', observed=True)['TARGET'].mean() * 100
overall = train['TARGET'].mean() * 100

fig, ax = plt.subplots(figsize=(7.5, 5))
bars = ax.bar(rates.index.astype(str), rates.values, color='#4C78A8', width=0.55)
for b, v in zip(bars, rates.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%", ha='center', fontsize=10)
ax.axhline(overall, color='#B54A4A', linestyle='--', linewidth=1.3, label=f'Overall rate ({overall:.2f}%)')
ax.set_xlabel('Age bracket (years)')
ax.set_ylabel('Default rate')
ax.set_title('Default Rate by Age Bracket', fontsize=12, fontweight='bold')
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_default_by_age.png", dpi=180, facecolor='white')
print("Saved:", OUTPUT_DIR / "08_default_by_age.png")
