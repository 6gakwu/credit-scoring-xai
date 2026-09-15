"""
Figure 10 - correlation matrix across the key numeric features, TARGET
included.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\data\processed")
OUTPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\outputs\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(INPUT_DIR / "train_merged.csv")

cols = [
    'TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'CREDIT_GOODS_RATIO',
    'CREDIT_TO_INCOME_RATIO', 'ANNUITY_TO_INCOME_RATIO', 'AGE_YEARS', 'EMPLOYMENT_YEARS',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'TOTAL_PREVIOUS_LOANS', 'TOTAL_BUREAU_DEBT', 'PREVIOUS_APPROVAL_RATE',
    'LATE_PAYMENT_RATE', 'REGION_POPULATION_RELATIVE',
]
corr = train[cols].corr()

fig, ax = plt.subplots(figsize=(10.5, 9))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=90, fontsize=8.5)
ax.set_yticklabels(cols, fontsize=8.5)

for i in range(len(cols)):
    for j in range(len(cols)):
        v = corr.iloc[i, j]
        color = 'white' if abs(v) > 0.6 else 'black'
        ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=6.5, color=color)

fig.colorbar(im, ax=ax, shrink=0.8, label='Correlation')
ax.set_title('Correlation Among Key Numeric Features', fontsize=12, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_correlation_heatmap.png", dpi=170, facecolor='white')
print("Saved:", OUTPUT_DIR / "10_correlation_heatmap.png")
