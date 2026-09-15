"""
Figure 4 - distributions of income, credit amount, age, and the
credit-to-goods ratio.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\data\processed")
OUTPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\outputs\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(INPUT_DIR / "train_merged.csv")

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))

axes[0, 0].hist(train['AMT_INCOME_TOTAL'], bins=80, color='#4C78A8', edgecolor='white')
axes[0, 0].set_title('AMT_INCOME_TOTAL')
axes[0, 0].set_xlim(0, 600000)
axes[0, 0].set_xlabel('Income')
axes[0, 0].set_ylabel('Applicants')

axes[0, 1].hist(train['AMT_CREDIT'], bins=80, color='#54A24B', edgecolor='white')
axes[0, 1].set_title('AMT_CREDIT')
axes[0, 1].set_xlabel('Credit amount')
axes[0, 1].set_ylabel('Applicants')

axes[1, 0].hist(train['AGE_YEARS'], bins=50, color='#E45756', edgecolor='white')
axes[1, 0].set_title('AGE_YEARS')
axes[1, 0].set_xlabel('Age (years)')
axes[1, 0].set_ylabel('Applicants')

# clip the ratio for display only - a handful of extreme ratios would
# otherwise squash the whole histogram into one bar
axes[1, 1].hist(train['CREDIT_GOODS_RATIO'].clip(upper=3), bins=80, color='#F58518', edgecolor='white')
axes[1, 1].set_title('CREDIT_GOODS_RATIO (clipped at 3 for display)')
axes[1, 1].set_xlabel('Credit \u00f7 Goods price')
axes[1, 1].set_ylabel('Applicants')

for ax in axes.flat:
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_numeric_distributions.png", dpi=170, facecolor='white')
print("Saved:", OUTPUT_DIR / "04_numeric_distributions.png")
