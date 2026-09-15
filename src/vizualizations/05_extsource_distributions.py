"""
Figure 5 - distributions of the three external risk scores.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\data\processed")
OUTPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\outputs\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(INPUT_DIR / "train_merged.csv")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
colors = ['#4C78A8', '#54A24B', '#E45756']
for ax, col, c in zip(axes, cols, colors):
    ax.hist(train[col], bins=50, color=c, edgecolor='white')
    ax.set_title(col)
    ax.set_xlabel('Score (imputed where originally missing)')
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Applicants')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_extsource_distributions.png", dpi=170, facecolor='white')
print("Saved:", OUTPUT_DIR / "05_extsource_distributions.png")
