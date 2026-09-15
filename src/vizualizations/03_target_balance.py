"""
Figure 3 - how imbalanced TARGET actually is (defaulters vs everyone else).
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\data\processed")
OUTPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\outputs\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(INPUT_DIR / "train_merged.csv")

counts = train["TARGET"].value_counts().sort_index()
pcts = counts / counts.sum() * 100

fig, ax = plt.subplots(figsize=(6.5, 5))
bars = ax.bar(['No Default (0)', 'Default (1)'], counts.values, color=['#4C78A8', '#B54A4A'], width=0.5)
for b, c, p in zip(bars, counts.values, pcts.values):
    ax.text(b.get_x() + b.get_width() / 2, c + 3000, f"{c:,}\n({p:.2f}%)", ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel("Number of applicants")
ax.set_title("TARGET Class Balance (Training Set)", fontsize=12, fontweight='bold')
ax.set_ylim(0, max(counts.values) * 1.18)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_target_balance.png", dpi=180, facecolor='white')
print("Saved:", OUTPUT_DIR / "03_target_balance.png")
