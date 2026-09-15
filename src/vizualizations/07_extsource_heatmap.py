"""
Figure 7 - default rate by EXT_SOURCE_1 and EXT_SOURCE_2, binned.

Started out as a scatter plot coloured by default status, but with only
8% of applicants defaulting, the minority class is sparse everywhere and
raw dot density doesn't actually tell you where risk is highest - the
riskiest corner of the grid also happens to have the fewest applicants
in it, so it never looked "dense" no matter what. Binning and computing
the real rate per cell is a far more honest way to show this.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\data\processed")
OUTPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\outputs\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(INPUT_DIR / "train_merged.csv")

n_bins = 7  # enough resolution to see the gradient, still keeps every cell's
            # sample size big enough to trust (smallest cell here is >100 rows)
e1_edges = np.linspace(train['EXT_SOURCE_1'].min(), train['EXT_SOURCE_1'].max(), n_bins + 1)
e2_edges = np.linspace(train['EXT_SOURCE_2'].min(), train['EXT_SOURCE_2'].max(), n_bins + 1)

train['e1_bin'] = pd.cut(train['EXT_SOURCE_1'], bins=e1_edges, include_lowest=True)
train['e2_bin'] = pd.cut(train['EXT_SOURCE_2'], bins=e2_edges, include_lowest=True)

pivot = train.pivot_table(values='TARGET', index='e2_bin', columns='e1_bin', aggfunc='mean', observed=False) * 100
pivot = pivot.sort_index(ascending=True)  # low EXT_SOURCE_2 at the bottom, same orientation as a normal scatter

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(pivot.values, origin='lower', cmap='RdYlGn_r', aspect='auto',
                extent=[e1_edges[0], e1_edges[-1], e2_edges[0], e2_edges[-1]])

cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Default rate (%)')

ax.set_xlabel('EXT_SOURCE_1')
ax.set_ylabel('EXT_SOURCE_2')
ax.set_title('Default Rate by EXT_SOURCE_1 and EXT_SOURCE_2\n(binned across full training set)', fontsize=12, fontweight='bold')

# label each cell with its actual rate so the numbers are there for anyone
# who wants the precise value, not just the colour
for i, e2 in enumerate(pivot.index):
    for j, e1 in enumerate(pivot.columns):
        val = pivot.values[i, j]
        if not np.isnan(val):
            x = (e1_edges[j] + e1_edges[j + 1]) / 2
            y = (e2_edges[i] + e2_edges[i + 1]) / 2
            color = 'white' if val > 18 or val < 4 else 'black'
            ax.text(x, y, f'{val:.1f}', ha='center', va='center', fontsize=8, color=color)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_extsource_heatmap.png", dpi=180, facecolor='white')
print("Saved:", OUTPUT_DIR / "07_extsource_heatmap.png")
