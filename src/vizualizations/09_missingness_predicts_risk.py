"""
Figure 9 - does missingness itself relate to default risk? Compares
default rate for applicants with a value present vs originally missing,
across a few features.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\data\processed")
OUTPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\outputs\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(INPUT_DIR / "train_merged.csv")
# OCCUPATION_TYPE is already target-encoded by this point in train_merged.csv,
# so its missing/present split has to come from the pre-encoding file instead
app = pd.read_csv(INPUT_DIR / "application_train_processed.csv")

# EXT_SOURCE_1 / EXT_SOURCE_3 / social circle - the *_MISSING flags are
# already sitting right in train_merged.csv
ext1_present = train.loc[train["EXT_SOURCE_1_MISSING"] == 0, "TARGET"].mean() * 100
ext1_missing = train.loc[train["EXT_SOURCE_1_MISSING"] == 1, "TARGET"].mean() * 100

ext3_present = train.loc[train["EXT_SOURCE_3_MISSING"] == 0, "TARGET"].mean() * 100
ext3_missing = train.loc[train["EXT_SOURCE_3_MISSING"] == 1, "TARGET"].mean() * 100

social_present = train.loc[train["SOCIAL_CIRCLE_MISSING"] == 0, "TARGET"].mean() * 100
social_missing = train.loc[train["SOCIAL_CIRCLE_MISSING"] == 1, "TARGET"].mean() * 100

# occupation type - "Unknown" in the pre-encoding file means it was blank
occ_present = app.loc[app["OCCUPATION_TYPE"] != "Unknown", "TARGET"].mean() * 100
occ_missing = app.loc[app["OCCUPATION_TYPE"] == "Unknown", "TARGET"].mean() * 100

groups = ['EXT_SOURCE_1', 'EXT_SOURCE_3', 'Social circle\ncounts', 'Occupation\ntype']
present = [ext1_present, ext3_present, social_present, occ_present]
missing = [ext1_missing, ext3_missing, social_missing, occ_missing]

overall = train['TARGET'].mean() * 100

x = np.arange(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))
b1 = ax.bar(x - width / 2, present, width, label='Value present', color='#4C78A8')
b2 = ax.bar(x + width / 2, missing, width, label='Value originally missing', color='#F58518')

for bars in [b1, b2]:
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{b.get_height():.2f}%", ha='center', fontsize=9)

ax.axhline(overall, color='#888888', linestyle=':', linewidth=1.2)
ax.text(3.6, overall + 0.2, f'overall {overall:.2f}%', fontsize=8, color='#888888')

ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel('Default rate')
ax.set_title('Default Rate: Value Present vs. Originally Missing', fontsize=12, fontweight='bold')
ax.legend(frameon=False)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_missingness_predicts_risk.png", dpi=180, facecolor='white')
print("Saved:", OUTPUT_DIR / "09_missingness_predicts_risk.png")
