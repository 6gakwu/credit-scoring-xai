"""
Figure 6 - how much genuine missingness each column had before any
imputation. Uses application_train_processed.csv since that's the point
in the pipeline where real NaNs are still intact (before the merge step
and before postprocess_data_quality.py fills anything in).
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\data\processed")
OUTPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\outputs\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = pd.read_csv(INPUT_DIR / "application_train_processed.csv")
total = len(app)

# the 6 bureau enquiry columns and 4 social circle columns are always
# missing together, so they're reported as one grouped bar each rather
# than 6/4 near-identical bars
bureau_cols = [
    "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
]
social_cols = [
    "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
]

own_car_age_pct = app["OWN_CAR_AGE"].isna().mean() * 100
ext1_pct = app["EXT_SOURCE_1"].isna().mean() * 100
ext3_pct = app["EXT_SOURCE_3"].isna().mean() * 100
bureau_pct = app[bureau_cols[0]].isna().mean() * 100
suite_pct = app["NAME_TYPE_SUITE"].isna().mean() * 100
social_pct = app[social_cols[0]].isna().mean() * 100
ext2_pct = app["EXT_SOURCE_2"].isna().mean() * 100
goods_pct = app["AMT_GOODS_PRICE"].isna().mean() * 100

labels = [
    "OWN_CAR_AGE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_3",
    "AMT_REQ_CREDIT_BUREAU_*\n(6 columns)",
    "NAME_TYPE_SUITE",
    "Social circle counts\n(4 columns)",
    "EXT_SOURCE_2",
    "AMT_GOODS_PRICE",
]
values = [own_car_age_pct, ext1_pct, ext3_pct, bureau_pct, suite_pct, social_pct, ext2_pct, goods_pct]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(labels[::-1], values[::-1], color='#4C78A8')
for b, v in zip(bars, values[::-1]):
    ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.2f}%", va='center', fontsize=9)

ax.set_xlabel("% of applicants with a genuinely missing value")
ax.set_title("True Missingness by Column (Before Imputation)", fontsize=12, fontweight='bold')
ax.set_xlim(0, 78)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_missingness_overview.png", dpi=180, facecolor='white')
print("Saved:", OUTPUT_DIR / "06_missingness_overview.png")
