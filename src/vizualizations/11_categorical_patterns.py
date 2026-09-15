"""
Figure 11 - default rate by education level, housing type, and the
biggest organization types. Uses application_train_processed.csv since
that still has the real category text (train_merged.csv has these
target-encoded to numbers by this point).
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

INPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\data\processed")
OUTPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\outputs\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = pd.read_csv(INPUT_DIR / "application_train_processed.csv")
overall = app['TARGET'].mean() * 100

# education - keep the natural low-to-high order, not alphabetical
edu_order = ['Lower secondary', 'Secondary / secondary special', 'Incomplete higher', 'Higher education', 'Academic degree']
edu_rates = app.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().reindex(edu_order) * 100

housing_rates = app.groupby('NAME_HOUSING_TYPE')['TARGET'].mean().sort_values() * 100

# only the 10 biggest organization types - long tail is too noisy/cluttered to show here
org_counts = app['ORGANIZATION_TYPE'].value_counts()
top_orgs = org_counts.head(10).index
org_rates = app[app['ORGANIZATION_TYPE'].isin(top_orgs)].groupby('ORGANIZATION_TYPE')['TARGET'].mean().sort_values() * 100

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

axes[0].bar(range(len(edu_rates)), edu_rates.values, color='#4C78A8')
axes[0].set_xticks(range(len(edu_rates)))
axes[0].set_xticklabels(['Lower\nsecondary', 'Secondary', 'Incomplete\nhigher', 'Higher\neducation', 'Academic\ndegree'], fontsize=8)
axes[0].axhline(overall, color='#B54A4A', linestyle='--', linewidth=1)
axes[0].set_title('By Education Level\n(ordered low \u2192 high)', fontsize=10.5)
axes[0].set_ylabel('Default rate (%)')

axes[1].barh(housing_rates.index, housing_rates.values, color='#54A24B')
axes[1].axvline(overall, color='#B54A4A', linestyle='--', linewidth=1)
axes[1].set_title('By Housing Type', fontsize=10.5)
axes[1].set_xlabel('Default rate (%)')

axes[2].barh(org_rates.index, org_rates.values, color='#F58518')
axes[2].axvline(overall, color='#B54A4A', linestyle='--', linewidth=1)
axes[2].set_title('By Organization Type\n(top 10 largest categories)', fontsize=10.5)
axes[2].set_xlabel('Default rate (%)')
axes[2].tick_params(axis='y', labelsize=8)

for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "11_categorical_patterns.png", dpi=160, facecolor='white')
print("Saved:", OUTPUT_DIR / "11_categorical_patterns.png")
