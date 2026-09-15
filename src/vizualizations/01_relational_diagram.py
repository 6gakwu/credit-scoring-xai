"""
Figure 1 - how the raw tables combine into the final merged dataset.
This one's just illustrative, no data actually gets loaded here, just
boxes and arrows laid out by hand.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MplPath
from pathlib import Path

OUTPUT_DIR = Path(r"C:\Users\jeffo\Documents\HomeCredit_XAI_Project\outputs\figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'


def box(ax, x, y, w, h, text, facecolor, edgecolor="#333333", fontsize=10, fontweight='normal'):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.5,
                           edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, color="#222222")


def straight_arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.4):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=13,
                         linewidth=lw, color=color)
    ax.add_patch(a)


def elbow_arrow(ax, x1, y1, x_mid, y2, x2, color="#555555", lw=1.4):
    # goes across, then down/up, then across again into the target box
    verts = [(x1, y1), (x_mid, y1), (x_mid, y2), (x2, y2)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO]
    path = MplPath(verts, codes)
    patch = plt.matplotlib.patches.PathPatch(path, facecolor='none', edgecolor=color, linewidth=lw)
    ax.add_patch(patch)
    straight_arrow(ax, x_mid, y2, x2, y2, color=color, lw=lw)


fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

# raw tables, left column
box(ax, 0.3, 5.5, 2.6, 1.0, "application_train.csv\napplication_test.csv\n(main applicant table)", "#DCEAF7")
box(ax, 0.3, 3.9, 2.6, 0.9, "bureau.csv\n(credit bureau history)", "#FCEAD8")
box(ax, 0.3, 2.55, 2.6, 0.9, "previous_application.csv\n(prior Home Credit apps)", "#E4F1E1")
box(ax, 0.3, 1.2, 2.6, 0.9, "installments_payments.csv\n(repayment records)", "#F3E3F0")

# aggregated versions, middle column
box(ax, 3.9, 3.9, 2.4, 0.9, "bureau_processed\n(1 row per applicant)", "#FCEAD8")
box(ax, 3.9, 2.55, 2.4, 0.9, "previous_applications_processed\n(1 row per applicant)", "#E4F1E1")
box(ax, 3.9, 1.2, 2.4, 0.9, "installments_processed\n(1 row per applicant)", "#F3E3F0")

straight_arrow(ax, 2.9, 4.35, 3.9, 4.35)
straight_arrow(ax, 2.9, 3.0, 3.9, 3.0)
straight_arrow(ax, 2.9, 1.65, 3.9, 1.65)
ax.text(3.4, 4.62, "aggregate by\nSK_ID_CURR", fontsize=7, ha='center', color="#666666")
ax.text(3.4, 3.27, "aggregate by\nSK_ID_CURR", fontsize=7, ha='center', color="#666666")
ax.text(3.4, 1.92, "aggregate by\nSK_ID_CURR", fontsize=7, ha='center', color="#666666")

# final merged table, right column
box(ax, 7.4, 2.7, 2.4, 2.3, "train_merged.csv\ntest_merged.csv\n\n(one row per\napplicant, full\nfeature set)", "#D9E8FC", fontweight='bold')

# elbow connectors into the merged box, keeps things from crossing over each other
elbow_arrow(ax, 2.9, 6.0, 6.8, 4.6, 7.4)
elbow_arrow(ax, 6.3, 4.35, 6.9, 4.2, 7.4)
elbow_arrow(ax, 6.3, 3.0, 6.9, 3.6, 7.4)
elbow_arrow(ax, 6.3, 1.65, 6.9, 3.0, 7.4)

ax.text(5.3, 6.55, "left join on SK_ID_CURR", fontsize=8.5, ha='center', color="#444444", style='italic')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_relational_diagram.png", dpi=180, bbox_inches='tight', facecolor='white')
print("Saved:", OUTPUT_DIR / "01_relational_diagram.png")
