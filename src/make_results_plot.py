"""
make_results_plot.py — grouped bar chart of the 2×2 results for the README.
Produces results.png: accuracy by model (base / FIB / FIB+USB) × domain (in / OOD).

    python make_results_plot.py     # writes results.png
"""

import matplotlib.pyplot as plt
import numpy as np

# accuracy (%) — [in-domain FIB test, out-of-domain USB test]
data = {
    "base":     [61.2, 55.3],
    "FIB":      [90.5, 75.8],
    "FIB+USB":  [85.4, 84.2],
}
domains = ["In-domain\n(FIB / news)", "Out-of-domain\n(USB / Wikipedia)"]
colors = ["#9aa0a6", "#4c78a8", "#59a14f"]   # base grey, FIB blue, FIB+USB green

x = np.arange(len(domains))
width = 0.26

fig, ax = plt.subplots(figsize=(7, 3.6))
for i, (label, vals) in enumerate(data.items()):
    bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=colors[i])
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy by model and domain", fontsize=11)
ax.set_xticks(x, domains)
ax.set_ylim(0, 105)
# legend to the RIGHT of the plot, boxes stacked vertically (ncol=1)
ax.legend(title="trained on", frameon=False, ncol=1,
          loc="center left", bbox_to_anchor=(1.02, 0.5))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig("results.png", dpi=150, bbox_inches="tight")
print("wrote results.png")
