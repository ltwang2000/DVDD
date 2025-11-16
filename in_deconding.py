# -*- coding: utf-8 -*-
# Combined figure with DISTINCT color families for BLEU (blue) and METEOR (orange).

import matplotlib.pyplot as plt
import numpy as np

datasets = ["Test2016", "Test2017", "MSCOCO"]

# data
bleu_cong = [40.33, 32.27, 28.53]
bleu_incg = [39.10, 31.80, 28.14]
meteor_cong = [65.39, 59.79, 52.99]
meteor_incg = [64.85, 58.98, 47.36]

x = np.arange(len(datasets))
w = 0.17

# color palette: BLEU = blues, METEOR = oranges
C_BLEU_CONG = "#1f77b4"   # deep blue
C_BLEU_INCG = "#6baed6"   # light blue
C_MET_CONG  = "#ff7f0e"   # deep orange
C_MET_INCG  = "#ffbb78"   # light orange

fig, ax1 = plt.subplots(figsize=(10, 5.2))

# BLEU (left axis)
b1 = ax1.bar(x - w*1.1, bleu_cong, width=w, label="BLEU (Congruent)",
             color=C_BLEU_CONG, edgecolor="black", linewidth=0.6)
b2 = ax1.bar(x - w*0.0, bleu_incg, width=w, label="BLEU (Incongruent)",
             color=C_BLEU_INCG, edgecolor="black", linewidth=0.6)
ax1.set_ylabel("BLEU"); ax1.set_ylim(24, 42)
ax1.set_xticks(x); ax1.set_xticklabels(datasets)
ax1.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)

# METEOR (right axis)
ax2 = ax1.twinx()
m1 = ax2.bar(x + w*1.1, meteor_cong, width=w, label="METEOR (Congruent)",
             color=C_MET_CONG, edgecolor="black", linewidth=0.6)
m2 = ax2.bar(x + w*2.2, meteor_incg, width=w, label="METEOR (Incongruent)",
             color=C_MET_INCG, edgecolor="black", linewidth=0.6)
ax2.set_ylabel("METEOR"); ax2.set_ylim(46, 67)

# value labels
ax1.bar_label(b1, labels=[f"{v:.2f}" for v in bleu_cong], padding=2, fontsize=9)
ax1.bar_label(b2, labels=[f"{v:.2f}" for v in bleu_incg], padding=2, fontsize=9)
ax2.bar_label(m1, labels=[f"{v:.2f}" for v in meteor_cong], padding=2, fontsize=9)
ax2.bar_label(m2, labels=[f"{v:.2f}" for v in meteor_incg], padding=2, fontsize=9)

# one legend below axes
handles = [b1, b2, m1, m2]
labels  = [h.get_label() for h in handles]
ax1.legend(handles, labels, ncol=2, loc="upper center",
           bbox_to_anchor=(0.5, -0.08), frameon=False)

fig.suptitle("Multi30K En–De: Congruent vs. Incongruent Decoding", y=0.99)
fig.tight_layout(); plt.subplots_adjust(bottom=0.18)
plt.savefig("multi30k_congruent_incongruent_combined_colors.png", dpi=300, bbox_inches="tight")
plt.show()
