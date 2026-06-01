"""Re-render Figure 1 (per-modality CE drop) locally from cached numbers.
No model / GPU needed -- reads report_figures/fig1_ce_drop.json.

Fix: value labels no longer collide with bars or the legend.
  (a) labels printed INSIDE each model bar (rotated, light) so nothing floats
      into the legend; legend pinned top-left in clear space.
  (b) drop labels above bars with a widened headroom.
Clean greyscale + Computer-Modern serif (LaTeX look without a TeX install).
"""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
J = json.load(open(HERE / "report_figures" / "fig1_ce_drop.json"))
ce, baseline, drop = J["ce"], J["baseline"], J["drop"]

# style: CM serif mathtext, greyscale, minimal chrome
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm", "mathtext.rm": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
})
GREY = {"baseline": "#d9d9d9", "bar": "#3a3a3a", "edge": "#000000"}

LBL = {"tok_rgb@196": "RGB", "tok_audio@512": "Audio", "tok_depth@196": "Depth",
       "tok_normal@196": "Normal", "scene_desc": "Caption"}
mods = list(ce.keys())
order = sorted(mods, key=lambda m: -drop[m])
labels = [LBL[m] for m in order]
cevals = [ce[m] for m in order]
bvals  = [baseline[m] for m in order]
drops  = [drop[m] for m in order]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.2))
x = np.arange(len(order)); w = 0.4

# ---- panel (a): CE vs chance ----
axA.bar(x - w/2, bvals, w, color=GREY["baseline"], edgecolor=GREY["edge"],
        linewidth=0.8, label=r"chance $\ln V$")
axA.bar(x + w/2, cevals, w, color=GREY["bar"], edgecolor=GREY["edge"],
        linewidth=0.8, label="model")
axA.set_xticks(x); axA.set_xticklabels(labels, rotation=20, ha="right")
axA.set_ylabel("cross-entropy (nats)")
axA.set_ylim(0, max(bvals) * 1.08)
# labels INSIDE the model bar (rotated, white) when tall enough, else just above it
for xi, c in zip(x, cevals):
    if c > 1.2:
        axA.text(xi + w/2, c - 0.25, f"{c:.2f}", ha="center", va="top",
                 fontsize=7.5, color="white", rotation=90)
    else:
        axA.text(xi + w/2, c + 0.12, f"{c:.2f}", ha="center", va="bottom",
                 fontsize=7.5, color="black")
# legend OUTSIDE the axes (above) -> can never overlap a bar; title under it
axA.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.14),
           ncol=2, columnspacing=1.4, handlelength=1.3)
axA.set_title("(a) test CE vs. chance", pad=26)

# ---- panel (b): information captured (CE drop) ----
bars = axB.bar(x, drops, 0.62, color=GREY["bar"], edgecolor=GREY["edge"], linewidth=0.8)
axB.set_xticks(x); axB.set_xticklabels(labels, rotation=20, ha="right")
axB.set_ylabel(r"CE drop  $\ln V - \mathrm{CE}$  (nats)")
axB.set_title("(b) information captured per modality")
axB.set_ylim(0, max(drops) * 1.16)
for b, d in zip(bars, drops):
    axB.text(b.get_x() + b.get_width()/2, d + max(drops)*0.015, f"{d:.2f}",
             ha="center", va="bottom", fontsize=8.5, fontweight="bold")

fig.tight_layout()
out = HERE / "report_figures"
fig.savefig(out / "fig1_ce_drop.png")
fig.savefig(out / "fig1_ce_drop.pdf")
plt.close(fig)
from PIL import Image
print("saved fig1_ce_drop.{png,pdf}  size:", Image.open(out / "fig1_ce_drop.png").size)
