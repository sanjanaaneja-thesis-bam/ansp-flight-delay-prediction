#Generate Data and Methods figures: delay distribution and daily departure delay.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT = Path(__file__).parent
FIGS_DIR = ROOT / "thesis_outputs" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading unified analytical dataset")
df = pd.read_csv(ROOT / "input_data_unified_60.csv")
df = df.rename(columns={"FAA_class": "hub_category"})
df["hub_category"] = pd.Categorical(
    df["hub_category"], categories=["Large", "Medium", "Small"], ordered=True
)
df["FL_DATE"] = pd.to_datetime(df["FL_DATE"])
print(f"Shape: {df.shape}")

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

HUB_COLORS = {
    "Large":  "#1f4e79",
    "Medium": "#d97706",
    "Small":  "#15803d",
}

# Figure 7 - Delay distribution (KDE) by hub
print("\n[Figure 7] Delay distribution by hub")
XMIN, XMAX = -30, 150
fig, ax = plt.subplots(figsize=(7, 4))
for hub in ["Large", "Medium", "Small"]:
    vals = df.loc[df["hub_category"] == hub, "DEP_DELAY"].dropna().values
    vals_trunc = vals[(vals >= XMIN) & (vals <= XMAX)]
    if len(vals_trunc) > 200000:
        rng = np.random.default_rng(42)
        vals_trunc = rng.choice(vals_trunc, size=200000, replace=False)
    kde = gaussian_kde(vals_trunc, bw_method=0.25)
    xs = np.linspace(XMIN, XMAX, 400)
    ys = kde(xs)
    ax.fill_between(xs, ys, alpha=0.10, color="#888888")
    ax.plot(xs, ys, color=HUB_COLORS[hub], linewidth=2.0, label=hub)

ax.axvline(15, color="#c0392b", linestyle="--", linewidth=1.2)
ax.text(15.8, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.01,
        " 15 min threshold", color="#c0392b", fontsize=9, va="top")
ax.set_xlabel("Departure delay (minutes)")
ax.set_ylabel("Density")
ax.set_xlim(XMIN, XMAX)
ax.grid(axis="y", alpha=0.4)
ax.grid(axis="x", alpha=0.2)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
leg = ax.legend(
    title="Hub category",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    frameon=True,
    fancybox=False,
    edgecolor="#333333",
    handlelength=2.5,
    handletextpad=0.8,
    columnspacing=2.0,
    borderpad=0.8,
)
leg.get_frame().set_linewidth(1.5)

f7_png = FIGS_DIR / "delay_distribution_by_hub.png"
f7_pdf = FIGS_DIR / "delay_distribution_by_hub.pdf"
fig.savefig(f7_png)
fig.savefig(f7_pdf)
plt.close(fig)
print(f"Saved: {f7_png}")

# Figure 7 - Daily mean departure delay over time
print("\n[Figure 7] Daily mean departure delay")
daily = (
    df.groupby(["FL_DATE", "hub_category"], observed=True)["DEP_DELAY"]
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 5))
for hub in ["Large", "Medium", "Small"]:
    sub = daily[daily["hub_category"] == hub].sort_values("FL_DATE")
    ax.plot(sub["FL_DATE"], sub["DEP_DELAY"],
            color=HUB_COLORS[hub], linewidth=1.6,
            marker="o", markersize=3, label=hub)

ax.set_xlabel("Date", labelpad=8)
ax.set_ylabel("Mean departure delay (minutes)")
ax.grid(axis="y", alpha=0.4)
ax.grid(axis="x", visible=False)
for lbl in ax.get_xticklabels():
    lbl.set_rotation(30)
    lbl.set_ha("right")
fig.subplots_adjust(bottom=0.30)
leg = ax.legend(
    title="Hub category",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.32),
    ncol=3,
    frameon=True,
    fancybox=False,
    edgecolor="#333333",
    handlelength=2.5,
    handletextpad=0.8,
    columnspacing=2.0,
    borderpad=0.8,
)
leg.get_frame().set_linewidth(1.5)

f8_png = FIGS_DIR / "daily_delay_timeseries.png"
f8_pdf = FIGS_DIR / "daily_delay_timeseries.pdf"
fig.savefig(f8_png, bbox_inches="tight")
fig.savefig(f8_pdf, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {f8_png}")

print("\n[OK] data_and_methods_outputs complete.")
