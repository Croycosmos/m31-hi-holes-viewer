from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("figures/final/bb86_vs_new247_2dcg")
OUT.mkdir(parents=True, exist_ok=True)

BB86_PATH = Path("data/holes_catalog_streamlit.csv")
NEW_POP_PATH = Path("figures/intermediate/RefineV9Candidates2DCG_v11_merged_population/v11_population_merged247_derived.csv")

bb86 = pd.read_csv(BB86_PATH)
new = pd.read_csv(NEW_POP_PATH)

def first_numeric(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                return s
    if required:
        raise RuntimeError(f"None of these columns found with numeric values: {candidates}")
    return pd.Series(np.nan, index=df.index)

# BB86: use 2DCG/refit quantities, not the original BB86 table values.
bb86_maj = first_numeric(bb86, ["cg2d__Maj_growth_pc", "joint__Maj_best_pc"])
bb86_min = first_numeric(bb86, ["cg2d__Min_growth_pc", "joint__Min_best_pc"])
bb86_vel = first_numeric(bb86, ["cg2d__hrv_kms", "joint__HRV_used_kms", "HRV"])
bb86_pa  = first_numeric(bb86, ["cg2d__PA_geometry_deg", "joint__PA_astro_best_deg", "PA"])

# New candidates: use the merged 2DCG refit population values.
new_maj = first_numeric(new, ["final_Maj_pc", "Maj_pc", "initial_Maj_pc"])
new_min = first_numeric(new, ["final_Min_pc", "Min_pc", "initial_Min_pc"])
new_vel = first_numeric(new, ["final_v_center_kms", "trial_v_center_kms", "initial_v_center_kms"])
new_pa  = first_numeric(new, ["final_PA_astro_deg", "PA_astro_deg", "initial_PA_astro_deg"])

bb86_eq = np.sqrt(bb86_maj * bb86_min)
new_eq = np.sqrt(new_maj * new_min)

bb86_q = bb86_min / bb86_maj
new_q = new_min / new_maj

datasets = [
    ("Major diameter [pc]", bb86_maj, new_maj, 40),
    ("Minor diameter [pc]", bb86_min, new_min, 40),
    ("Equivalent diameter [pc]", bb86_eq, new_eq, 40),
    ("Velocity [km/s]", bb86_vel, new_vel, 40),
    ("PA [deg]", bb86_pa, new_pa, 36),
    ("Axis ratio b/a", bb86_q, new_q, 30),
]

BB86_COLOR = "#4C78A8"
NEW_COLOR = "#F58518"

fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
axes = axes.ravel()

summary_rows = []

for ax, (label, braw, nraw, nbins) in zip(axes, datasets):
    b = pd.to_numeric(braw, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    n = pd.to_numeric(nraw, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)

    if label == "Axis ratio b/a":
        b = b[(b > 0) & (b <= 1.5)]
        n = n[(n > 0) & (n <= 1.5)]
    elif "diameter" in label.lower():
        b = b[b > 0]
        n = n[n > 0]

    allv = np.concatenate([b, n])
    allv = allv[np.isfinite(allv)]

    if allv.size == 0:
        ax.set_title(label)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        continue

    vmin = np.nanmin(allv)
    vmax = np.nanmax(allv)

    if label == "PA [deg]":
        vmin, vmax = -90.0, 90.0
        bins = np.linspace(vmin, vmax, nbins + 1)
    elif label == "Axis ratio b/a":
        vmin, vmax = 0.0, 1.1
        bins = np.linspace(vmin, vmax, nbins + 1)
    else:
        pad = 0.03 * (vmax - vmin) if vmax > vmin else 1.0
        bins = np.linspace(vmin - pad, vmax + pad, nbins + 1)

    # Same bin edges for both populations.
    ax.hist(b, bins=bins, histtype="stepfilled", alpha=0.22, color=BB86_COLOR, label=f"BB86 2DCG (N={len(b)})")
    ax.hist(b, bins=bins, histtype="step", linewidth=1.8, color=BB86_COLOR)
    ax.hist(n, bins=bins, histtype="step", linewidth=2.4, color=NEW_COLOR, label=f"New candidates 2DCG (N={len(n)})")

    ax.axvline(np.nanmedian(b), color=BB86_COLOR, linestyle="--", linewidth=1.2)
    ax.axvline(np.nanmedian(n), color=NEW_COLOR, linestyle="--", linewidth=1.2)

    ax.set_title(label)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    summary_rows.append({
        "quantity": label,
        "bb86_n": len(b),
        "new_n": len(n),
        "bb86_median": np.nanmedian(b),
        "new_median": np.nanmedian(n),
        "shared_bin_min": bins[0],
        "shared_bin_max": bins[-1],
        "shared_bin_width": bins[1] - bins[0],
        "n_bins": len(bins) - 1,
    })

fig.suptitle("BB86 2DCG refit vs new candidate 2DCG refit", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])

png = OUT / "bb86_2dcg_vs_new247_2dcg_histograms_samebins_orange.png"
pdf = OUT / "bb86_2dcg_vs_new247_2dcg_histograms_samebins_orange.pdf"
csv = OUT / "bb86_2dcg_vs_new247_2dcg_summary_samebins_orange.csv"

fig.savefig(png, dpi=180)
fig.savefig(pdf)
pd.DataFrame(summary_rows).to_csv(csv, index=False)

print("[saved]", png)
print("[saved]", pdf)
print("[saved]", csv)
print("[BB86 rows used]", len(bb86))
print("[New candidate rows used]", len(new))
