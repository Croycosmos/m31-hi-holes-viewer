from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

APP = Path(".")

BB86_PATH = APP / "figures/final/bb86_2dcg_histograms/bb86_2dcg_joint_histogram_input_with_derived_columns.csv"
NEW_PATH = APP / "figures/intermediate/RefineV9Candidates2DCG_v11_merged_population/v11_population_merged247_derived.csv"

OUT_DIR = APP / "figures/final/bb86_vs_new247_2dcg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "bb86_2dcg_vs_new247_2dcg_histograms.png"
OUT_PDF = OUT_DIR / "bb86_2dcg_vs_new247_2dcg_histograms.pdf"
OUT_CSV = OUT_DIR / "bb86_2dcg_vs_new247_2dcg_summary.csv"


def first_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def num(df, names):
    col = first_col(df, names)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def normalise_pa_0_180(x):
    return np.mod(pd.to_numeric(x, errors="coerce"), 180.0)


def build_bb86(df):
    out = pd.DataFrame(index=df.index)

    # Strict: prefer explicit 2DCG columns when they exist.
    out["Maj_pc"] = num(df, ["cg2d_Maj_pc", "Maj_growth_pc", "Maj_pc"])
    out["Min_pc"] = num(df, ["cg2d_Min_pc", "Min_growth_pc", "Min_pc"])
    out["PA_deg"] = num(df, ["cg2d_PA_astro_deg", "PA_geometry_deg", "PA_astro_deg"])
    out["v_center_kms"] = num(df, ["cg2d_v_center_kms", "hrv_kms", "v_center_kms"])

    if "geometry_source" in df.columns:
        out["geometry_source"] = df["geometry_source"].astype(str)
        # Do not include rows that only fall back to the original BB86 table.
        keep = out["geometry_source"].str.contains("CumulativeGrowth2D|2DCG|joint", case=False, na=False)
        out = out.loc[keep].copy()
    else:
        out["geometry_source"] = "2DCG/refit"

    out["sample"] = "BB86 2DCG/refit"
    return clean_geometry(out)


def build_new247(df):
    out = pd.DataFrame(index=df.index)
    out["Maj_pc"] = num(df, ["final_Maj_pc", "Maj_pc"])
    out["Min_pc"] = num(df, ["final_Min_pc", "Min_pc"])
    out["PA_deg"] = num(df, ["final_PA_astro_deg", "PA_astro_deg"])
    out["v_center_kms"] = num(df, ["final_v_center_kms", "v_center_kms", "trial_v_center_kms"])
    out["geometry_source"] = "2DCG refit population"
    out["sample"] = "New candidates 2DCG"
    return clean_geometry(out)


def clean_geometry(df):
    out = df.copy()

    for col in ["Maj_pc", "Min_pc", "PA_deg", "v_center_kms"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    bad = (out["Maj_pc"] <= 0) | (out["Min_pc"] <= 0)
    out.loc[bad, ["Maj_pc", "Min_pc"]] = np.nan

    swap = out["Min_pc"] > out["Maj_pc"]
    maj = out.loc[swap, "Maj_pc"].copy()
    out.loc[swap, "Maj_pc"] = out.loc[swap, "Min_pc"]
    out.loc[swap, "Min_pc"] = maj

    out["geom_mean_pc"] = np.sqrt(out["Maj_pc"] * out["Min_pc"])
    out["axis_ratio"] = out["Min_pc"] / out["Maj_pc"]
    out["PA_mod_180_deg"] = normalise_pa_0_180(out["PA_deg"])

    return out


def nice_step(raw):
    if not np.isfinite(raw) or raw <= 0:
        return 1.0
    base = 10 ** np.floor(np.log10(raw))
    for m in [1, 2, 2.5, 5, 10]:
        step = m * base
        if raw <= step:
            return step
    return 10 * base


def shared_bins(x1, x2, nbins_target=18, hard_range=None):
    vals = np.concatenate([
        pd.to_numeric(pd.Series(x1), errors="coerce").dropna().to_numpy(float),
        pd.to_numeric(pd.Series(x2), errors="coerce").dropna().to_numpy(float),
    ])
    vals = vals[np.isfinite(vals)]

    if hard_range is not None:
        lo, hi = hard_range
    elif len(vals) == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = np.nanmin(vals)
        hi = np.nanmax(vals)

    if hi <= lo:
        hi = lo + 1.0

    step = nice_step((hi - lo) / nbins_target)
    lo = np.floor(lo / step) * step
    hi = np.ceil(hi / step) * step

    return np.arange(lo, hi + 0.5 * step, step)


def add_hist(ax, bb, new, col, xlabel, title, bins):
    x_bb = pd.to_numeric(bb[col], errors="coerce").dropna()
    x_new = pd.to_numeric(new[col], errors="coerce").dropna()

    ax.hist(
        x_bb,
        bins=bins,
        alpha=0.62,
        color="0.55",
        edgecolor="black",
        linewidth=0.45,
        label=f"BB86 2DCG/refit, N={len(x_bb)}",
    )
    ax.hist(
        x_new,
        bins=bins,
        alpha=0.55,
        color="darkorange",
        edgecolor="black",
        linewidth=0.45,
        label=f"New candidates 2DCG, N={len(x_new)}",
    )

    if len(x_bb):
        ax.axvline(np.nanmedian(x_bb), color="black", linestyle="--", linewidth=1.1)
    if len(x_new):
        ax.axvline(np.nanmedian(x_new), color="darkorange", linestyle="--", linewidth=1.1)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8)


bb86_raw = pd.read_csv(BB86_PATH)
new_raw = pd.read_csv(NEW_PATH)

bb86 = build_bb86(bb86_raw)
new = build_new247(new_raw)

panels = [
    ("Maj_pc", "Major axis [pc]", "Major axis"),
    ("Min_pc", "Minor axis [pc]", "Minor axis"),
    ("geom_mean_pc", "sqrt(Maj × Min) [pc]", "Mean geometric size"),
    ("axis_ratio", "Minor / major axis", "Axis ratio"),
    ("PA_mod_180_deg", "PA modulo 180° [deg]", "Position angle"),
    ("v_center_kms", "Central velocity [km/s]", "Central velocity"),
]

fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), dpi=180)
axes = axes.ravel()

for ax, (col, xlabel, title) in zip(axes, panels):
    if col == "axis_ratio":
        bins = np.linspace(0.0, 1.05, 22)
    elif col == "PA_mod_180_deg":
        bins = np.arange(0.0, 180.0 + 15.0, 15.0)
    elif col == "v_center_kms":
        bins = shared_bins(bb86[col], new[col], nbins_target=18)
    else:
        bins = shared_bins(bb86[col], new[col], nbins_target=18)

    add_hist(ax, bb86, new, col, xlabel, title, bins)

fig.suptitle("BB86 2DCG/refit holes vs 2DCG new candidates", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_PNG, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

rows = []
for sample_name, df in [("BB86 2DCG/refit", bb86), ("New candidates 2DCG", new)]:
    for col, _, _ in panels:
        x = pd.to_numeric(df[col], errors="coerce").dropna()
        rows.append({
            "sample": sample_name,
            "quantity": col,
            "N": len(x),
            "median": float(np.nanmedian(x)) if len(x) else np.nan,
            "mean": float(np.nanmean(x)) if len(x) else np.nan,
            "std": float(np.nanstd(x)) if len(x) else np.nan,
            "min": float(np.nanmin(x)) if len(x) else np.nan,
            "max": float(np.nanmax(x)) if len(x) else np.nan,
        })

pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

print("[saved]", OUT_PNG)
print("[saved]", OUT_PDF)
print("[saved]", OUT_CSV)
print("[BB86 rows used]", len(bb86))
print("[New candidate rows used]", len(new))
print("[BB86 geometry sources]")
print(bb86["geometry_source"].value_counts(dropna=False).to_string())
