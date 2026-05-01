import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = ["bi_lstm", "mamba2", "transformer"]
SIZES  = ["l"]
SEEDS  = [42, 1, 2, 3, 4]
SIZE_ORDER = {s: i for i, s in enumerate(SIZES)}

COLORS = {
    "bi_lstm":     "#1f77b4",
    "mamba2":      "#ff7f0e",
    "transformer": "#2ca02c",
}
DISPLAY_NAMES = {
    "bi_lstm":     "BiLSTM (RNN)",
    "mamba2":      "Mamba2 (SSM)",
    "transformer": "Transformer",
}


# ──────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate and visualize multi-seed training results from JSON files."
    )
    parser.add_argument(
        "--results_dir", type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Directory containing {model}_{size}_seed{seed}.json files",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default=str(PROJECT_ROOT / "results" / "analysis"),
        help="Directory for output plots and CSV (created if absent)",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for size in SIZES:
            for seed in SEEDS:
                path = results_dir / f"{model}_{size}_seed{seed}.json"
                nan_row = {
                    "model": model, "size": size, "seed": seed,
                    "best_val_mae": np.nan, "val_r2": np.nan,
                    "peak_gpu_memory_mb": np.nan,
                    "total_time_seconds": np.nan, "best_epoch": np.nan,
                }
                if not path.exists():
                    warnings.warn(f"Missing result file: {path}")
                    rows.append(nan_row)
                    continue
                try:
                    with open(path) as f:
                        data = json.load(f)
                except Exception as e:
                    warnings.warn(f"Failed to parse {path}: {e}")
                    rows.append(nan_row)
                    continue

                epoch_list = data.get("epochs", [])
                if epoch_list:
                    best_idx = min(
                        range(len(epoch_list)),
                        key=lambda i: epoch_list[i].get("val_mae", float("inf")),
                    )
                    best_epoch_number = epoch_list[best_idx].get("epoch", np.nan)
                    best_val_r2 = epoch_list[best_idx].get("val_r2", np.nan)
                else:
                    best_epoch_number = np.nan
                    best_val_r2 = np.nan

                rows.append({
                    "model": model,
                    "size": size,
                    "seed": seed,
                    "best_val_mae": data.get("best_val_mae", np.nan),
                    "val_r2": best_val_r2,
                    "peak_gpu_memory_mb": data.get("peak_gpu_memory_mb", np.nan),
                    "total_time_seconds": data.get("total_time_seconds", np.nan),
                    "best_epoch": best_epoch_number,
                })

    return pd.DataFrame(rows)


def load_epoch_data(results_dir: Path) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for size in SIZES:
            for seed in SEEDS:
                path = results_dir / f"{model}_{size}_seed{seed}.json"
                if not path.exists():
                    continue
                try:
                    with open(path) as f:
                        data = json.load(f)
                except Exception:
                    continue
                for ep in data.get("epochs", []):
                    rows.append({
                        "model": model,
                        "size": size,
                        "seed": seed,
                        "epoch": ep.get("epoch"),
                        "train_loss": ep.get("train_loss"),
                        "val_loss": ep.get("val_loss"),
                        "val_mae": ep.get("val_mae"),
                        "val_r2": ep.get("val_r2"),
                        "epoch_time_seconds": ep.get("epoch_time_seconds"),
                    })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["best_val_mae", "val_r2", "peak_gpu_memory_mb", "total_time_seconds", "best_epoch"]
    records = []
    for model in MODELS:
        for size in SIZES:
            sub = df[(df["model"] == model) & (df["size"] == size)]
            row = {"model": model, "size": size}
            for m in metrics:
                vals = sub[m].dropna().values
                row[f"mean_{m}"] = np.nanmean(vals) if len(vals) > 0 else np.nan
                row[f"std_{m}"]  = np.nanstd(vals)  if len(vals) > 0 else np.nan
            row["n_seeds"] = int(sub["best_val_mae"].notna().sum())
            records.append(row)
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Table Print & Save
# ──────────────────────────────────────────────────────────────────────────────

def print_and_save_table(agg: pd.DataFrame, raw: pd.DataFrame, output_dir: Path) -> None:
    print("\n" + "=" * 85)
    print("AGGREGATED RESULTS (mean ± std over seeds)")
    print("=" * 85)
    header = (f"{'Model':<18} {'Size':<5} {'n':>3}  {'ValMAE':>12}  {'ValR2':>10}"
              f"  {'GPU(MB)':>10}  {'Time(s)':>12}  {'BestEpoch':>10}")
    print(header)
    print("-" * 85)
    for _, r in agg.iterrows():
        def fmt(m, s, w=10):
            if np.isnan(m):
                return f"{'N/A':>{w}}"
            return f"{m:.3f}±{s:.3f}".rjust(w)

        print(
            f"{DISPLAY_NAMES[r['model']]:<18} {r['size']:<5} {r['n_seeds']:>3}  "
            f"{fmt(r['mean_best_val_mae'], r['std_best_val_mae'], 12)}  "
            f"{fmt(r['mean_val_r2'], r['std_val_r2'])}  "
            f"{fmt(r['mean_peak_gpu_memory_mb'], r['std_peak_gpu_memory_mb'])}  "
            f"{fmt(r['mean_total_time_seconds'], r['std_total_time_seconds'], 12)}  "
            f"{fmt(r['mean_best_epoch'], r['std_best_epoch'])}"
        )
    print("=" * 85 + "\n")

    csv_cols = ["model", "size", "seed", "best_val_mae", "val_r2",
                "peak_gpu_memory_mb", "total_time_seconds", "best_epoch"]
    raw[csv_cols].to_csv(output_dir / "summary_table.csv", index=False)
    print(f"summary_table.csv saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 — Learning Curves (Val MAE & Train Loss per Epoch)
# ──────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(epoch_df: pd.DataFrame, output_dir: Path) -> None:
    size = "l"
    fig, axes = plt.subplots(1, len(MODELS), figsize=(18, 5), sharey=False)
    fig.suptitle(
        "Learning Curves — Mean ± Std over 5 Seeds (Size L)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for ax, model in zip(axes, MODELS):
        sub = epoch_df[(epoch_df["model"] == model) & (epoch_df["size"] == size)]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        color = COLORS[model]
        epochs = sorted(sub["epoch"].unique())
        grouped = sub.groupby("epoch")

        for metric, label, ls, alpha_line in [
            ("train_loss", "Train Loss",  "-",  1.0),
            ("val_mae",    "Val MAE",     "--", 0.85),
        ]:
            means = np.array([grouped.get_group(e)[metric].mean() for e in epochs])
            stds  = np.array([grouped.get_group(e)[metric].std()  for e in epochs])
            ax.plot(epochs, means, ls=ls, color=color, linewidth=2.2,
                    label=label, alpha=alpha_line)
            ax.fill_between(epochs, means - stds, means + stds,
                            alpha=0.18, color=color)

        ax.set_title(DISPLAY_NAMES[model], fontweight="bold", color=color)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss / MAE")
        ax.legend()
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))

    plt.tight_layout()
    path = output_dir / "learning_curves.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2 — Val MAE Comparison (with seed scatter)
# ──────────────────────────────────────────────────────────────────────────────

def plot_val_mae_comparison(raw: pd.DataFrame, agg: pd.DataFrame, output_dir: Path) -> None:
    size = "l"
    models_present = [m for m in MODELS
                      if not agg[(agg["model"] == m) & (agg["size"] == size)]["mean_best_val_mae"].isna().all()]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models_present))
    width = 0.45

    for i, model in enumerate(models_present):
        row = agg[(agg["model"] == model) & (agg["size"] == size)].iloc[0]
        mean_v = row["mean_best_val_mae"]
        std_v  = row["std_best_val_mae"]
        color  = COLORS[model]

        ax.bar(i, mean_v, width, yerr=std_v, color=color, edgecolor="black",
               alpha=0.80, capsize=6, label=DISPLAY_NAMES[model], zorder=2)

        # Individual seed dots
        seed_vals = raw[(raw["model"] == model) & (raw["size"] == size)]["best_val_mae"].dropna()
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, size=len(seed_vals))
        ax.scatter(i + jitter, seed_vals, color="black", s=30, zorder=3, alpha=0.7)

    # Annotate % reduction vs worst
    worst = max(
        agg[(agg["model"] == m) & (agg["size"] == size)]["mean_best_val_mae"].values[0]
        for m in models_present
        if not np.isnan(agg[(agg["model"] == m) & (agg["size"] == size)]["mean_best_val_mae"].values[0])
    )
    for i, model in enumerate(models_present):
        mean_v = agg[(agg["model"] == model) & (agg["size"] == size)]["mean_best_val_mae"].values[0]
        std_v  = agg[(agg["model"] == model) & (agg["size"] == size)]["std_best_val_mae"].values[0]
        pct = (worst - mean_v) / worst * 100
        label = "ref" if abs(pct) < 0.5 else f"−{pct:.1f}%"
        ax.text(i, mean_v + std_v + worst * 0.015, label,
                ha="center", fontsize=11, fontweight="bold", color=COLORS[model])

    ax.set_title("Best Validation MAE by Model — Size L\n(mean ± std, dots = individual seeds)",
                 fontweight="bold")
    ax.set_ylabel("Best Val MAE")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in models_present], fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    path = output_dir / "val_mae_comparison.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3 — Clock Time Reduction
# ──────────────────────────────────────────────────────────────────────────────

def plot_clock_time_reduction(agg: pd.DataFrame, epoch_df: pd.DataFrame, output_dir: Path) -> None:
    size = "l"
    models_present = [m for m in MODELS
                      if not agg[(agg["model"] == m) & (agg["size"] == size)]["mean_total_time_seconds"].isna().all()]

    total_means, total_stds, ep_means, ep_stds = [], [], [], []
    for model in models_present:
        row = agg[(agg["model"] == model) & (agg["size"] == size)].iloc[0]
        total_means.append(row["mean_total_time_seconds"])
        total_stds.append(row["std_total_time_seconds"])
        sub = epoch_df[(epoch_df["model"] == model) & (epoch_df["size"] == size)]
        ep_means.append(sub["epoch_time_seconds"].mean())
        ep_stds.append(sub["epoch_time_seconds"].std())

    slowest_total = max(total_means)
    slowest_ep    = max(ep_means)
    colors = [COLORS[m] for m in models_present]
    x = np.arange(len(models_present))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Wall-Clock Time Reduction vs. Slowest Baseline (Size L, mean ± std over 5 seeds)",
        fontsize=13, fontweight="bold",
    )

    # Left: per-epoch time
    ax = axes[0]
    ax.bar(x, ep_means, yerr=ep_stds, color=colors, edgecolor="black",
           alpha=0.85, capsize=6, width=0.5, zorder=2)
    for i, (m, ep, es) in enumerate(zip(models_present, ep_means, ep_stds)):
        pct = (slowest_ep - ep) / slowest_ep * 100
        lbl = "ref" if abs(pct) < 0.5 else f"−{pct:.0f}%"
        ax.text(i, ep + es + slowest_ep * 0.015, lbl,
                ha="center", fontsize=12, fontweight="bold", color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in models_present])
    ax.set_ylabel("Mean Epoch Time (s)")
    ax.set_title("Per-Epoch Wall-Clock Time", fontweight="bold")

    # Right: total training time (in hours)
    ax = axes[1]
    total_h  = [t / 3600 for t in total_means]
    total_hs = [t / 3600 for t in total_stds]
    ax.bar(x, total_h, yerr=total_hs, color=colors, edgecolor="black",
           alpha=0.85, capsize=6, width=0.5, zorder=2)
    for i, (m, th, ths, tt, ts) in enumerate(zip(models_present, total_h, total_hs,
                                                   total_means, total_stds)):
        pct = (slowest_total - tt) / slowest_total * 100
        lbl = "ref" if abs(pct) < 0.5 else f"−{pct:.0f}%"
        ax.text(i, th + ths + (slowest_total / 3600) * 0.015, lbl,
                ha="center", fontsize=12, fontweight="bold", color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in models_present])
    ax.set_ylabel("Total Training Time (hours)")
    ax.set_title("Total Training Time", fontweight="bold")

    plt.tight_layout()
    path = output_dir / "clock_time_reduction.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4 — Epoch Time Curve (time per epoch over training)
# ──────────────────────────────────────────────────────────────────────────────

def plot_epoch_time_curve(epoch_df: pd.DataFrame, output_dir: Path) -> None:
    size = "l"
    fig, ax = plt.subplots(figsize=(11, 5))

    for model in MODELS:
        sub = epoch_df[(epoch_df["model"] == model) & (epoch_df["size"] == size)]
        if sub.empty:
            continue
        color  = COLORS[model]
        epochs = sorted(sub["epoch"].unique())
        grouped = sub.groupby("epoch")
        means = np.array([grouped.get_group(e)["epoch_time_seconds"].mean() for e in epochs])
        stds  = np.array([grouped.get_group(e)["epoch_time_seconds"].std()  for e in epochs])
        ax.plot(epochs, means, color=color, linewidth=2, label=DISPLAY_NAMES[model])
        ax.fill_between(epochs, means - stds, means + stds, alpha=0.18, color=color)

    ax.set_title("Epoch Wall-Clock Time over Training — Size L\n(mean ± std over 5 seeds)",
                 fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epoch Time (s)")
    ax.legend()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))
    plt.tight_layout()
    path = output_dir / "epoch_time_curve.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 5 — GPU Memory
# ──────────────────────────────────────────────────────────────────────────────

def plot_gpu_memory(raw: pd.DataFrame, agg: pd.DataFrame, output_dir: Path) -> None:
    if agg["mean_peak_gpu_memory_mb"].isna().all():
        print("WARNING: No GPU memory data. Skipping gpu_memory.png.")
        return

    size = "l"
    models_present = [m for m in MODELS
                      if not agg[(agg["model"] == m) & (agg["size"] == size)]["mean_peak_gpu_memory_mb"].isna().all()]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models_present))

    for i, model in enumerate(models_present):
        row = agg[(agg["model"] == model) & (agg["size"] == size)].iloc[0]
        mean_v = row["mean_peak_gpu_memory_mb"]
        std_v  = row["std_peak_gpu_memory_mb"]
        ax.bar(i, mean_v / 1024, 0.5, yerr=std_v / 1024,
               color=COLORS[model], edgecolor="black", alpha=0.80, capsize=6,
               label=DISPLAY_NAMES[model], zorder=2)

        seed_vals = raw[(raw["model"] == model) & (raw["size"] == size)]["peak_gpu_memory_mb"].dropna()
        jitter = np.random.default_rng(1).uniform(-0.08, 0.08, size=len(seed_vals))
        ax.scatter(i + jitter, seed_vals / 1024, color="black", s=30, zorder=3, alpha=0.7)

    ax.set_title("Peak GPU Memory by Model — Size L\n(mean ± std, dots = individual seeds)",
                 fontweight="bold")
    ax.set_ylabel("Peak GPU Memory (GB)")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in models_present], fontsize=11)
    ax.legend()
    plt.tight_layout()
    path = output_dir / "gpu_memory.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 6 — Val R² Comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_val_r2(raw: pd.DataFrame, agg: pd.DataFrame, output_dir: Path) -> None:
    size = "l"
    models_present = [m for m in MODELS
                      if not agg[(agg["model"] == m) & (agg["size"] == size)]["mean_val_r2"].isna().all()]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models_present))

    for i, model in enumerate(models_present):
        row = agg[(agg["model"] == model) & (agg["size"] == size)].iloc[0]
        mean_v = row["mean_val_r2"]
        std_v  = row["std_val_r2"]
        ax.bar(i, mean_v, 0.45, yerr=std_v,
               color=COLORS[model], edgecolor="black", alpha=0.80, capsize=6,
               label=DISPLAY_NAMES[model], zorder=2)

        seed_vals = raw[(raw["model"] == model) & (raw["size"] == size)]["val_r2"].dropna()
        jitter = np.random.default_rng(2).uniform(-0.08, 0.08, size=len(seed_vals))
        ax.scatter(i + jitter, seed_vals, color="black", s=30, zorder=3, alpha=0.7)

    ax.set_title("Best Validation R² by Model — Size L\n(mean ± std, dots = individual seeds)",
                 fontweight="bold")
    ax.set_ylabel("Val R²")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in models_present], fontsize=11)
    ax.legend()
    plt.tight_layout()
    path = output_dir / "val_r2_comparison.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df   = load_results(results_dir)
    epoch_df = load_epoch_data(results_dir)

    n_found = int(raw_df["best_val_mae"].notna().sum())
    total   = len(MODELS) * len(SIZES) * len(SEEDS)
    print(f"Loaded {n_found} / {total} result files.")

    agg_df = aggregate(raw_df)
    print_and_save_table(agg_df, raw_df, output_dir)

    plot_learning_curves(epoch_df, output_dir)
    plot_val_mae_comparison(raw_df, agg_df, output_dir)
    plot_clock_time_reduction(agg_df, epoch_df, output_dir)
    plot_epoch_time_curve(epoch_df, output_dir)
    plot_gpu_memory(raw_df, agg_df, output_dir)
    plot_val_r2(raw_df, agg_df, output_dir)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
