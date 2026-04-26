import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = ["mamba2", "bi_lstm", "transformer"]
SIZES  = ["xs", "s", "m", "l"]
SEEDS  = [42, 1, 2, 3, 4]
SIZE_ORDER = {s: i for i, s in enumerate(SIZES)}

COLORS = {
    "mamba2":      "#ff7f0e",
    "bi_lstm":     "#1f77b4",
    "transformer": "#2ca02c",
}
DISPLAY_NAMES = {
    "mamba2":      "Mamba2 (SSM)",
    "bi_lstm":     "Bi-LSTM",
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
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS (mean ± std over seeds)")
    print("=" * 80)
    header = f"{'Model':<18} {'Size':<5} {'n':>3}  {'ValMAE':>10}  {'ValR2':>10}  {'GPU(MB)':>10}  {'Time(s)':>10}  {'BestEpoch':>10}"
    print(header)
    print("-" * 80)
    for _, r in agg.iterrows():
        def fmt(m, s, w=10):
            if np.isnan(m):
                return f"{'N/A':>{w}}"
            return f"{m:.3f}±{s:.3f}".rjust(w)

        print(
            f"{DISPLAY_NAMES[r['model']]:<18} {r['size']:<5} {r['n_seeds']:>3}  "
            f"{fmt(r['mean_best_val_mae'], r['std_best_val_mae'])}  "
            f"{fmt(r['mean_val_r2'], r['std_val_r2'])}  "
            f"{fmt(r['mean_peak_gpu_memory_mb'], r['std_peak_gpu_memory_mb'])}  "
            f"{fmt(r['mean_total_time_seconds'], r['std_total_time_seconds'])}  "
            f"{fmt(r['mean_best_epoch'], r['std_best_epoch'])}"
        )
    print("=" * 80 + "\n")

    csv_cols = ["model", "size", "seed", "best_val_mae", "val_r2",
                "peak_gpu_memory_mb", "total_time_seconds", "best_epoch"]
    raw[csv_cols].to_csv(output_dir / "summary_table.csv", index=False)
    print(f"summary_table.csv saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Shared Plot Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sorted_agg(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy()
    agg["_size_order"] = agg["size"].map(SIZE_ORDER)
    return agg.sort_values(["_size_order", "model"]).drop(columns="_size_order")


def _grouped_bar(agg: pd.DataFrame, mean_col: str, std_col: str,
                 title: str, ylabel: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    width = 0.25
    offsets = {m: (i - 1) * width for i, m in enumerate(MODELS)}
    x = np.arange(len(SIZES))

    for model in MODELS:
        sub = _sorted_agg(agg[agg["model"] == model])
        means = []
        stds  = []
        for size in SIZES:
            row = sub[sub["size"] == size]
            means.append(row[mean_col].values[0] if len(row) else np.nan)
            stds.append(row[std_col].values[0]   if len(row) else np.nan)

        means = np.array(means, dtype=float)
        stds  = np.array(stds,  dtype=float)
        valid = ~np.isnan(means)
        if not valid.any():
            continue

        ax.bar(
            x[valid] + offsets[model],
            means[valid],
            width,
            yerr=stds[valid],
            label=DISPLAY_NAMES[model],
            color=COLORS[model],
            edgecolor="black",
            alpha=0.85,
            capsize=4,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Model Size Tier", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(SIZES, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1 — Val MAE Comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_val_mae_comparison(agg: pd.DataFrame, output_dir: Path) -> None:
    _grouped_bar(
        agg,
        mean_col="mean_best_val_mae",
        std_col="std_best_val_mae",
        title="Validation MAE by Model and Size (mean ± std over 5 seeds)",
        ylabel="Mean Val MAE",
        output_path=output_dir / "val_mae_comparison.png",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2 — Scaling Curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_scaling_curve(agg: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(SIZES))

    for model in MODELS:
        sub = _sorted_agg(agg[agg["model"] == model])
        means = []
        stds  = []
        for size in SIZES:
            row = sub[sub["size"] == size]
            means.append(row["mean_best_val_mae"].values[0] if len(row) else np.nan)
            stds.append(row["std_best_val_mae"].values[0]   if len(row) else np.nan)

        means = np.array(means, dtype=float)
        stds  = np.array(stds,  dtype=float)
        valid = ~np.isnan(means)
        if valid.sum() < 2:
            warnings.warn(f"Skipping scaling curve for {model}: fewer than 2 valid size points.")
            continue

        ax.plot(x[valid], means[valid], marker="o", linewidth=2,
                label=DISPLAY_NAMES[model], color=COLORS[model])
        ax.fill_between(x[valid], means[valid] - stds[valid], means[valid] + stds[valid],
                        alpha=0.2, color=COLORS[model])

    ax.set_title("Val MAE Scaling Curve (mean ± std over 5 seeds)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model Size", fontsize=12)
    ax.set_ylabel("Mean Val MAE", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(SIZES, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    path = output_dir / "scaling_curve.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3 — Training Time
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_time(agg: pd.DataFrame, output_dir: Path) -> None:
    _grouped_bar(
        agg,
        mean_col="mean_total_time_seconds",
        std_col="std_total_time_seconds",
        title="Training Time by Model and Size (mean ± std over 5 seeds)",
        ylabel="Total Training Time (seconds)",
        output_path=output_dir / "training_time.png",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4 — GPU Memory
# ──────────────────────────────────────────────────────────────────────────────

def plot_gpu_memory(agg: pd.DataFrame, output_dir: Path) -> None:
    if agg["mean_peak_gpu_memory_mb"].isna().all():
        print("WARNING: No GPU memory data found in any result file. Skipping gpu_memory.png.")
        return
    _grouped_bar(
        agg,
        mean_col="mean_peak_gpu_memory_mb",
        std_col="std_peak_gpu_memory_mb",
        title="Peak GPU Memory by Model and Size (mean ± std over 5 seeds)",
        ylabel="Peak GPU Memory (MB)",
        output_path=output_dir / "gpu_memory.png",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Plot 5 — Convergence Epochs + Step Reduction %
# ──────────────────────────────────────────────────────────────────────────────

def plot_convergence_epochs(agg: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    width = 0.25
    offsets = {m: (i - 1) * width for i, m in enumerate(MODELS)}
    x = np.arange(len(SIZES))

    bar_info = {}  # (size_idx, model) -> (x_pos, bar_height, std)

    for model in MODELS:
        sub = _sorted_agg(agg[agg["model"] == model])
        means = []
        stds  = []
        for size in SIZES:
            row = sub[sub["size"] == size]
            means.append(row["mean_best_epoch"].values[0] if len(row) else np.nan)
            stds.append(row["std_best_epoch"].values[0]   if len(row) else np.nan)

        means = np.array(means, dtype=float)
        stds  = np.array(stds,  dtype=float)

        for si, size in enumerate(SIZES):
            if not np.isnan(means[si]):
                bar_info[(si, model)] = (x[si] + offsets[model], means[si], stds[si])

        valid = ~np.isnan(means)
        if not valid.any():
            continue

        ax.bar(
            x[valid] + offsets[model],
            means[valid],
            width,
            yerr=stds[valid],
            label=DISPLAY_NAMES[model],
            color=COLORS[model],
            edgecolor="black",
            alpha=0.85,
            capsize=4,
        )

    # Annotate step reduction % vs slowest model per size
    for si, size in enumerate(SIZES):
        size_means = {
            m: bar_info[(si, m)][1]
            for m in MODELS if (si, m) in bar_info
        }
        if len(size_means) < 2:
            continue
        slowest = max(size_means.values())
        for model, mean_val in size_means.items():
            if np.isnan(slowest) or slowest == 0:
                continue
            pct = (slowest - mean_val) / slowest * 100
            x_pos, bar_h, bar_std = bar_info[(si, model)]
            label = "ref" if abs(pct) < 0.5 else f"-{pct:.0f}%"
            ax.text(x_pos, bar_h + (bar_std if not np.isnan(bar_std) else 0) + 0.3,
                    label, ha="center", va="bottom", fontsize=7.5, color="black")

    ax.set_title("Convergence Speed — Epoch of Best Val MAE (mean ± std over 5 seeds)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Model Size Tier", fontsize=12)
    ax.set_ylabel("Mean Epoch of Best Val MAE", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(SIZES, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    path = output_dir / "convergence_epochs.png"
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

    raw_df = load_results(results_dir)
    n_found = int(raw_df["best_val_mae"].notna().sum())
    total   = len(MODELS) * len(SIZES) * len(SEEDS)
    print(f"Loaded {n_found} / {total} result files.")

    agg_df = aggregate(raw_df)

    print_and_save_table(agg_df, raw_df, output_dir)

    plot_val_mae_comparison(agg_df, output_dir)
    plot_scaling_curve(agg_df, output_dir)
    plot_training_time(agg_df, output_dir)
    plot_gpu_memory(agg_df, output_dir)
    plot_convergence_epochs(agg_df, output_dir)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
