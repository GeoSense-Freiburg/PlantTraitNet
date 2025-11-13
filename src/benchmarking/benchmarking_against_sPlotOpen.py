"""
benchmarking script
Author: Ayushi Sharma
Description: Compares model predictions with sPlotOpen trait maps.
"""

import os
import re
import argparse
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
from scipy.stats import linregress
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns

from utils.benchmark_conf import get_config
from utils.benchmark_utils import read_trait_map, global_grid_df, lat_weights

cfg = get_config()


# -------------------- METRICS --------------------
def weighted_pearsonr(x, y, w):
    mean_x, mean_y = np.average(x, weights=w), np.average(y, weights=w)
    cov_xy = np.sum(w * (x - mean_x) * (y - mean_y))
    return cov_xy / np.sqrt(np.sum(w * (x - mean_x) ** 2) * np.sum(w * (y - mean_y) ** 2))


def compute_weighted_metrics(df, latwts, pred_col, true_col, log_scale=False):
    """Compute weighted metrics between prediction and ground truth"""
    df = df.copy()
    if df.index.names != ["y", "x"]:
        df.index.set_names(["y", "x"], inplace=True)

    df["weight"] = df.index.get_level_values("y").map(latwts)
    df = df.dropna(subset=["weight", pred_col, true_col])

    y_true, y_pred, w = df[true_col].values, df[pred_col].values, df["weight"].values
    if log_scale:
        y_true, y_pred = np.log1p(y_true), np.log1p(y_pred)

    if len(y_true) == 0:
        return dict.fromkeys(["n", "mae", "nmae", "rmse", "nrmse", "r2", "slope", "pearson_r"], np.nan)

    err = y_pred - y_true
    mae = np.average(np.abs(err), weights=w)
    rmse = np.sqrt(np.average(err ** 2, weights=w))
    rng = np.quantile(y_true, 0.99) - np.quantile(y_true, 0.01)
    nmae, nrmse = mae / rng, rmse / rng

    ss_res, ss_tot = np.sum(w * err ** 2), np.sum(w * (y_true - np.average(y_true, weights=w)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    slope, _, *_ = linregress(y_true, y_pred)
    r = weighted_pearsonr(y_true, y_pred, w)
    return dict(n=len(df), mae=mae, nmae=nmae, rmse=rmse, nrmse=nrmse, r2=r2, slope=slope, pearson_r=r)


def plothexbin(df, true_col, pred_col, latwts, ax, label=None,
               scaleaxis=True, allmetrics=True, n_min=1, log_scale=False, title=''):
    """Plot weighted hexbin with regression metrics overlay"""
    if df.empty:
        print("⚠️ Empty DataFrame passed to plothexbin")
        return ax

    metrics = compute_weighted_metrics(df, latwts, pred_col, true_col, log_scale=log_scale)
    y_true, y_pred = df[true_col].values, df[pred_col].values
    if log_scale:
        y_true, y_pred = np.log1p(y_true), np.log1p(y_pred)

    ax.set_aspect('equal', adjustable='box')

    if scaleaxis:
        maxval, minval = max(y_true.max(), y_pred.max()), min(y_true.min(), y_pred.min())
        ax.set_xlim(minval, maxval)
        ax.set_ylim(minval, maxval)
        ax.plot([minval, maxval], [minval, maxval], color='red', lw=2)
        extent = [minval, maxval, minval, maxval]
    else:
        extent = None

    hb = ax.hexbin(y_true, y_pred, gridsize=60, cmap='plasma', mincnt=1, extent=extent,
                   norm=Normalize(vmin=0, vmax=1))
    counts = hb.get_array()
    norm_counts = counts / counts.max()
    hb.set_array(norm_counts)

    if label:
        ax.set_xlabel(f"Observed {label}")
        ax.set_ylabel(f"Predicted {label}")

    if allmetrics:
        ax.text(
            0.05, 0.95,
            f"R²: {metrics['r2']:.2f}\nnMAE: {metrics['nmae']:.2f}\nR: {metrics['pearson_r']:.2f}",
            transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
        )

    ax.set_title(title)
    return ax


# -------------------- PREPARE DATA --------------------
def prepare_data(val_path, normval_path, valmeta, traits):
    normval = pd.read_csv(normval_path)
    pred = pd.read_csv(val_path)
    pred = pd.concat([
        pred,
        normval[[c for c in normval.columns if c.endswith("_uncertainty") and c.split("_uncertainty")[0] in traits]],
        valmeta[['Longitude', 'Latitude']]
    ], axis=1)
    for t in traits:
        if t.lower() == "leafarea":
            pred[f"{t}_true"] *= 100
            pred[f"{t}_pred"] *= 100
    return pred


# -------------------- SPLOT BENCHMARKING --------------------
def splot_benchmarking(pred, cfg, traits, out_dir, n_min=20):
    results = []
    out_dir = Path(out_dir)

    for t in traits:
        splot_df = read_trait_map(t, "splot", band=1).to_dataframe(name=t).drop(columns=["band", "spatial_ref"]).dropna()

        grid_true = global_grid_df(pred, f"{t}_true", lon="Longitude", lat="Latitude", res=1, stats=["mean"])
        grid_pred = global_grid_df(pred, f"{t}_pred", lon="Longitude", lat="Latitude", res=1, stats=["mean"])

        # Join on grid index (y, x)
        merged = grid_true.join(grid_pred, lsuffix="_true", rsuffix="_pred", how="inner")
        merged = merged.join(splot_df, how="inner")

        merged.rename(columns={t: f"{t}_splot", "mean_true": f"{t}_true", "mean_pred": f"{t}_pred"}, inplace=True)
        merged = merged[[f"{t}_splot", f"{t}_true", f"{t}_pred"]].dropna()

        if merged.empty:
            continue

        # Save combined data
        combined_path = out_dir / "combined_data" / f"{t}_splot_vs_model_1deg_nmin{n_min}.parquet"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        merged.reset_index().to_parquet(combined_path)

        lat_wts = lat_weights(merged.index.get_level_values("y").unique().values, 1)

        # Plotting
        with sns.plotting_context("notebook"), sns.axes_style("ticks"):
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))

            plothexbin(merged, f"{t}_splot", f"{t}_true", lat_wts, ax[0],
                       label=t, title="sPlot vs TRY6 mean", log_scale=True)

            plothexbin(merged, f"{t}_splot", f"{t}_pred", lat_wts, ax[1],
                       label=t, title="sPlot vs Model", log_scale=True)

            fig_path = out_dir / "plots" / f"nmin{n_min}" / f"{t}_vs_splot_1deg_nmin{n_min}.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_path, dpi=300)
            plt.close(fig)

        # Compute metrics
        metrics = compute_weighted_metrics(merged, lat_wts, f"{t}_pred", f"{t}_splot", log_scale=True)
        metrics.update(trait=t)
        results.append(metrics)

    if results:
        pd.DataFrame(results).to_csv(out_dir / "splot_metrics.csv", index=False)
    return results


# -------------------- UTILITIES --------------------
def get_result_dirs(base_dir):
    dirs = sorted(glob(os.path.join(base_dir, "results_ckpt_epoch_*")))
    file_pairs = []
    for d in dirs:
        for val_path in glob(os.path.join(d, "ckpt_*.csv")):
            norm = os.path.join(d, f"normalized_{os.path.basename(val_path)}")
            if os.path.exists(norm):
                file_pairs.append((val_path, norm))
    return {int(re.search(r'epoch_(\d+)', p[0]).group(1)): {"val": p[0], "normval": p[1]} for p in file_pairs}


def run_benchmark(seed_dirs, valmeta, cfg, out_dir, traits, n_min):
    summary = []
    for seed, epochs in seed_dirs.items():
        for epoch, files in epochs.items():
            pred = prepare_data(files["val"], files["normval"], valmeta, traits)
            print(f"Running benchmarking for Seed {seed}, Epoch {epoch} — {len(pred)} samples")
            res = splot_benchmarking(pred, cfg, traits, out_dir, n_min)
            for r in res:
                r.update(seed=seed, epoch=epoch)
                summary.append(r)
    if summary:
        df = pd.DataFrame(summary)
        df.to_csv(Path(out_dir) / "epoch_wise_summary.csv", index=False)
        return df
    return pd.DataFrame()


# -------------------- MAIN --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run unified sPlot benchmarking for multiple traits and seeds")
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--valmeta_path", type=str, required=True)
    parser.add_argument("--n_min", type=int, default=20)
    parser.add_argument("--traits", nargs="+", default=["Height"])
    parser.add_argument("--seed_pattern", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    valmeta = pd.read_csv(args.valmeta_path)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed_pattern:
        seed_paths = sorted(glob(os.path.join(args.base_dir, args.seed_pattern)))
    else:
        seed_paths = [args.base_dir]

    all_seed_dirs = {}
    for i, seed_dir in enumerate(seed_paths):
        m = re.findall(r"(\d+)", os.path.basename(seed_dir))
        seed_id = int(m[0]) if m else i
        all_seed_dirs[seed_id] = get_result_dirs(seed_dir)

    print(f"Detected seeds: {list(all_seed_dirs.keys())}")
    run_benchmark(all_seed_dirs, valmeta, cfg, args.output_dir, args.traits, args.n_min)
    print(f"Benchmarking completed. Results saved in {args.output_dir}")
