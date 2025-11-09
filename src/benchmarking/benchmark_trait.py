#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified benchmarking script for single- and multi-trait evaluation
Author: Ayushi Sharma
Description: Compares model predictions with sPlot data and computes weighted metrics.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import rioxarray as riox
from scipy.stats import linregress
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

# project setup
os.environ['PROJECT_ROOT'] = '../cit-sci-traits'
sys.path.append(os.getenv('PROJECT_ROOT'))

from src.conf.conf import get_config
from src.utils.dataset_utils import read_trait_map
from src.utils.df_utils import global_grid_df
from src.utils.spatial_utils import lat_weights
from box import ConfigBox

cfg = get_config()


# -------------------- ARGUMENT PARSER --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run sPlot benchmarking for multiple traits and seeds")
    parser.add_argument("--base_dir_seed0", type=str, required=True)
    parser.add_argument("--base_dir_seed100", type=str, required=True)
    parser.add_argument("--base_dir_seed200", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--traits", nargs="+", default=["LeafN"], help="List of traits to benchmark")
    return parser.parse_args()


# -------------------- PREPARE DATA --------------------
def prepare_data(val_path, normval_path, valmeta, traits):
    """Load, normalize, and filter predictions."""
    normval = pd.read_csv(normval_path)
    pred = pd.read_csv(val_path)
    pred = pd.concat([pred, normval[[f"{t}_uncertainty" for t in traits if f"{t}_uncertainty" in normval.columns]]], axis=1)
    pred = pd.concat([pred, valmeta[['longitude', 'latitude']]], axis=1)

    for t in traits:
        pred = pred.rename(columns={f"{t}_true": f"{t}_true", f"{t}_pred": f"{t}_pred"})
        if t.lower() == "leafarea":  # for benchmarking, unit conversion 
            pred[f"{t}_true"] *= 100
            pred[f"{t}_pred"] *= 100

    filters = np.ones(len(pred), dtype=bool)
    for t in traits:
        q = pred[f"{t}_pred"].quantile(0.95)
        filters &= pred[f"{t}_pred"] < q

    return pred, pred[filters]


# -------------------- METRIC FUNCTIONS --------------------
def weighted_pearsonr(x, y, w):
    mean_x, mean_y = np.average(x, weights=w), np.average(y, weights=w)
    cov_xy = np.sum(w * (x - mean_x) * (y - mean_y))
    return cov_xy / np.sqrt(np.sum(w * (x - mean_x) ** 2) * np.sum(w * (y - mean_y) ** 2))


def compute_weighted_metrics(df, latwts, pred_col, true_col, nmin=1, log_scale=False):
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
    rmse = np.sqrt(np.average(err**2, weights=w))
    rng = np.quantile(y_true, 0.99) - np.quantile(y_true, 0.01)
    nmae, nrmse = mae / rng, rmse / rng
    ss_res, ss_tot = np.sum(w * err**2), np.sum(w * (y_true - np.average(y_true, weights=w))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    slope, _, *_ = linregress(y_true, y_pred)
    r = weighted_pearsonr(y_true, y_pred, w)
    return dict(n=len(df), mae=mae, nmae=nmae, rmse=rmse, nrmse=nrmse, r2=r2, slope=slope, pearson_r=r)


# -------------------- HEXBIN PLOT --------------------
def plothexbin(df, true_col, pred_col, latwts, ax, label, title):
    metrics = compute_weighted_metrics(df, latwts, pred_col, true_col)
    ax.set_aspect('equal')
    ax.hexbin(df[true_col], df[pred_col], gridsize=60, cmap='plasma', mincnt=1)
    lims = [min(df[true_col].min(), df[pred_col].min()), max(df[true_col].max(), df[pred_col].max())]
    ax.plot(lims, lims, 'r--')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Observed {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(title)
    return metrics


# -------------------- SPLOT BENCHMARKING --------------------
def splot_benchmarking(pred, cfg, traits, out_dir):
    results = []
    for t in traits:
        splot_df = read_trait_map(t, "splot").to_dataframe(name=t).dropna()
        grid_true = global_grid_df(pred, f"{t}_true", lon="longitude", lat="latitude", res=1, stats=["mean"])
        grid_pred = global_grid_df(pred, f"{t}_pred", lon="longitude", lat="latitude", res=1, stats=["mean"])
        combined = grid_true.join(grid_pred, how="inner").rename(columns={"mean_x": f"{t}_true", "mean_y": f"{t}_pred"})

        latwts = lat_weights(combined.index.get_level_values("y").unique().values, 1)
        metrics = compute_weighted_metrics(combined, latwts, f"{t}_pred", f"{t}_true")
        results.append(dict(trait=t, **metrics))
    pd.DataFrame(results).to_csv(Path(out_dir) / "splot_metrics.csv", index=False)
    return results


# -------------------- MAIN EXECUTION --------------------
def get_result_dirs(base_dir):
    dirs = sorted(glob(os.path.join(base_dir, "results_ckpt_epoch_*")))
    file_pairs = []
    for d in dirs:
        for val_path in glob(os.path.join(d, "V3_VAL_ckpt_*.csv")):
            norm = os.path.join(d, f"normalized_{os.path.basename(val_path)}")
            if os.path.exists(norm):
                file_pairs.append((val_path, norm))
    return {int(re.search(r'epoch_(\d+)', p[0]).group(1)): {"VAL": p[0], "normval": p[1]} for p in file_pairs}


def run_benchmark(all_seed_paths, valmeta, cfg, out_dir, traits):
    summary = []
    for seed, epochs in all_seed_paths.items():
        for epoch, files in epochs.items():
            pred, pred_filt = prepare_data(files["VAL"], files["normval"], valmeta, traits)
            res = splot_benchmarking(pred_filt, cfg, traits, out_dir)
            for r in res:
                r.update(dict(seed=seed, epoch=epoch))
                summary.append(r)
    df = pd.DataFrame(summary)
    df.to_csv(Path(out_dir) / "epoch_wise_summary.csv", index=False)
    return df


if __name__ == "__main__":
    args = parse_args()
    valmeta = pd.read_csv("/path/to/val.csv")

    all_seed_paths = {
        0: get_result_dirs(args.base_dir_seed0),
        100: get_result_dirs(args.base_dir_seed100),
        200: get_result_dirs(args.base_dir_seed200),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    run_benchmark(all_seed_paths, valmeta, cfg, args.output_dir, args.traits)
    print(f"✅ Benchmarking complete. Results saved in {args.output_dir}")
