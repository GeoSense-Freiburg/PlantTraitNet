#!/usr/bin/env python3
"""
Filter training metadata based on prediction uncertainty (top X%) and
normalized MAE thresholds across multiple traits.

This script:
1. Loads normalized and unnormalized prediction CSVs.
2. Merges uncertainty columns.
3. Computes min–max normalized MAE (nMAE) per trait.
4. Selects samples with uncertainty above a quantile (e.g., 95th).
5. Among these, filters samples with nMAE > threshold (e.g., 50%).
6. Collects union of bad indices across all traits.
7. Merges with existing filtered indices (e.g., common_idx).
8. Produces a cleaned `train_meta` CSV.

All inputs, thresholds, and traits are passed as arguments.
"""

import argparse
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_nmae(df, trait):
    """Compute min–max normalized MAE for a trait."""
    pred_col = f"{trait}_median_pred"
    true_col = f"{trait}_median_true"

    if pred_col not in df.columns or true_col not in df.columns:
        raise KeyError(f"Missing predicted or true column for trait {trait}")

    tmin, tmax = df[true_col].min(), df[true_col].max()
    rng = max(tmax - tmin, 1e-8)

    df[f"{trait}_nMAE"] = (df[pred_col] - df[true_col]).abs() / rng * 100
    return df


def get_uncertain_indices(df, trait, quantile):
    """Return indices where uncertainty > given quantile."""
    ucol = f"{trait}_median_uncertainty"
    if ucol not in df.columns:
        raise KeyError(f"Missing uncertainty column for trait {trait}")

    thresh = df[ucol].quantile(quantile)
    return df[df[ucol] > thresh].index


def get_high_nmae_indices(df, trait, indices, nmae_thresh):
    """Return subset of indices where nMAE > threshold."""
    ncol = f"{trait}_nMAE"
    if ncol not in df.columns:
        raise KeyError(f"nMAE column missing for trait {trait}")

    subset = df.loc[indices]
    return subset[subset[ncol] > nmae_thresh].index.tolist()


def main(args):
    # Load all CSVs
    norm_val = pd.read_csv(args.normalized_csv)
    val = pd.read_csv(args.val_csv)
    train_meta = pd.read_csv(args.train_meta_csv)

    print(f"Loaded normalized: {norm_val.shape}")
    print(f"Loaded unnormalized: {val.shape}")
    print(f"Loaded train_meta: {train_meta.shape}")

    # Merge last 4 uncertainty columns (your original notebook logic)
    uncertainty_cols = norm_val.columns[::-1][:4]
    val = pd.concat([val, norm_val[uncertainty_cols]], axis=1)
    
    # Attach metadata (image, Species, pft)
    metadata_cols = ["image", "Species", "pft"]
    metadata_cols = [c for c in metadata_cols if c in train_meta.columns]
    val = pd.concat([val, train_meta[metadata_cols]], axis=1)

    # For each trait compute:
    # 1. nMAE
    # 2. uncertainty > quantile threshold
    # 3. among those: nMAE > threshold
    all_bad_indices = set()

    for trait in args.traits:
        print(f"\nProcessing trait: {trait}")

        # ---- Compute nMAE
        val = compute_nmae(val, trait)

        # ---- Indices with high uncertainty
        uncertain_idx = get_uncertain_indices(val, trait, args.uncertainty_quantile)
        print(f"  Uncertain (> quantile): {len(uncertain_idx)}")

        # ---- Among uncertain, find nMAE > threshold
        bad_idx = get_high_nmae_indices(val, trait, uncertain_idx, args.nmae_threshold)
        print(f"  High nMAE (> {args.nmae_threshold}%): {len(bad_idx)}")

        all_bad_indices.update(bad_idx)

    print(f"\nTotal unique bad samples across traits: {len(all_bad_indices)}")

    # Merge with provided bad index file (e.g., common_idx)
    if args.common_idx_file is not None:
        common_idx = pd.read_csv(args.common_idx_file, header=None)[0].tolist()
        all_bad_indices.update(common_idx)
        print(f"Merged with common_idx → total: {len(all_bad_indices)}")

    # Drop from train_meta
    train_clean = train_meta.drop(index=all_bad_indices, errors="ignore")
    print(f"Final cleaned size: {len(train_clean)} (original {len(train_meta)})")

    train_clean.to_csv(args.output_csv, index=False)
    print(f"\nSaved cleaned train_meta: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter training metadata using uncertainty + nMAE thresholds.")

    parser.add_argument("--normalized_csv", type=str, required=True,
                        help="CSV with normalized outputs (containing uncertainty).")

    parser.add_argument("--val_csv", type=str, required=True,
                        help="CSV with unnormalized model predictions.")

    parser.add_argument("--train_meta_csv", type=str, required=True,
                        help="Training metadata CSV to clean.")

    parser.add_argument("--traits", nargs="+",
                        default=["Height_median", "LeafArea_median", "SLA_median", "Leaf_N_median"],
                        help="Trait base names used for _pred, _true, _uncertainty columns.")

    parser.add_argument("--uncertainty_quantile", type=float, default=0.95,
                        help="Quantile threshold to define high uncertainty.")

    parser.add_argument("--nmae_threshold", type=float, default=50,
                        help="nMAE percentage threshold for filtering uncertain predictions.")

    parser.add_argument("--common_idx_file", type=str, default=None,
                        help="Optional CSV file containing additional bad indices (one index per row).")

    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to save cleaned train_meta CSV.")

    args = parser.parse_args()
    main(args)
