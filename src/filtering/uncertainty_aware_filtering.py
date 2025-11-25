#!/usr/bin/env python3
"""
Filter training metadata using uncertainty thresholds derived from validation predictions.

This script:
1. Loads the normalized and unnormalized validation CSV files.
2. Extracts the four uncertainty columns (highest priority columns).
3. Merges uncertainties into the main validation dataframe.
4. Adds image metadata (Species, pft) from the training metadata file.
5. Computes the 95th percentile uncertainty thresholds for all traits.
6. Selects indices where uncertainty for all traits is below the 95th percentile.
7. Filters the training metadata accordingly.
8. Saves an updated training metadata file.

This creates a high-quality, uncertainty-filtered dataset for training the next model stage.
"""

import pandas as pd
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Filter training metadata using 95th percentile uncertainty thresholds.")

    parser.add_argument("--val", type=str, required=True,
                        help="Path to the unnormalized validation CSV.")
    parser.add_argument("--norm_val", type=str, required=True,
                        help="Path to the normalized validation CSV.")
    parser.add_argument("--train_meta", type=str, required=True,
                        help="Path to the original train metadata CSV.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output CSV path for filtered training metadata.")

    return parser.parse_args()

def main():
    args = parse_args()

    val = pd.read_csv(args.val)
    norm_val = pd.read_csv(args.norm_val)

    print("Loaded validation tables:", val.shape, norm_val.shape)

    # ---------------------------------
    # Extract the four uncertainty columns
    # (they appear as last 4 columns in reversed order)
    # ---------------------------------
    uncertainty_cols = norm_val.columns[::-1][:4]
    print("Using uncertainty columns:", list(uncertainty_cols))

    # Merge the uncertainty columns into val
    val = pd.concat([val, norm_val[uncertainty_cols]], axis=1)

    # Remove empty columns (if any)
    val = val.dropna(axis=1, how="all")

    # ---------------------------------
    # Add image metadata (Species, pft)
    # ---------------------------------
    train_meta = pd.read_csv(args.train_meta)
    submeta = train_meta[["image", "Species", "pft"]]

    val = pd.concat([val, submeta], axis=1)

    # ---------------------------------
    # Define main traits
    # ---------------------------------
    traits = ["Height_median", "LeafArea_median", "SLA_median", "Leaf_N_median"]

    # Compute 95th percentile thresholds
    thresholds = {
        trait: val[f"{trait}_uncertainty"].quantile(0.95)
        for trait in traits
    }

    print("\n95th percentile uncertainty thresholds:")
    for trait, th in thresholds.items():
        print(f"  {trait}: {th:.4f}")

    # ---------------------------------
    # Compute indices below 95th percentile for each trait
    # ---------------------------------
    idx_sets = [
        val[val[f"{trait}_uncertainty"] < thresholds[trait]].index
        for trait in traits
    ]

    # Intersection: indices valid for ALL traits
    common_indices = idx_sets[0]
    for idx in idx_sets[1:]:
        common_indices = common_indices.intersection(idx)

    print("\nNumber of patches passing all uncertainty filters:", len(common_indices))

    # ---------------------------------
    # Filter the training metadata
    # ---------------------------------
    train_meta_filtered = train_meta.loc[train_meta.index.intersection(common_indices)]

    print("Filtered training metadata shape:", train_meta_filtered.shape)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    train_meta_filtered.to_csv(args.output, index=False)

    print(f"\nSaved filtered training metadata → {args.output}")


if __name__ == "__main__":
    main()
