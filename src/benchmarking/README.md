# Unified Trait Benchmarking Script

This script benchmarks predicted plant traits (e.g., LeafN, Height, SLA, LeafArea) against the sPlot dataset across multiple seeds and epochs.

### What it does
- Loads model predictions and normalizations for each seed and epoch.
- Merges validation metadata.
- Computes weighted metrics (R², nMAE, nRMSE, Pearson r, slope).
- Saves both per-trait and epoch-wise summaries.

### Usage

```bash
python unified_benchmark.py \
  --base_dir_seed0 /path/to/seed0 \
  --base_dir_seed100 /path/to/seed100 \
  --base_dir_seed200 /path/to/seed200 \
  --output_dir /path/to/output \
  --traits LeafN Height LeafArea
