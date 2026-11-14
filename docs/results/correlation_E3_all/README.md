# Feature correlation, E3_all (79 features)

Output tables of `analysis/correlation_map.py` for `configs/experiments/E3_all.toml`
(run on UBELIX, 2026-11-13). Spearman correlation on a random sample of brain voxels of the 20 training
subjects; no labels and no test subjects are used.

| File | Content |
|---|---|
| `correlation.csv` | full 79 x 79 matrix (feature x feature) |
| `group_summary.csv` | mean and max \|r\| between and within the feature groups |
| `high_pairs.csv` | the 217 feature pairs with \|r\| >= 0.9 |

To make more tables or the figure yourself:

    python analysis/correlation_map.py --config configs/experiments/E3_all.toml

(see the docstring at the top of the script for all options, e.g. `--threshold` and `--r2_targets`).
