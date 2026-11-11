# Team rules for experiments

Short rules so that we can work in parallel without conflicts and every result can be trusted and repeated.
For installing, see [docs/SETUP.md](docs/SETUP.md); for branches and pull requests, see [docs/WORKFLOW.md](docs/WORKFLOW.md).

## 1. Where things live

| What | Where | How it gets to UBELIX |
|---|---|---|
| Code | GitHub | `git clone` / `git pull` on UBELIX |
| Dataset | OneDrive | copied once from the laptop with `scp` |
| Run results (small) | `runs/` → OneDrive | `scp` to the laptop after each run, then into the shared OneDrive `runs/` folder |
| Segmentations, cache (large) | `outputs/`, `cache/` | stay where they were computed |

Never commit data, results or caches to GitHub.

Project folder, the same for everybody (on UBELIX: `~/MIA_Lab_Project`):

```
MIA_Lab_Project/
    code/                     this repository (on the laptop it may be called mialab/)
    data/MIALab_2020_Dataset/
    envs/mialab/              Python environment (UBELIX)
    cache/  runs/  outputs/  logs/
```

Set up UBELIX once with `bash ubelix/setup_ubelix.sh`.

## 2. Running experiments

- One experiment = one config file in `configs/experiments/`, e.g. `E1_texture.toml`. Its `name` is used in the run
  folder name. Never change a config after it was used for a reported run; copy it instead.
- Run on UBELIX: `sbatch ubelix/run_pipeline.sbatch --config configs/experiments/E1_texture.toml`.
- Try new code first with `--smoke` (3 subjects, a few minutes).
- Commit your code before a run that you report. The commit is saved in the run folder; uncommitted changes are
  saved as `uncommitted.patch` and marked in `runs_index.csv`.
- Everybody uses the same `seed` (42) and the same folds (`configs/folds.json`). Never regenerate the folds.

## 3. The test set

- Every decision (features, groups, k, settings) is made with cross-validation on the training subjects only.
- `pipeline.py --final-test` is used only for the runs we fixed in advance, after all decisions are made, by one
  person. The code refuses a second test run of the same config.
- Never look at test images or test results to choose anything.

## 4. No leakage: fit inside the fold

Anything that **learns from data** must be learned from the training subjects of the current fold only:
probabilistic atlas, PCA, feature selection (top-k), correlation filter, scalers, autoencoders.
For a feature group, set `fold_dependent = True` and learn in `fit()`; the pipeline then fits it again in every fold.
Feature groups never receive the labels in `compute()`.

## 5. Adding a feature group

1. Work on your own branch, e.g. `feature/texture`.
2. Create one new file in `mialab/features/` (see `baseline.py` for an example) and add the class to `GROUPS` in
   `mialab/features/__init__.py`. Do not change other groups' files.
3. Give every parameter a default in `defaults`. Increase `version` whenever you change how a group is computed,
   so old cached values are not reused.
4. Add a config in `configs/experiments/` that switches the group on, run `--smoke`, then open a pull request.

## 6. Results

Each run writes `runs/<date>_<time>_<config name>_<user>/` with `config_used.json`, `commit.txt`,
`environment.txt`, `run_info.json`, `results_per_subject.csv` (Dice and HD95 per subject and tissue),
`results_summary.csv`, `feature_importance.csv` and `features.txt`.

- Copy **only your run folders** from `runs/` to the shared OneDrive `runs/` folder. The job prints the `scp` command.
- Never edit files inside a run folder.
- `python analysis/build_runs_index.py --runs_dir <OneDrive runs folder>` rebuilds `runs_index.csv`.
- All analysis and figures are made by scripts in `analysis/` from the saved files, so nothing has to be trained
  again: `compare_runs.py` (paired Wilcoxon test), `correlation_map.py` (feature correlation).
