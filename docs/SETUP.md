# Setup and Workflow Guide

This guide explains how to set up the environment, test a pull request, and work with branches in this repository.

## 1. Folder layout

The dataset must be in the **parent folder** of the repository, not inside it. The pipeline reads it from `../MIALab_2020_Dataset`.

```
<your project folder>/
├── MIALab_2020_Dataset/   <- data (never committed)
│   ├── atlas/
│   ├── test/
│   └── train/
└── mialab/                <- this repository
```

The dataset, image files (`.nii.gz`, `.mha`, ...), `mia-result/`, zip files and cached features/models are listed in `.gitignore`, so they cannot be committed by accident. The repository is public, so please never force-add them.

## 2. Install uv

We use [uv](https://docs.astral.sh/uv/) to manage Python and packages. Check if it is installed:

```bash
uv --version
```

If not, install it.

Windows (PowerShell):

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen the terminal after installing.

## 3. Create the environment

Run these commands inside the repository folder. `uv` reads `.python-version` (Python 3.11) and downloads that Python version automatically if needed.

```bash
uv venv
```

```bash
uv pip install -r requirements.txt -e .
```

All package versions are pinned in `requirements.txt`, so everyone gets exactly the same environment. Do not upgrade packages on your own; if a package must change, update `requirements.txt` in a pull request.

The `.venv` folder is local to your machine and is never committed.

## 4. Activate the environment

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

## 5. Check the installation

```bash
python test/test_install.py
```

Expected output ends with: `Everything seems to work fine!`

## 6. Run the pipeline

```bash
python pipeline.py
```

- It takes several minutes and prints many warnings. This is expected while the exercise TODOs (normalization, skull stripping, registration, T2 features) are not implemented yet.
- Results are written to `mia-result/<date-time>/` (`results.csv`, `results_summary.csv`).
- Reference Dice scores for the unchanged pipeline (1 tree, depth 5): white matter about 0.61, grey matter about 0.44, hippocampus / amygdala / thalamus 0.00. Small differences are normal because the random forest is random.

### Running an experiment with a config file

All experiment settings (pre-processing steps, features, random forest parameters, random seed) are in TOML files in the `configs/` folder. Without `--config`, the pipeline uses `configs/default.toml`.

```bash
python pipeline.py --config configs/default.toml
```

To create a new experiment, copy `configs/default.toml`, give it a new `name`, and change only the settings you need. Settings left out of a file keep their default values, and a misspelled setting stops the run with an error.

Do not edit `configs/default.toml` to try things out: it holds the shared baseline settings and is only changed through a pull request that the team agrees on.

Each run creates `mia-result/<date-time>_<name>/` and saves the complete settings used in `config_used.json`, so every result can be traced back to its configuration. With the same config and seed, the results are identical.

## 7. Testing a pull request

Test a pull request in a **separate folder**, so your own unfinished work is not affected. Open a terminal in the parent folder (the one that contains `MIALab_2020_Dataset`), then:

```bash
git clone -b <branch-name> https://github.com/ElineDonze/mialab.git mialab-test
```

```bash
cd mialab-test
```

Then follow steps 3 to 6 in this folder. When you are done, you can delete the `mialab-test` folder.

## 8. Reviewing and merging a pull request on GitHub

1. Open the pull request and go to the **Files changed** tab to see the changes (green = added, red = removed).
2. Test it locally (section 7).
3. Click **Review changes**, choose **Approve** (or **Request changes** if something fails), add a short comment, and click **Submit review**.
4. Go back to the **Conversation** tab and click **Merge pull request**, then **Confirm merge**.
5. Click **Delete branch** if offered.

If something fails, do not merge. Paste the error into a comment on the pull request.

## 9. Branch workflow

- Never commit directly to `main`. Every change goes through a branch and a pull request reviewed by a teammate.
- One branch per task, named by type and topic, for example `feature/normalization`, `feature/t2-features`, `setup/environment`.
- Before starting work each day, update `main` and your branch:

```bash
git switch main
```

```bash
git pull
```

```bash
git switch <your-branch>
```

```bash
git merge main
```

- Keep pull requests small and focused, so they are easy to review and conflicts stay rare.
- Avoid editing the same file as a teammate at the same time; agree in the chat first.
