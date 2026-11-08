# Working on Your Own Branch

This guide explains how to work on your own branch and how to get your work into `main` through a pull request. For installing the environment, see [SETUP.md](SETUP.md).

Run all commands in a terminal, inside the `mialab` folder. In the examples, replace `<your-branch>` with your branch name (for example `ED-working-branch` or `feature/t2-features`).

## The golden rule

Before you commit, always check which branch you are on:

```bash
git branch
```

The line with `*` is your current branch. It must show your branch, **not** `main`. In VS Code you can also see the current branch in the bottom-left corner of the window.

Never commit or push directly to `main`. Every change goes through a pull request that a teammate reviews.

## Part A: Get your branch onto your laptop

If the branch was created on GitHub, download it once:

```bash
git fetch
```

```bash
git switch <your-branch>
```

If you want to create a new branch yourself, start from an up-to-date `main`:

```bash
git switch main
```

```bash
git pull
```

```bash
git switch -c <your-branch>
```

Check with `git branch` that you are on it.

## Part B: Every time you work

1. Update your local `main` and bring its changes into your branch:

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

2. Edit the code as usual.

3. See what changed:

```bash
git status
```

4. Stage only the files you want to upload:

```bash
git add <file>
```

For example `git add mialab/filtering/preprocessing.py`. Avoid `git add .`, so nothing unwanted is uploaded by accident.

5. Commit with a short, clear message that says what changed:

```bash
git commit -m "Normalize intensities using brain voxels only"
```

6. Push to your branch. The first time:

```bash
git push -u origin <your-branch>
```

After that, just:

```bash
git push
```

Your changes are now on your branch on GitHub. `main` is not changed.

## Part C: Merge your work into `main` with a pull request

1. Go to https://github.com/ElineDonze/mialab
2. A yellow bar usually appears: **"<your-branch> had recent pushes"**. Click **Compare & pull request**. If you do not see it: open the **Pull requests** tab, click **New pull request**, set **base: `main`** and **compare: `<your-branch>`**.
3. Write a title and a short description: what you changed and how you tested it (for example "pipeline.py runs without errors"). Click **Create pull request**.
4. Wait for a teammate to review it and click **Approve**. Nobody merges their own pull request without a review.
5. Click **Merge pull request**, then **Confirm merge**.
6. Update your laptop:

```bash
git switch main
```

```bash
git pull
```

7. Before continuing, bring your branch up to date (or create a new branch for the next task, see Part A):

```bash
git switch <your-branch>
```

```bash
git merge main
```

## Part D: "Oops, I was on `main`!"

**You edited files but did not commit yet.** Just switch; uncommitted changes come with you:

```bash
git switch <your-branch>
```

Then commit there as usual.

**You already committed on `main` but did not push.** Do not push. Ask a teammate for help moving the commit to your branch.

**You already pushed to `main`.** Nothing is lost. Tell the team so the next pull requests start from the right state.

## Part E: Merge conflicts

A conflict happens when you and a teammate changed the same lines. Git marks them in the file like this:

```
<<<<<<< HEAD
your version
=======
the other version
>>>>>>> main
```

Keep the correct version, delete the marker lines, then:

```bash
git add <file>
```

```bash
git commit
```

To avoid conflicts: keep pull requests small, merge `main` into your branch often, and agree in the chat before editing the same file as someone else.

## Cheat sheet

| I want to... | Command |
|---|---|
| See which branch I am on | `git branch` |
| Go to my branch | `git switch <your-branch>` |
| Create a new branch | `git switch -c <new-branch>` |
| See what changed | `git status` |
| Prepare a file for commit | `git add <file>` |
| Save a snapshot | `git commit -m "message"` |
| Upload to my branch | `git push` |
| Get the latest `main` into my branch | `git switch main`, `git pull`, `git switch <your-branch>`, `git merge main` |
