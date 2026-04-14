# autoresearch — Metabarcoding

Autonomous experimentation loop for the metabarcoding abundance-prediction model.

## Setup (do this once per session)

Work with the user to complete the following before starting the loop:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr13`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.

2. **Create the branch** from `master`:
   ```
   git checkout master
   git checkout -b autoresearch/<tag>
   ```

3. **Read the in-scope files** for full context:
   - `prepare.py` — fixed: data loading, preprocessing cache, `Loss` class, `MBDataset`, `collate_samples`, `TIME_BUDGET`. Do **not** modify.
   - `train.py` — the **only** file you edit: `Config`, model classes (`MLPModel`, `Model`, `LatentSolver`, `NeighbourGraph`), `Trainer`, optimizer, hyperparameters.

4. **Verify data exists** at the paths in `Config`:
   - `data_path` → `../data/data_merged.csv` (main observations CSV)
   - `embedding_path` → set in Config to `../data/embeddings.npy` (cache; computed on first run, reused after)
   - Preprocessed splits (`X_train.csv`, `X_val.csv`, etc.) are auto-generated in `../data/` on the first run and reused on all subsequent runs.

5. **Initialize `results.tsv`** (not committed to git — leave untracked):
   ```
   commit	val_kl	status	description
   ```

6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU, within a **fixed time budget of 5 minutes** (wall-clock training time; preprocessing and evaluation overhead are excluded). Launch with:

```
./submit.sh --wait
```

This submits a SLURM job and blocks until it completes (~7 min). Training output (including the `---` summary) is written to `run.log`.

**What you CAN do** (modify `train.py` only):
- `Config` — any hyperparameter: lr, batch size, epochs, dropout, embed_dim, K, neighbor_mode, gating_fn, loss_type, regularization weights, etc.
- Model architecture — `MLPModel`, `Model`, `LatentSolver`, `NeighbourGraph`
- Optimizer / scheduler logic in `Trainer`
- Training loop in `Trainer.run()`

**What you CANNOT do**:
- Modify `prepare.py` — it is read-only. The data pipeline, `Loss` class, and `TIME_BUDGET` are fixed.
- Install new packages. Use only what is in `pyproject.toml`.
- Modify the KL-divergence metric computed in `compute_metrics()`. It is the ground truth objective.

**The goal is simple: get the lowest `val_kl_divergence`** (mean per-sample KL divergence on the validation set, lower is better). Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_kl gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_kl improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_kl improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_kl_divergence:  0.123456
test_kl_divergence: 0.125000
val_loss:           0.456789
test_loss:          0.460000
training_seconds:   300.1
total_seconds:      340.2
num_epochs:         18
test/kl_divergence:  0.125000
test/rmse_macro:     0.034500
...
```

Extract the key metric:
```
grep "^val_kl_divergence:" run.log
```

---

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and the following columns:
```
commit	val_kl	status	description
```

1. Short git commit hash (7 chars)
2. `val_kl_divergence` from the run (use `0.000000` for crashes)
3. status: `keep`, `discard`, or `crash`
4. Short description of what was tried

Example:
```
commit	val_kl	status	description
a1b2c3d	0.123456	keep	baseline
b2c3d4e	0.119200	keep	increase K from 25 to 50
c3d4e5f	0.131000	discard	switch gating to exp
d4e5f6g	0.000000	crash	embed_dim=50 (OOM)
```

---

## Experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr13` or `autoresearch/apr13-gpu0`).

```
LOOP FOREVER:

1. Check git state (current branch/commit)
2. Edit train.py with an experimental idea by directly modifying the code.
3. git commit -m "short description"
4. `./submit.sh --wait` — submits a SLURM job and blocks until it completes. Training output goes to `run.log`. Do NOT run train.py directly.
5. Read out the results: `grep "^val_kl_divergence:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_kl improved (lower): you "advance" the branch, keeping the git commit
9. If val_kl is equal or worse: git reset --hard HEAD~1 and try a new idea
```

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each run should complete in ~5 min of training + ~1 min overhead. If a run exceeds 12 minutes total, kill it and treat as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
