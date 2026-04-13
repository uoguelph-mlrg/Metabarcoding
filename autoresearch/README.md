# autoresearch — Metabarcoding

Autonomous experimentation loop for the metabarcoding invertebrate-abundance prediction model.

The idea: give an AI agent the training code and let it experiment autonomously. It modifies `train.py`, submits a training job, checks whether val KL divergence improved, keeps or discards the commit, and repeats. You come back to a log of experiments and (hopefully) a better model.

## How it works

Three files matter:

- **`prepare.py`** — fixed data pipeline, preprocessing cache, `Loss` class, `MBDataset`, `collate_samples`, and `TIME_BUDGET`. **Not modified by the agent.**
- **`train.py`** — the only file the agent edits. Contains `Config`, `MLPModel`, `Model`, `LatentSolver`, `NeighbourGraph`, `Trainer`, optimizer, and the training loop. Everything is fair game.
- **`program.md`** — instructions for the agent. Edit this to steer the research direction.

Training runs for a **fixed 5-minute time budget** (wall clock, batch-loop time only). The metric is **val_kl_divergence** (mean per-sample KL divergence on the validation set) — lower is better.

## Quick start (local / interactive)

```bash
# Install dependencies into existing venv
source ~/barcode/bin/activate
pip install scipy scikit-learn tqdm matplotlib pandas numpy torch

# First run — preprocesses data + computes embeddings (one-time, ~10-60 min)
cd Metabarcoding/autoresearch
python train.py > run.log 2>&1
grep "^val_kl_divergence:" run.log
```

## Running on Killarney (SLURM cluster)

Use `submit.sh` to dispatch each training experiment as a SLURM job:

```bash
# First run — longer walltime for one-time embedding computation
./submit.sh --wait --time 90:00

# All subsequent experiments
./submit.sh --wait          # blocks ~7 min, then returns
grep "^val_kl_divergence:" run.log
```

See `submit.sh --help` for all options (GPU spec, memory, QoS, etc.).

## Running the agent

Start the agent in a `tmux` session on the login node and point it at `program.md`:

```
Hi, have a look at program.md and let's kick off a new experiment! Let's do the setup first.
```

The agent handles the entire loop: editing `train.py`, committing, submitting jobs, reading results, and deciding keep/discard.

## Project structure

```
prepare.py      — data pipeline + evaluation harness (read-only)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
submit.sh       — SLURM submission script (one job per experiment)
pyproject.toml  — Python dependencies
results.tsv     — experiment log (untracked by git)
run.log         — training output from the latest run (untracked by git)
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. Diffs stay small and reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes of batch-loop time, regardless of what the agent changes (model size, optimizer, batch size, etc.). This makes experiments directly comparable.
- **Cached preprocessing.** Data splits and BarcodeBERT embeddings are computed once and reused. Each subsequent experiment starts training in seconds.
- **KL divergence as metric.** Mean per-sample KL divergence is directly interpretable as a distributional prediction quality measure and is independent of class count.
