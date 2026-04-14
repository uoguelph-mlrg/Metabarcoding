# Metabarcoding

Objective: Train a non-parametric (latent-based) + parametric (MLP) model to predict species relative abundance from metabarcoding data

## Repository layout

- `src/`: Core architecture and training code
	- `train.py`: Main entrypoint for training and evaluation
	- `config.py`: Default experiment configuration
	- `dataset.py`: Dataset loading
    - `model.py`: Model architecture definitions (latent and MLP modules)
    - `mlp.py`: MLP-specific architectures
    - `loss.py`: Losses function definitions (cross-entropy and logistic)
    - `latent_solver.py`: Non-parametric latent solver implementation (precompute interpolation operators, optimize latent representations, etc.)
    - `neighbor_graph.py`: Build neighbour lists and compute interpolation weights + handle barcode embedding when applicable
    - `gating_functions.py`: Define a variety of gating functions to combine latent and MLP predictions (when both are vectors)
    - `utils.py`: Data loading and preprocessing utilities
- `analysis/`: Experiment variants, cluster launch scripts, and visualization helpers
    - one subdirectory per experiment variant (e.g. `baselines/`, `ablation/`, etc.)
	- `submit_subanalysis.sh`: Unified SLURM launcher
	- `LAUNCHING.md`: Detailed cluster usage
    - `visualize_results.py`: Plotting and result analysis scripts
- `data/`: Input data files and EDA notebooks
- `autoresearch/`: Adaptation of the AutoResearch framework for this project, including training loop, agent instructions, and experiment management utilities

## Quick start

Run a baseline training job from the `Metabarcoding/` directory:

```bash
python src/train.py --model baseline
```

Run any other model variant:

```bash
python src/train.py --model <variant_name>
```

Resume the latest checkpoint for a variant:

```bash
python src/train.py --model <variant_name> --resume
```

## Subanalysis jobs (cluster)

From `Metabarcoding/analysis`:

```bash
./submit_subanalysis.sh --list-targets
./submit_subanalysis.sh --target location_embedding
```

For full cluster workflow details, see `analysis/LAUNCHING.md`.
