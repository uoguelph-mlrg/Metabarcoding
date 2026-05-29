#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging as log
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import Config, set_seed  # noqa: E402
from dataset import MBDataset, collate_samples  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from mlp import MLPModel  # noqa: E402
from model import Model  # noqa: E402
from neighbor_graph import NeighbourGraph  # noqa: E402
from utils import load, load_preprocessing_state  # noqa: E402


class _StubLatentSolver:
    """Satisfies Model.__init__ when interpolation_enabled=False (solver never called)."""

    def get_interpolation_operator(self, include_self: bool) -> None:
        raise RuntimeError("_StubLatentSolver: interpolation is disabled in eval mode")


def _resolve_checkpoint_path(model_arg: str) -> str:
    path = os.path.abspath(os.path.expanduser(model_arg))

    if os.path.isfile(path):
        if not path.endswith(".pt"):
            raise ValueError(f"Expected a .pt checkpoint file, got: {path}")
        return path

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Model path does not exist: {path}")

    checkpoints_dir = path if os.path.basename(path) == "checkpoints" else os.path.join(path, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(
            f"Could not find checkpoints directory under: {path}. "
            "Pass either a checkpoint file or a run directory containing checkpoints/."
        )

    preferred = [
        os.path.join(checkpoints_dir, "best.pt"),
        os.path.join(checkpoints_dir, "latest.pt"),
    ]
    for cand in preferred:
        if os.path.isfile(cand):
            return cand

    pt_files = [
        os.path.join(checkpoints_dir, name)
        for name in os.listdir(checkpoints_dir)
        if name.endswith(".pt")
    ]
    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoints found in: {checkpoints_dir}")

    pt_files.sort(key=os.path.getmtime, reverse=True)
    return pt_files[0]


def _config_from_checkpoint(saved_cfg: Dict[str, Any]) -> Config:
    cfg = Config()
    for key, value in saved_cfg.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _infer_results_layout(checkpoint_path: str, fallback_model_name: str) -> Tuple[str, str]:
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if os.path.basename(ckpt_dir) == "checkpoints":
        base_artifact_dir = os.path.dirname(ckpt_dir)
        return os.path.dirname(base_artifact_dir), os.path.basename(base_artifact_dir)
    return os.path.dirname(ckpt_dir), fallback_model_name


def _extract_bin_uris(preprocessing_state_path: str) -> Optional[List[str]]:
    state = load_preprocessing_state(preprocessing_state_path)
    raw = state.get("bin_uris")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("Invalid preprocessing artifact: 'bin_uris' must be a list")
    return [str(b) for b in raw]


def _preprocess_global_dataset(
    cfg: Config,
    training_state_path: str,
) -> Tuple[Any, Any, Dict[Any, int], Dict[Any, int], Any]:
    """
    Load the global CSV applying training normalization stats (replay mode).

    Replay mode is triggered by passing the training preprocessing_state_path to load(),
    which reuses training means/stds for normalization instead of recomputing them.
    All three splits (train/val/test) on the global data are merged into one eval set.

    Returns: (X_all, y_all, bin_index, sample_index, taxonomy_df)
    """
    import pandas as pd

    data, taxonomy_df, _emb, _bwe, bin_index, sample_index, _split_indices, _ = load(
        cfg,
        fixed_split_indices=None,
        preprocessing_state_path=training_state_path,
    )

    X_all = pd.concat([data["train"]["X"], data["val"]["X"], data["test"]["X"]])
    y_all = pd.concat([data["train"]["y"], data["val"]["y"], data["test"]["y"]])

    return X_all, y_all, bin_index, sample_index, taxonomy_df


def _interpolate_new_bin_latents(
    new_bin_uris: List[str],
    global_bin_index: Dict[str, int],
    taxonomy_df: Any,
    training_bin_uris: List[str],
    checkpoint_latent: torch.Tensor,
    cfg: Config,
) -> Tuple[torch.Tensor, int, int]:
    """
    For BINs in the global dataset not present in original training, interpolate their
    latents from the NW-weighted average of trained neighbors (by taxonomy distance).

    Returns: (latent_matrix, n_interpolated, n_zero_init)
        latent_matrix: [n_global_bins, embed_dim] with interpolated rows filled in
                        (shared BIN rows must already be set before calling)
        n_interpolated: number of new BINs that got at least one trained neighbor
        n_zero_init: number of new BINs with no trained neighbors (remain zero)
    """
    n_global = len(global_bin_index)
    embed_dim = checkpoint_latent.shape[1] if checkpoint_latent.dim() == 2 else 1
    latent_out = torch.zeros(
        (n_global, embed_dim) if embed_dim > 1 else (n_global,),
        dtype=checkpoint_latent.dtype,
    )

    training_bin_set = set(training_bin_uris)
    training_bin_to_idx = {uri: i for i, uri in enumerate(training_bin_uris)}

    # Build neighbor graph on global BINs (taxonomy only, no embeddings needed)
    log.info("Building taxonomy neighbor graph on %d global BINs for latent interpolation...", n_global)
    ng_cfg = Config()
    # Copy neighbor-relevant fields from training config
    for attr in ("use_taxonomy", "use_embedding", "neighbor_mode", "K", "dist_thres",
                 "kernel_q", "interpolation_method"):
        if hasattr(cfg, attr):
            setattr(ng_cfg, attr, getattr(cfg, attr))
    ng_cfg.use_embedding = False
    ng_cfg.use_taxonomy = True

    ng = NeighbourGraph(ng_cfg, taxonomy_df, embeddings=None, bins_with_embedding=None)
    ng.build()

    q = ng.compute_kernel_q()

    # global bin_uri list in index order
    global_bins_ordered = [""] * n_global
    for uri, idx in global_bin_index.items():
        global_bins_ordered[idx] = str(uri)

    n_interpolated = 0
    n_zero_init = 0

    new_bin_set = set(new_bin_uris)

    for new_uri in new_bin_uris:
        global_i = global_bin_index[new_uri]
        neighbor_idxs, weights = ng.nw_weights_for_node(global_i, q=q)

        # Keep only neighbors that are trained BINs
        trained_neighbor_mask = np.array(
            [global_bins_ordered[j] in training_bin_set for j in neighbor_idxs],
            dtype=bool,
        )
        trained_neighbor_idxs = neighbor_idxs[trained_neighbor_mask]
        trained_weights = weights[trained_neighbor_mask]

        if len(trained_neighbor_idxs) == 0:
            n_zero_init += 1
            continue

        # Re-normalize weights over the trained subset
        trained_weights = trained_weights / (trained_weights.sum() + 1e-12)

        # Weighted sum of trained latent rows
        interp = torch.zeros(embed_dim if embed_dim > 1 else 1, dtype=checkpoint_latent.dtype)
        for j, w in zip(trained_neighbor_idxs, trained_weights):
            neighbor_uri = global_bins_ordered[j]
            train_idx = training_bin_to_idx[neighbor_uri]
            if embed_dim > 1:
                interp += float(w) * checkpoint_latent[train_idx]
            else:
                interp[0] += float(w) * checkpoint_latent[train_idx]

        if embed_dim > 1:
            latent_out[global_i] = interp
        else:
            latent_out[global_i] = interp[0]

        n_interpolated += 1

    return latent_out, n_interpolated, n_zero_init


def _build_eval_model(
    checkpoint: Dict[str, Any],
    global_bin_index: Dict[str, int],
    training_bin_uris: List[str],
    taxonomy_df: Any,
    cfg: Config,
    device: torch.device,
) -> Tuple[Model, Dict[str, Any]]:
    """
    Reconstruct the model (MLP + latent_vec) from checkpoint for cross-dataset eval.

    The H matrix (LatentSolver) is NOT rebuilt — interpolation is disabled.
    The latent_vec is remapped to the global BIN universe:
      - shared BINs: copy trained latent row
      - new BINs: interpolate from trained taxonomy neighbors (NW weights)

    Returns: (model, bin_set_analysis_dict)
    """
    saved_cfg = checkpoint["config"]
    embed_dim = int(saved_cfg.get("embed_dim", 1))
    model_state_dict = dict(checkpoint["model_state_dict"])

    # Infer input_dim from first MLP layer weight
    first_weight_key = next(
        (k for k in model_state_dict if "mlp" in k and k.endswith(".weight")), None
    )
    if first_weight_key is None:
        raise ValueError("Cannot infer input_dim: no MLP weight found in model_state_dict")
    input_dim = model_state_dict[first_weight_key].shape[1]

    mlp = MLPModel(
        input_dim,
        hidden_dims=saved_cfg.get("mlp_hidden_dims", [256, 128]),
        output_dim=embed_dim,
        dropout=float(saved_cfg.get("dropout", 0.0)),
    ).to(device)

    n_global = len(global_bin_index)
    model = Model(
        mlp,
        _StubLatentSolver(),
        n_bins=n_global,
        device=device,
        latent_init_std=0.0,
        embed_dim=embed_dim,
        gating_fn=saved_cfg.get("gating_fn", "sigmoid"),
        gating_alpha=float(saved_cfg.get("gating_alpha", 1.0)),
        gating_kappa=float(saved_cfg.get("gating_kappa", 1.0)),
        gating_epsilon=float(saved_cfg.get("gating_epsilon", 0.0)),
        interpolation_enabled=False,
    )

    # Load MLP + final_linear weights (not latent_vec — handled separately)
    non_latent_state = {k: v for k, v in model_state_dict.items() if k != "latent_vec"}
    missing, unexpected = model.load_state_dict(non_latent_state, strict=False)
    # latent_vec is always "missing" since we skip it intentionally
    unexpected_real = [k for k in unexpected if k != "latent_vec"]
    missing_real = [k for k in missing if k != "latent_vec"]
    if unexpected_real or missing_real:
        log.warning("load_state_dict: missing=%s, unexpected=%s", missing_real, unexpected_real)

    # Compute BIN set analysis
    training_bin_set = set(training_bin_uris)
    global_bin_set = set(global_bin_index.keys())
    shared_bin_uris = sorted(training_bin_set & global_bin_set)
    new_bin_uris = sorted(global_bin_set - training_bin_set)
    train_only_bin_uris = sorted(training_bin_set - global_bin_set)

    log.info(
        "BIN universe: training=%d, global=%d, shared=%d, new=%d, train-only=%d",
        len(training_bin_uris), n_global,
        len(shared_bin_uris), len(new_bin_uris), len(train_only_bin_uris),
    )

    # Build remapped latent: shape [n_global, embed_dim] initialized to zeros
    checkpoint_latent = model_state_dict["latent_vec"]
    training_bin_to_old_idx = {uri: i for i, uri in enumerate(training_bin_uris)}

    if embed_dim > 1:
        remapped_latent = torch.zeros((n_global, embed_dim), dtype=checkpoint_latent.dtype)
    else:
        remapped_latent = torch.zeros(n_global, dtype=checkpoint_latent.dtype)

    # Copy shared BINs
    for uri in shared_bin_uris:
        new_idx = global_bin_index[uri]
        old_idx = training_bin_to_old_idx[uri]
        remapped_latent[new_idx] = checkpoint_latent[old_idx]

    # Interpolate new BINs from trained taxonomy neighbors
    n_interpolated = 0
    n_zero_init = len(new_bin_uris)
    if new_bin_uris:
        interp_latent, n_interpolated, n_zero_init = _interpolate_new_bin_latents(
            new_bin_uris=new_bin_uris,
            global_bin_index=global_bin_index,
            taxonomy_df=taxonomy_df,
            training_bin_uris=training_bin_uris,
            checkpoint_latent=checkpoint_latent,
            cfg=cfg,
        )
        # Copy interpolated rows into remapped_latent (zero-init rows stay zero)
        for uri in new_bin_uris:
            i = global_bin_index[uri]
            if embed_dim > 1:
                if interp_latent[i].any():
                    remapped_latent[i] = interp_latent[i]
            else:
                if interp_latent[i].item() != 0.0:
                    remapped_latent[i] = interp_latent[i]

    log.info(
        "Latent remap: %d shared (copied), %d new (interpolated), %d new (zero-init)",
        len(shared_bin_uris), n_interpolated, n_zero_init,
    )

    model.latent_vec.data.copy_(remapped_latent.to(device=device))

    bin_set_analysis = {
        "shared_bin_uris": shared_bin_uris,
        "new_bin_uris": new_bin_uris,
        "train_only_bin_uris": train_only_bin_uris,
        "shared_bin_fraction": len(shared_bin_uris) / max(1, len(training_bin_uris)),
        "latent_remap_info": {
            "shared_bins": len(shared_bin_uris),
            "interpolated_bins": n_interpolated,
            "zero_init_bins": n_zero_init,
        },
    }
    return model, bin_set_analysis


@torch.no_grad()
def _run_inference(
    model: Model,
    X_all: Any,
    y_all: Any,
    bin_index: Dict[Any, int],
    sample_index: Dict[Any, int],
    loss_mode: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run forward pass on the full merged dataset.
    Returns (predictions, targets, sample_labels, bin_labels).
    """
    from typing import Dict as _Dict, List as _List

    dataset = MBDataset(
        {"X": X_all, "y": y_all},
        bin_index,
        sample_index,
        loss_mode=loss_mode,
    )
    collate_fn = collate_samples if loss_mode == "sample" else None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)

    model.eval()
    model.to(device)

    sample_pred: _Dict[int, _List[float]] = {}
    sample_true: _Dict[int, _List[float]] = {}
    sample_bins: _Dict[int, _List[int]] = {}

    for batch in loader:
        if loss_mode == "sample":
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            bin_idx = batch["bin_idx"].to(device)
            sample_idx = batch["sample_idx"].to(device)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device)

            bsz = inputs.shape[0]
            for b in range(bsz):
                s_idx = int(sample_idx[b].item())
                valid_mask = mask[b].bool() if mask is not None else torch.ones(inputs.shape[1], dtype=torch.bool, device=device)
                inputs_flat = inputs[b][valid_mask]
                bin_idx_flat = bin_idx[b][valid_mask]
                outputs = model(inputs_flat, bin_idx_flat, interpolation_mask=None).unsqueeze(0)
                probs = F.softmax(outputs, dim=-1).squeeze(0).cpu().numpy()
                y_true = targets[b][valid_mask].cpu().numpy()
                sample_pred.setdefault(s_idx, []).extend(probs.tolist())
                sample_true.setdefault(s_idx, []).extend(y_true.tolist())
                sample_bins.setdefault(s_idx, []).extend(bin_idx_flat.cpu().numpy().tolist())
        else:
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            bin_idx = batch["bin_idx"].to(device)
            sample_idx = batch["sample_idx"].to(device)

            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
                targets = targets.unsqueeze(0)
                bin_idx = bin_idx.unsqueeze(0)
                sample_idx = sample_idx.unsqueeze(0)

            outputs = model(inputs, bin_idx, interpolation_mask=None)
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_trues = targets.cpu().numpy()
            s_idxs = sample_idx.cpu().numpy()
            b_idxs = bin_idx.cpu().numpy()
            for i in range(len(probs)):
                s = int(s_idxs[i])
                sample_pred.setdefault(s, []).append(float(probs[i]))
                sample_true.setdefault(s, []).append(float(y_trues[i]))
                sample_bins.setdefault(s, []).append(int(b_idxs[i]))

    if loss_mode == "bin":
        for s_idx, preds in sample_pred.items():
            pred_arr = np.array(preds)
            pred_sum = pred_arr.sum()
            if pred_sum > 0:
                sample_pred[s_idx] = (pred_arr / pred_sum).tolist()

    idx_to_sample = {v: k for k, v in sample_index.items()}
    idx_to_bin = {v: k for k, v in bin_index.items()}

    preds_flat, trues_flat, sample_labels, bin_labels = [], [], [], []
    for s_idx in sorted(sample_pred.keys()):
        n = len(sample_pred[s_idx])
        preds_flat.extend(sample_pred[s_idx])
        trues_flat.extend(sample_true[s_idx])
        sample_labels.extend([idx_to_sample[s_idx]] * n)
        bin_labels.extend([idx_to_bin[int(b)] for b in sample_bins[s_idx]])

    return (
        np.array(preds_flat, dtype=np.float32),
        np.array(trues_flat, dtype=np.float32),
        np.array(sample_labels),
        np.array(bin_labels),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint trained on one dataset on a different (larger) dataset."
    )
    parser.add_argument("--model", required=True, help="Checkpoint .pt path or run directory containing checkpoints/")
    parser.add_argument("--dataset", required=True, help="Path to evaluation CSV dataset (e.g. metabarcoding_dataset.csv)")
    parser.add_argument("--output", default=None, help="Output pickle path (default: next to checkpoint)")
    parser.add_argument("--device", default=None, choices=["cpu", "mps", "cuda"], help="Override device")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    log_level = log.DEBUG if args.verbose else log.INFO
    log.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    set_seed()

    checkpoint_path = _resolve_checkpoint_path(args.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint payload: expected dict in {checkpoint_path}")

    saved_cfg = checkpoint.get("config")
    if not isinstance(saved_cfg, dict):
        raise ValueError("Checkpoint is missing 'config' dictionary")
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing 'model_state_dict'")

    model_name = str(checkpoint.get("model_name") or "model")
    run_id = str(checkpoint.get("run_id") or "unknown")
    inferred_results_dir, inferred_model_name = _infer_results_layout(checkpoint_path, model_name)

    preprocessing_state_path = checkpoint.get("preprocessing_state_path")
    if not preprocessing_state_path:
        raise ValueError("Checkpoint is missing 'preprocessing_state_path'")
    preprocessing_state_path = os.path.abspath(os.path.expanduser(str(preprocessing_state_path)))
    if not os.path.isfile(preprocessing_state_path):
        raise FileNotFoundError(
            f"Preprocessing artifact not found at '{preprocessing_state_path}'. "
            "Evaluation requires the preprocessing artifact saved during training."
        )

    training_bin_uris = _extract_bin_uris(preprocessing_state_path)
    if training_bin_uris is None:
        raise ValueError(
            "Preprocessing artifact does not contain 'bin_uris'. "
            "Re-run training with the updated utils.py to save this field."
        )
    log.info("Training BIN universe: %d BINs loaded from preprocessing artifact", len(training_bin_uris))

    cfg = _config_from_checkpoint(saved_cfg)
    cfg.data_path = os.path.abspath(os.path.expanduser(args.dataset))
    cfg.results_dir = inferred_results_dir
    cfg.preprocessed_dir = None  # prevent stale original cache from being loaded
    cfg.inference_with_interpolation = False  # H matrix not available in cross-dataset eval
    if args.device is not None:
        cfg.device = args.device
    device = torch.device(cfg.device)

    log.info("Loading global dataset from %s (replay mode: reusing training normalization stats)...", cfg.data_path)
    X_all, y_all, global_bin_index, global_sample_index, taxonomy_df = _preprocess_global_dataset(
        cfg, preprocessing_state_path
    )
    log.info("Global dataset: %d observations, %d samples, %d BINs", len(X_all), len(global_sample_index), len(global_bin_index))

    log.info("Building eval model and remapping latent vectors...")
    model, bin_set_analysis = _build_eval_model(
        checkpoint=checkpoint,
        global_bin_index=global_bin_index,
        training_bin_uris=training_bin_uris,
        taxonomy_df=taxonomy_df,
        cfg=cfg,
        device=device,
    )

    loss_type = str(saved_cfg.get("loss_type", "cross_entropy"))
    loss_mode = "sample" if loss_type == "cross_entropy" else "bin"
    batch_size = int(saved_cfg.get("batch_size_sample", 32)) if loss_mode == "sample" else int(saved_cfg.get("batch_size_bin", 512))

    log.info("Running inference on %d global samples...", len(global_sample_index))
    predictions, targets, sample_labels, bin_labels = _run_inference(
        model=model,
        X_all=X_all,
        y_all=y_all,
        bin_index=global_bin_index,
        sample_index=global_sample_index,
        loss_mode=loss_mode,
        batch_size=batch_size,
        device=device,
    )

    eval_metrics = compute_metrics(predictions, targets, sample_labels=sample_labels)
    log.info("Eval metrics: %s", {k: f"{v:.4f}" for k, v in eval_metrics.items()})

    results = {
        "model": model_name,
        "run_id": run_id,
        "eval_dataset": cfg.data_path,
        "best_val_loss": float(checkpoint.get("best_val_loss", float("inf"))),
        "train_losses": list(checkpoint.get("train_losses", [])),
        "val_losses": list(checkpoint.get("val_losses", [])),
        "val_metrics": dict(checkpoint.get("val_metrics", {})),
        "predictions": predictions,
        "targets": targets,
        "sample_labels": sample_labels,
        "bin_labels": bin_labels,
        "eval_metrics": eval_metrics,
        "latent_vector": model.latent_vec.detach().cpu().numpy(),
        "n_training_bins": len(training_bin_uris),
        "n_global_bins": len(global_bin_index),
        **bin_set_analysis,
    }

    if args.output is not None:
        output_path = os.path.abspath(os.path.expanduser(args.output))
    else:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(
            os.path.dirname(checkpoint_path),
            f"results_cross_eval_{model_name}_{timestamp}.pkl",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(results, fh)

    log.info("Checkpoint: %s", checkpoint_path)
    log.info("Dataset: %s", cfg.data_path)
    log.info("Saved evaluation results to: %s", output_path)


if __name__ == "__main__":
    main()
