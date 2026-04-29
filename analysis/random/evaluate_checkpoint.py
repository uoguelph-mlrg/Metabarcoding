#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging as log
import os
import pickle
import shutil
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import Config, set_seed  # noqa: E402
from dataset import collate_samples  # noqa: E402
from train import Trainer  # noqa: E402
from utils import PREPROCESSING_STATE_FILENAME, load_preprocessing_state  # noqa: E402


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


def _extract_fixed_split_indices(preprocessing_state_path: str) -> Optional[Dict[str, np.ndarray]]:
    state = load_preprocessing_state(preprocessing_state_path)
    raw = state.get("split_indices")
    if not isinstance(raw, dict):
        return None

    required = ["train", "val", "test"]
    if any(k not in raw for k in required):
        return None

    return {k: np.asarray(raw[k], dtype=np.int64) for k in required}


def _extract_bin_uris(preprocessing_state_path: str) -> Optional[list[str]]:
    state = load_preprocessing_state(preprocessing_state_path)
    raw = state.get("bin_uris")
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("Invalid preprocessing artifact: 'bin_uris' must be a list")
    return [str(bin_uri) for bin_uri in raw]


def _rebuild_eval_loaders(trainer: Trainer) -> None:
    batch_size = trainer.cfg.batch_size_sample if trainer.loss_mode == "sample" else trainer.cfg.batch_size_bin
    collate_fn = collate_samples if trainer.loss_mode == "sample" else None
    num_workers = int(getattr(trainer.cfg, "num_workers", 0 if sys.platform == "darwin" else 8))
    pin_memory = bool(getattr(trainer.cfg, "pin_memory", trainer.device.type == "cuda"))

    loader_args = {
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }

    trainer.train_loader = DataLoader(trainer.train_dataset, shuffle=False, **loader_args)
    trainer.val_loader = DataLoader(trainer.val_dataset, shuffle=False, **loader_args)
    trainer.test_loader = DataLoader(trainer.test_dataset, shuffle=False, **loader_args)


def _concat_predictions(
    pred_train: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    pred_val: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    pred_test: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.concatenate([pred_train[0], pred_val[0], pred_test[0]], axis=0),
        np.concatenate([pred_train[1], pred_val[1], pred_test[1]], axis=0),
        np.concatenate([pred_train[2], pred_val[2], pred_test[2]], axis=0),
        np.concatenate([pred_train[3], pred_val[3], pred_test[3]], axis=0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a full dataset using saved preprocessing artifacts."
    )
    parser.add_argument("--model", required=True, help="Checkpoint .pt path or run directory containing checkpoints/")
    parser.add_argument("--dataset", required=True, help="Path to raw CSV dataset")
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

    saved_model_name = str(checkpoint.get("model_name") or "model")
    inferred_results_dir, inferred_model_name = _infer_results_layout(checkpoint_path, saved_model_name)
    model_name = saved_model_name or inferred_model_name

    cfg = _config_from_checkpoint(saved_cfg)
    cfg.data_path = os.path.abspath(os.path.expanduser(args.dataset))
    cfg.results_dir = inferred_results_dir
    if args.device is not None:
        cfg.device = args.device

    preprocessing_state_path = checkpoint.get("preprocessing_state_path")
    if not preprocessing_state_path:
        raise ValueError("Checkpoint is missing 'preprocessing_state_path'")
    preprocessing_state_path = os.path.abspath(os.path.expanduser(str(preprocessing_state_path)))
    if not os.path.isfile(preprocessing_state_path):
        raise FileNotFoundError(
            f"Preprocessing artifact not found at '{preprocessing_state_path}'. "
            "Evaluation requires the preprocessing artifact saved during training."
        )

    expected_state_path = os.path.join(
        os.path.abspath(cfg.results_dir),
        model_name,
        "checkpoints",
        PREPROCESSING_STATE_FILENAME,
    )
    if os.path.abspath(expected_state_path) != preprocessing_state_path:
        os.makedirs(os.path.dirname(expected_state_path), exist_ok=True)
        shutil.copy2(preprocessing_state_path, expected_state_path)
        log.info("Copied preprocessing artifact to %s for Trainer compatibility", expected_state_path)
    else:
        log.info("Using preprocessing artifact: %s", expected_state_path)

    fixed_split_indices = _extract_fixed_split_indices(expected_state_path)
    training_bin_uris = _extract_bin_uris(expected_state_path)

    trainer = Trainer(
        cfg=cfg,
        model_name=model_name,
        run_id=f"eval_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
        resume=False,
        fixed_split_indices=fixed_split_indices,
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing 'model_state_dict'")

    model_state_dict = dict(checkpoint["model_state_dict"])
    checkpoint_latent = model_state_dict.get("latent_vec")
    if checkpoint_latent is None:
        raise ValueError("Checkpoint model_state_dict is missing 'latent_vec'")

    current_latent_shape = tuple(trainer.model.latent_vec.shape)
    checkpoint_latent_shape = tuple(checkpoint_latent.shape)

    if checkpoint_latent_shape != current_latent_shape:
        if training_bin_uris is None:
            raise ValueError(
                "Checkpoint latent table shape does not match the evaluation dataset, and the "
                "preprocessing artifact does not contain 'bin_uris' to remap BIN-specific latent rows. "
                f"checkpoint_latent={checkpoint_latent_shape}, current_latent={current_latent_shape}. "
                "To evaluate a different BIN universe, regenerate the preprocessing artifact with BIN "
                "identity saved, or evaluate on the original processed dataset."
            )

        if len(training_bin_uris) != checkpoint_latent_shape[0]:
            raise ValueError(
                "Preprocessing artifact BIN list length does not match checkpoint latent table. "
                f"artifact_bins={len(training_bin_uris)}, checkpoint_latent={checkpoint_latent_shape[0]}"
            )

        adapted_latent = trainer.model.latent_vec.detach().clone()
        shared_bins = 0
        for old_idx, bin_uri in enumerate(training_bin_uris):
            new_idx = trainer.bin_index.get(bin_uri)
            if new_idx is None:
                continue
            adapted_latent[new_idx] = checkpoint_latent[old_idx].to(device=adapted_latent.device)
            shared_bins += 1

        if shared_bins == 0:
            raise ValueError(
                "The evaluation dataset shares no BIN URIs with the checkpoint's training BIN set. "
                "The latent table cannot be reused without overlapping bin identities."
            )

        model_state_dict["latent_vec"] = adapted_latent.to(dtype=trainer.model.latent_vec.dtype)
        log.info(
            "Remapped checkpoint latent rows onto %d shared BINs (%d total evaluation BINs).",
            shared_bins,
            len(trainer.bin_index),
        )

    try:
        trainer.model.load_state_dict(model_state_dict)
    except RuntimeError as exc:
        raise ValueError(
            "Model weights are incompatible with reconstructed architecture. "
            "Check that checkpoint config and evaluation dataset are compatible."
        ) from exc

    _rebuild_eval_loaders(trainer)

    train_pred = trainer.get_predictions(split="train")
    val_pred = trainer.get_predictions(split="val")
    test_pred = trainer.get_predictions(split="test")
    full_pred = _concat_predictions(train_pred, val_pred, test_pred)

    train_loss = trainer.validate(split="train")
    val_loss = trainer.validate(split="val")
    test_loss = trainer.validate(split="test")

    n_train = len(trainer.train_dataset)
    n_val = len(trainer.val_dataset)
    n_test = len(trainer.test_dataset)
    denom = max(1, n_train + n_val + n_test)
    full_loss = float((train_loss * n_train + val_loss * n_val + test_loss * n_test) / denom)

    full_metrics = trainer.compute_metrics(split="test", predictions=full_pred)

    predictions, targets, sample_labels, bin_labels = full_pred
    results = {
        "model": model_name,
        "run_id": str(checkpoint.get("run_id") or trainer.run_id),
        "best_val_loss": float(checkpoint.get("best_val_loss", float("inf"))),
        "test_loss": full_loss,
        "predictions": predictions,
        "targets": targets,
        "sample_labels": sample_labels,
        "bin_labels": bin_labels,
        "latent_vector": trainer.model.latent_vec.detach().cpu().numpy(),
        "train_losses": list(checkpoint.get("train_losses", [])),
        "val_losses": list(checkpoint.get("val_losses", [])),
        "val_metrics": dict(checkpoint.get("val_metrics", {})),
        "latent_diagnostics": list(checkpoint.get("latent_diagnostics", [])),
        "test_metrics": full_metrics,
    }

    if args.output is not None:
        output_path = os.path.abspath(os.path.expanduser(args.output))
    else:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        output_path = os.path.join(
            os.path.dirname(checkpoint_path),
            f"results_eval_full_{model_name}_{timestamp}.pkl",
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(results, fh)

    log.info("Checkpoint: %s", checkpoint_path)
    log.info("Dataset: %s", cfg.data_path)
    log.info("Saved evaluation results to: %s", output_path)


if __name__ == "__main__":
    main()
