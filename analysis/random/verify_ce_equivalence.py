#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from latent_solver import LatentSolver  # noqa: E402
from loss import Loss  # noqa: E402


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def build_variable_length_batch(
    batch_size: int,
    max_bins: int,
    valid_fraction: float,
    device: torch.device,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    min_bins = max(2, int(max_bins * 0.1))
    lengths = torch.randint(min_bins, max_bins + 1, (batch_size,), generator=g)

    valid_counts = torch.clamp((lengths.float() * valid_fraction).round().to(torch.int64), min=1)
    valid_counts = torch.minimum(valid_counts, lengths)

    logits = torch.full((batch_size, max_bins), float("-inf"), dtype=torch.float32, device=device)
    targets = torch.zeros((batch_size, max_bins), dtype=torch.float32, device=device)
    mask = torch.zeros((batch_size, max_bins), dtype=torch.float32, device=device)

    for i in range(batch_size):
        n_valid = int(valid_counts[i].item())
        logits_vals = torch.randn((n_valid,), generator=g, dtype=torch.float32).to(device)
        logits[i, :n_valid] = logits_vals

        target_raw = torch.rand((n_valid,), generator=g, dtype=torch.float32).to(device)
        targets[i, :n_valid] = target_raw / torch.clamp(target_raw.sum(), min=1e-12)
        mask[i, :n_valid] = 1.0

    sample_ids = torch.arange(batch_size, dtype=torch.int64, device=device).unsqueeze(1).expand(-1, max_bins)
    return logits, targets, mask, sample_ids


def mlp_ce_with_grad(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    criterion = Loss(task="cross_entropy")
    logits_mlp = logits.clone().detach().requires_grad_(True)
    loss = criterion.cross_entropy_soft_targets(logits_mlp, targets, mask)
    loss.backward()
    grad = logits_mlp.grad.detach().clone()
    return loss.detach(), grad


def latent_ce_with_grad(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    sample_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid = mask.bool()

    logits_flat = logits[valid].clone().detach().requires_grad_(True)
    targets_flat = targets[valid]
    sample_flat = sample_ids[valid]

    ce_sum, scale = LatentSolver._cross_entropy_loss(None, targets_flat, logits_flat, sample_flat)
    loss = ce_sum / scale
    loss.backward()

    grad_flat = logits_flat.grad.detach().clone()
    grad_full = torch.zeros_like(logits)
    grad_full[valid] = grad_flat
    return loss.detach(), grad_full


@dataclass
class EquivalenceResult:
    loss_mlp: float
    loss_latent: float
    abs_loss_diff: float
    max_abs_grad_diff: float
    mean_abs_grad_diff: float


def check_equivalence(
    batch_size: int,
    max_bins: int,
    valid_fraction: float,
    device: torch.device,
    seed: int,
) -> EquivalenceResult:
    logits, targets, mask, sample_ids = build_variable_length_batch(
        batch_size=batch_size,
        max_bins=max_bins,
        valid_fraction=valid_fraction,
        device=device,
        seed=seed,
    )

    loss_mlp, grad_mlp = mlp_ce_with_grad(logits, targets, mask)
    loss_latent, grad_latent = latent_ce_with_grad(logits, targets, mask, sample_ids)

    abs_loss_diff = float(torch.abs(loss_mlp - loss_latent).item())
    grad_diff = torch.abs(grad_mlp - grad_latent)[mask.bool()]

    return EquivalenceResult(
        loss_mlp=float(loss_mlp.item()),
        loss_latent=float(loss_latent.item()),
        abs_loss_diff=abs_loss_diff,
        max_abs_grad_diff=float(grad_diff.max().item()),
        mean_abs_grad_diff=float(grad_diff.mean().item()),
    )


class _DummyCfg:
    def __init__(self, include_self_in_interpolation: bool) -> None:
        self.include_self_in_interpolation = include_self_in_interpolation


class _DummySolver:
    def __init__(self, op: torch.Tensor, include_self_in_interpolation: bool) -> None:
        self.cfg = _DummyCfg(include_self_in_interpolation)
        self._op = op
        self.embed_dim = 1

    def _compute_logits_from_latent_values(self, latent_obs: torch.Tensor, intrinsic_t: torch.Tensor, final_weights_t=None) -> torch.Tensor:
        return intrinsic_t.squeeze(-1) + latent_obs.squeeze(-1)

    def get_interpolation_operator(self, include_self_in_interpolation: bool, device: torch.device | None = None) -> torch.Tensor:
        op = self._op
        target_device = device if device is not None else op.device
        return op.to(target_device)


def interpolation_path_checks(device: torch.device, seed: int) -> Dict[str, float]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 13)

    # Sparse CSR kernels are not fully supported/perf-stable on MPS; test interpolation routing on CPU.
    interp_device = torch.device("cpu") if device.type == "mps" else device

    n_bins = 12
    n_obs = 30
    latent = torch.randn((n_bins,), generator=g, dtype=torch.float32).to(interp_device)
    intrinsic = torch.randn((n_obs, 1), generator=g, dtype=torch.float32).to(interp_device)
    bin_ids = torch.randint(0, n_bins, (n_obs,), generator=g, dtype=torch.int64).to(interp_device)

    dense = torch.rand((n_bins, n_bins), generator=g, dtype=torch.float32)
    dense = dense / torch.clamp(dense.sum(dim=1, keepdim=True), min=1e-12)
    op = dense.to_sparse_csr().to(interp_device)

    solver = _DummySolver(op=op, include_self_in_interpolation=False)

    own_logits = intrinsic.squeeze(-1) + latent[bin_ids]

    mask_none = None
    out_none = LatentSolver._logits_from_latent(solver, latent, intrinsic, bin_ids, interpolation_mask=mask_none)

    mask_false = torch.zeros((n_obs,), dtype=torch.bool, device=interp_device)
    out_false = LatentSolver._logits_from_latent(solver, latent, intrinsic, bin_ids, interpolation_mask=mask_false)

    mask_mixed = torch.rand((n_obs,), generator=g, dtype=torch.float32).to(interp_device) > 0.5
    interp_full = torch.sparse.mm(op, latent.unsqueeze(-1)).squeeze(-1)
    expected_mixed = torch.where(mask_mixed, intrinsic.squeeze(-1) + interp_full[bin_ids], own_logits)
    out_mixed = LatentSolver._logits_from_latent(solver, latent, intrinsic, bin_ids, interpolation_mask=mask_mixed)

    return {
        "none_vs_own_max_abs": float(torch.max(torch.abs(out_none - own_logits)).item()),
        "false_vs_own_max_abs": float(torch.max(torch.abs(out_false - own_logits)).item()),
        "mixed_vs_expected_max_abs": float(torch.max(torch.abs(out_mixed - expected_mixed)).item()),
    }


def _benchmark_once_mlp(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    criterion = Loss(task="cross_entropy")
    logits_b = logits.clone().detach().requires_grad_(True)
    _maybe_sync(logits.device)
    t0 = time.perf_counter()
    loss = criterion.cross_entropy_soft_targets(logits_b, targets, mask)
    loss.backward()
    _maybe_sync(logits.device)
    t1 = time.perf_counter()
    return t1 - t0


def _benchmark_once_latent(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    sample_ids: torch.Tensor,
) -> float:
    valid = mask.bool()
    logits_b = logits[valid].clone().detach().requires_grad_(True)
    y_b = targets[valid]
    s_b = sample_ids[valid]

    _maybe_sync(logits.device)
    t0 = time.perf_counter()
    ce_sum, scale = LatentSolver._cross_entropy_loss(None, y_b, logits_b, s_b)
    loss = ce_sum / scale
    loss.backward()
    _maybe_sync(logits.device)
    t1 = time.perf_counter()
    return t1 - t0


def benchmark(
    shapes: Sequence[Tuple[int, int]],
    valid_fraction: float,
    repeats: int,
    warmup: int,
    device: torch.device,
    seed: int,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    for i, (bsz, max_bins) in enumerate(shapes):
        logits, targets, mask, sample_ids = build_variable_length_batch(
            batch_size=bsz,
            max_bins=max_bins,
            valid_fraction=valid_fraction,
            device=device,
            seed=seed + i,
        )

        for _ in range(warmup):
            _ = _benchmark_once_mlp(logits, targets, mask)
            _ = _benchmark_once_latent(logits, targets, mask, sample_ids)

        mlp_times = []
        latent_times = []
        for _ in range(repeats):
            mlp_times.append(_benchmark_once_mlp(logits, targets, mask))
            latent_times.append(_benchmark_once_latent(logits, targets, mask, sample_ids))

        mlp_med = statistics.median(mlp_times)
        latent_med = statistics.median(latent_times)
        speed_ratio = latent_med / mlp_med if mlp_med > 0 else float("inf")

        rows.append(
            {
                "batch_size": float(bsz),
                "max_bins": float(max_bins),
                "mlp_median_ms": 1000.0 * mlp_med,
                "latent_median_ms": 1000.0 * latent_med,
                "latent_over_mlp": speed_ratio,
            }
        )

    return rows


def print_equivalence(results: Iterable[EquivalenceResult]) -> None:
    print("=== Numerical Equivalence (MLP CE vs Latent CE) ===")
    for idx, r in enumerate(results):
        print(
            f"case={idx} loss_mlp={r.loss_mlp:.8f} loss_latent={r.loss_latent:.8f} "
            f"abs_loss_diff={r.abs_loss_diff:.3e} max_abs_grad_diff={r.max_abs_grad_diff:.3e} "
            f"mean_abs_grad_diff={r.mean_abs_grad_diff:.3e}"
        )


def print_interpolation(results: Dict[str, float]) -> None:
    print("=== Interpolation Path Checks (_logits_from_latent) ===")
    for k, v in results.items():
        print(f"{k}={v:.3e}")


def print_bench(rows: Sequence[Dict[str, float]]) -> None:
    print("=== Micro-benchmark (forward+backward) ===")
    print("batch_size max_bins mlp_median_ms latent_median_ms latent_over_mlp")
    for r in rows:
        print(
            f"{int(r['batch_size']):>10d} {int(r['max_bins']):>8d} "
            f"{r['mlp_median_ms']:>13.3f} {r['latent_median_ms']:>16.3f} {r['latent_over_mlp']:>14.3f}"
        )


def parse_shapes(text: str) -> List[Tuple[int, int]]:
    shapes: List[Tuple[int, int]] = []
    for item in text.split(","):
        b, m = item.split("x")
        shapes.append((int(b), int(m)))
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify cross-entropy parity and efficiency.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--valid-fraction", type=float, default=0.7)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--shapes", type=str, default="32x128,64x256,128x512")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")

    device = torch.device(args.device)
    shapes = parse_shapes(args.shapes)

    eq_results = [
        check_equivalence(b, m, args.valid_fraction, device=device, seed=args.seed + i)
        for i, (b, m) in enumerate(shapes)
    ]
    interp_result = interpolation_path_checks(device=device, seed=args.seed)
    bench_rows = benchmark(
        shapes=shapes,
        valid_fraction=args.valid_fraction,
        repeats=args.repeats,
        warmup=args.warmup,
        device=device,
        seed=args.seed,
    )

    print_equivalence(eq_results)
    print_interpolation(interp_result)
    print_bench(bench_rows)


if __name__ == "__main__":
    main()
