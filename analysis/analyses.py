"""
Single source of truth for all analysis definitions.

Each analysis declares:
  - Per-variant: config overrides, display label, hex color, SLURM walltime
  - Per-analysis: subset_top_n, latent_concat_keys, baseline_label/color
  - Optional hooks: add_cli_args, make_variants, apply_args_to_cfg
    (for dynamic analyses whose variant list depends on CLI args)
  - run_script: path to the legacy training script (relative to analysis/),
    set only for analyses that cannot use run_analysis.py

Used by:
  run_analysis.py          — training entry point (generic analyses)
  submit_subanalysis.sh    — SLURM job submission (one job per variant)
  run_all_visualizations.py — visualization runner
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Baseline display config (fallback defaults)
# ---------------------------------------------------------------------------

BASELINE_LABEL = "MLP + Latent (Baseline)"
BASELINE_COLOR = "#95a5a6"

_ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    name: str
    config: Dict[str, Any]          # config keys/values to override from default Config()
    label: str                       # display label for plots
    color: str                       # hex color for plots
    time: str                        # SLURM walltime, e.g. "0:45:00"


@dataclass
class Analysis:
    name: str
    variants: List[Variant]
    subset_top_n: Optional[int] = None
    latent_concat_keys: Optional[List[str]] = None
    baseline_label: str = BASELINE_LABEL   # display label for the shared baseline in plots
    baseline_color: str = BASELINE_COLOR   # hex color for the shared baseline in plots
    include_baseline: bool = True          # whether to include the shared baseline in visualizations
    # Path to the legacy training script (relative to analysis/), for analyses
    # that cannot use run_analysis.py. submit_subanalysis.sh uses this when set.
    run_script: Optional[str] = None
    # Optional hooks for dynamic analyses (variant list depends on CLI args)
    add_cli_args: Optional[Callable] = field(default=None, repr=False)
    make_variants: Optional[Callable] = field(default=None, repr=False)
    apply_args_to_cfg: Optional[Callable] = field(default=None, repr=False)

    @property
    def DEFAULT_VARIANTS(self) -> List[Dict[str, Any]]:
        """Compatibility shim used by run_analysis / variant_helpers."""
        return [{"name": v.name, "config": v.config} for v in self.variants]

    @property
    def ANALYSIS_NAME(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, Analysis] = {}


def _reg(analysis: Analysis) -> Analysis:
    REGISTRY[analysis.name] = analysis
    return analysis


# ---------------------------------------------------------------------------
# Barcode embeddings
# ---------------------------------------------------------------------------

_reg(Analysis(
    name="barcode_embeddings",
    variants=[
        Variant(
            name="taxonomy_distance",
            config={"use_taxonomy": True, "use_embedding": False},
            label="Taxonomic Distance",
            color="#9b59b6",
            time="0:45:00",
        ),
        Variant(
            name="emb_emmabhl_finetuned",
            config={
                "use_taxonomy": False,
                "use_embedding": True,
                "barcode_hf_model": "emmabhl/BarcodeBERT_finetuned",
            },
            label="emmabhl/BarcodeBERT_finetuned",
            color="#3498db",
            time="0:45:00",
        ),
        Variant(
            name="emb_bioscan_barcodebert",
            config={
                "use_taxonomy": False,
                "use_embedding": True,
                "barcode_hf_model": "bioscan-ml/BarcodeBERT",
            },
            label="bioscan-ml/BarcodeBERT",
            color="#2ecc71",
            time="0:45:00",
        ),
        Variant(
            name="emb_bioscan_barcodemamba",
            config={
                "use_taxonomy": False,
                "use_embedding": True,
                "barcode_hf_model": "bioscan-ml/BarcodeMamba",
            },
            label="bioscan-ml/BarcodeMamba",
            color="#e67e22",
            time="0:45:00",
        ),
    ],
    include_baseline=False,
))


# ---------------------------------------------------------------------------
# Loss comparison
# ---------------------------------------------------------------------------

_reg(Analysis(
    name="loss_comparison",
    variants=[
        Variant(
            name="logistic",
            config={"loss_type": "logistic", "latent_present_only": True},
            label="Logistic (BCE)",
            color="#9b59b6",
            time="0:45:00",
        ),
    ],
    baseline_label="Cross-Entropy (CE)",
    baseline_color="#e74c3c",
))


# ---------------------------------------------------------------------------
# Interpolated latent
# ---------------------------------------------------------------------------

_reg(Analysis(
    name="interpolated_latent",
    variants=[
        Variant(
            name="default_with_interpolation",
            config={
                "interpolated_sample_fraction": 0.2,
                "train_MLP_with_interpolation": True,
                "inference_with_interpolation": False,
                "include_self_in_interpolation": True,
            },
            label="Interpolation (20%)",
            color="#e74c3c",
            time="0:45:00",
        ),
        Variant(
            name="include_self_false",
            config={
                "interpolated_sample_fraction": 0.2,
                "train_MLP_with_interpolation": True,
                "inference_with_interpolation": False,
                "include_self_in_interpolation": False,
            },
            label="Interpolation (20%, no self latent)",
            color="#e67e22",
            time="0:45:00",
        ),
        Variant(
            name="inference_true",
            config={
                "interpolated_sample_fraction": 0.2,
                "train_MLP_with_interpolation": True,
                "inference_with_interpolation": True,
                "include_self_in_interpolation": True,
            },
            label="Interpolation (20%, at inference)",
            color="#f39c12",
            time="0:45:00",
        ),
        Variant(
            name="train_mlp_false",
            config={
                "interpolated_sample_fraction": 0.2,
                "train_MLP_with_interpolation": False,
                "inference_with_interpolation": False,
                "include_self_in_interpolation": True,
            },
            label="Interpolation (20%, no MLP interpolation)",
            color="#2ecc71",
            time="0:45:00",
        ),
        Variant(
            name="fraction_0p1",
            config={
                "interpolated_sample_fraction": 0.1,
                "train_MLP_with_interpolation": True,
                "inference_with_interpolation": False,
                "include_self_in_interpolation": True,
            },
            label="Interpolation (10%)",
            color="#3498db",
            time="0:45:00",
        ),
        Variant(
            name="fraction_0p5",
            config={
                "interpolated_sample_fraction": 0.5,
                "train_MLP_with_interpolation": True,
                "inference_with_interpolation": False,
                "include_self_in_interpolation": True,
            },
            label="Interpolation (50%)",
            color="#9b59b6",
            time="0:45:00",
        ),
        Variant(
            name="fraction_1p0",
            config={
                "interpolated_sample_fraction": 1.0,
                "train_MLP_with_interpolation": True,
                "inference_with_interpolation": False,
                "include_self_in_interpolation": True,
            },
            label="Interpolation (100%)",
            color="#1abc9c",
            time="0:45:00",
        ),
    ],
    baseline_label="No Interpolation",
    baseline_color="#95a5a6",
))


# ---------------------------------------------------------------------------
# Optimal K
# ---------------------------------------------------------------------------

_optimal_K_default_variants = [
    Variant(name="K_5",   config={"K": 5},   label="K=5",   color="#5c2f00", time="0:45:00"),
    Variant(name="K_100", config={"K": 100}, label="K=100", color="#e89f2e", time="1:30:00"),
    Variant(name="K_500", config={"K": 500}, label="K=500", color="#f5c842", time="3:00:00"),
]


def _optimal_K_add_cli_args(parser) -> None:
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=None,
        help="K values to run (default: 5, 100, 500)",
    )


def _optimal_K_make_variants(args) -> List[Dict[str, Any]]:
    if not args.k_values:
        return [{"name": v.name, "config": v.config} for v in _optimal_K_default_variants]
    return [{"name": f"K_{k}", "config": {"K": k}} for k in sorted(args.k_values)]


_reg(Analysis(
    name="optimal_K",
    variants=_optimal_K_default_variants,
    baseline_label="K=25 (default)",
    baseline_color="#95a5a6",
    add_cli_args=_optimal_K_add_cli_args,
    make_variants=_optimal_K_make_variants,
))


# ---------------------------------------------------------------------------
# Dimensionality — vector size
# The default embed_dim=10 is the baseline config, so dim_10 is excluded
# from variants to avoid training the same config twice.
# include_baseline=False: the baseline is the default run itself (dim=10),
# already present as a training artifact — no separate comparison makes sense.
# ---------------------------------------------------------------------------

_dim_vector_default_variants = [
    Variant(name="dim_1",  config={"embed_dim": 1},  label="Dim=1",  color="#95a5a6", time="0:45:00"),
    Variant(name="dim_2",  config={"embed_dim": 2},  label="Dim=2",  color="#824e05", time="0:45:00"),
    Variant(name="dim_5",  config={"embed_dim": 5},  label="Dim=5",  color="#e74c3c", time="0:45:00"),
    Variant(name="dim_6",  config={"embed_dim": 6},  label="Dim=6",  color="#e67e22", time="0:45:00"),
    Variant(name="dim_8",  config={"embed_dim": 8},  label="Dim=8",  color="#f39c12", time="0:45:00"),
    # dim_10 omitted: it is identical to the baseline config (embed_dim=10)
    Variant(name="dim_12", config={"embed_dim": 12}, label="Dim=12", color="#a2f10f", time="0:45:00"),
    Variant(name="dim_15", config={"embed_dim": 15}, label="Dim=15", color="#2ecc71", time="0:45:00"),
    Variant(name="dim_20", config={"embed_dim": 20}, label="Dim=20", color="#1d8d4b", time="0:45:00"),
    Variant(name="dim_50", config={"embed_dim": 50}, label="Dim=50", color="#3498db", time="0:45:00"),
]


def _dim_vector_add_cli_args(parser) -> None:
    parser.add_argument(
        "--dimensions", type=int, nargs="+", default=None,
        help="Embedding dimensions to run (default: 1 2 5 6 8 12 15 20 50)",
    )


def _dim_vector_make_variants(args) -> List[Dict[str, Any]]:
    if not args.dimensions:
        return [{"name": v.name, "config": v.config} for v in _dim_vector_default_variants]
    return [{"name": f"dim_{d}", "config": {"embed_dim": d}} for d in sorted(args.dimensions)]


_reg(Analysis(
    name="dimensionality_vector",
    variants=_dim_vector_default_variants,
    baseline_label="Dim=10",
    baseline_color="#f1c40f",
    add_cli_args=_dim_vector_add_cli_args,
    make_variants=_dim_vector_make_variants,
))


# ---------------------------------------------------------------------------
# Dimensionality — gating function
# The default gating_fn=sigmoid is the baseline config, so it is excluded
# from variants to avoid training the same config twice.
# ---------------------------------------------------------------------------

_dim_gating_default_variants = [
    Variant(name="exp",         config={"embed_dim": 10, "gating_fn": "exp"},         label="Exponential",        color="#e74c3c", time="0:45:00"),
    Variant(name="scaled_exp",  config={"embed_dim": 10, "gating_fn": "scaled_exp"},  label="Scaled Exponential", color="#e67e22", time="0:45:00"),
    Variant(name="additive",    config={"embed_dim": 10, "gating_fn": "additive"},    label="Additive (1+h)",     color="#f39c12", time="0:45:00"),
    Variant(name="softplus",    config={"embed_dim": 10, "gating_fn": "softplus"},    label="Softplus",           color="#2ecc71", time="0:45:00"),
    Variant(name="tanh",        config={"embed_dim": 10, "gating_fn": "tanh"},        label="Tanh",               color="#3498db", time="0:45:00"),
    Variant(name="dot_product", config={"embed_dim": 10, "gating_fn": "dot_product"}, label="Dot Product",        color="#1abc9c", time="0:45:00"),
    # sigmoid omitted: it is identical to the baseline config (gating_fn=sigmoid)
]

_GATING_FUNCTION_NAMES = [v.name for v in _dim_gating_default_variants]


def _dim_gating_add_cli_args(parser) -> None:
    parser.add_argument(
        "--gating_functions", nargs="+", default=None,
        choices=_GATING_FUNCTION_NAMES,
        help="Gating functions to run (default: all except sigmoid which is the baseline)",
    )


def _dim_gating_make_variants(args) -> List[Dict[str, Any]]:
    if not args.gating_functions:
        return [{"name": v.name, "config": v.config} for v in _dim_gating_default_variants]
    return [{"name": v.name, "config": v.config} for v in _dim_gating_default_variants
            if v.name in args.gating_functions]


_reg(Analysis(
    name="dimensionality_gating",
    variants=_dim_gating_default_variants,
    baseline_label="Sigmoid",
    baseline_color="#9b59b6",
    add_cli_args=_dim_gating_add_cli_args,
    make_variants=_dim_gating_make_variants,
))


# ---------------------------------------------------------------------------
# Location embedding
# ---------------------------------------------------------------------------

_EMBEDDER_NAMES = ["satclip", "range", "geoclip", "alphaearth"]

_loc_emb_default_variants = [
    Variant(name="satclip",    config={"location_embedder": "satclip"}, label="SatCLIP (256D)",   color="#e74c3c", time="0:45:00"),
    Variant(name="range",      config={"location_embedder": "range"}, label="RANGE (1280D)",    color="#3498db", time="0:45:00"),
    Variant(name="geoclip",    config={"location_embedder": "geoclip"}, label="GeoCLIP (512D)",   color="#2ecc71", time="0:45:00"),
    Variant(name="alphaearth", config={"location_embedder": "alphaearth"}, label="AlphaEarth (64D)", color="#f39c12", time="0:45:00"),
]


def _loc_emb_add_cli_args(parser) -> None:
    parser.add_argument(
        "--embedders", nargs="+", default=None,
        choices=_EMBEDDER_NAMES,
        help="Location embedding variants to run (default: all)",
    )
    parser.add_argument("--satclip_ckpt_path", type=str, default=None,
                        help="Path to SatCLIP checkpoint (.pth)")
    parser.add_argument("--range_db_path", type=str, default=None,
                        help="Path to RANGE database (.npz)")
    parser.add_argument("--keep_raw_gps", action="store_true",
                        help="Keep raw lat/lon alongside location embeddings")


def _loc_emb_make_variants(args) -> List[Dict[str, Any]]:
    if not args.embedders:
        return [{"name": v.name, "config": v.config} for v in _loc_emb_default_variants]
    return [{"name": v.name, "config": v.config} for v in _loc_emb_default_variants
            if v.name in args.embedders]


def _loc_emb_apply_args_to_cfg(cfg, args) -> None:
    cfg.satclip_ckpt_path = args.satclip_ckpt_path
    cfg.range_db_path = args.range_db_path
    cfg.keep_raw_gps_features = args.keep_raw_gps


_reg(Analysis(
    name="location_embedding",
    variants=_loc_emb_default_variants,
    baseline_label="Raw GPS (lat/lon)",
    baseline_color="#95a5a6",
    add_cli_args=_loc_emb_add_cli_args,
    make_variants=_loc_emb_make_variants,
    apply_args_to_cfg=_loc_emb_apply_args_to_cfg,
))


# ---------------------------------------------------------------------------
# CE with present BIN only  (uses src Trainer — generic runner)
# Compare latent update on all BINs vs. present-only BINs
# ---------------------------------------------------------------------------

_reg(Analysis(
    name="CE_with_present_BIN_only",
    variants=[
        Variant(
            name="present_only",
            config={"latent_present_only": True},
            label="Present BINs only",
            color="#e74c3c",
            time="0:45:00",
        )
    ],
    baseline_label="All BINs",
    baseline_color="#3498db",
))


# ---------------------------------------------------------------------------
# Preprocessing  (uses src Trainer — generic runner with preprocessed_dir override)
# ---------------------------------------------------------------------------

_PREPROCESSING_DATA = os.path.join(_ANALYSIS_DIR, "preprocessing", "data")

_reg(Analysis(
    name="preprocessing",
    variants=[
        Variant(
            name="normalized",
            config={"preprocessed_dir": os.path.join(_PREPROCESSING_DATA, "normalized")},
            label="Normalized",
            color="#1f77b4",
            time="0:45:00",
        ),
        Variant(
            name="original",
            config={"preprocessed_dir": os.path.join(_PREPROCESSING_DATA, "original")},
            label="Raw counts",
            color="#2ca02c",
            time="0:45:00",
        ),
    ],
    baseline_label="Logarithm",
    baseline_color="#ff7f0e",
))


# ---------------------------------------------------------------------------
# Ablation study  (custom MLPOnlyTrainer — legacy script)
# ---------------------------------------------------------------------------

_reg(Analysis(
    name="ablation_study",
    variants=[
        Variant(
            name="mlp_no_taxonomy",
            config={},
            label="MLP (no taxonomy)",
            color="#1f77b4",
            time="0:25:00",
        ),
        Variant(
            name="mlp_with_taxonomy",
            config={},
            label="MLP (with taxonomy)",
            color="#2ca02c",
            time="0:25:00",
        ),
    ],
    baseline_label="MLP + Latent",
    baseline_color="#ff7f0e",
    run_script="ablation_study/ablation_study.py",
))


# ---------------------------------------------------------------------------
# Latent as input and output  (local custom Trainer — legacy script)
# ---------------------------------------------------------------------------

_reg(Analysis(
    name="latent_as_in_and_output",
    variants=[
        Variant(
            name="both_dim_5",
            config={"latent_input_dim": 5, "embed_dim": 5},
            label="Latent In+Out (dim=5)",
            color="#e74c3c",
            time="0:45:00",
        ),
        Variant(
            name="input_only_dim_10",
            config={"latent_input_dim": 10, "embed_dim": 0},
            label="Latent Input Only (dim=10)",
            color="#3498db",
            time="0:45:00",
        ),
    ],
    baseline_label="Latent + MLP (Baseline)",
    baseline_color="#95a5a6",
    latent_concat_keys=["both_dim_5"],
    run_script="latent_as_in_and_output/latent_as_in_and_output.py",
))


# ---------------------------------------------------------------------------
# Baselines comparison  (sklearn classical models — legacy script)
# ---------------------------------------------------------------------------

_reg(Analysis(
    name="baselines_comparison",
    variants=[
        Variant(name="mean",               config={}, label="Mean",               color="#808080", time="0:02:00"),
        Variant(name="zero",               config={}, label="Zero",               color="#4d4d4d", time="0:02:00"),
        Variant(name="linear_regression",  config={}, label="Linear Regression",  color="#f28e2b", time="0:02:00"),
        Variant(name="ridge",              config={}, label="Ridge",              color="#4e79a7", time="0:02:00"),
        Variant(name="elasticnet",         config={}, label="ElasticNet",         color="#e15759", time="0:02:00"),
        Variant(name="decision_tree",      config={}, label="Decision Tree",      color="#76b7b2", time="0:02:00"),
        Variant(name="random_forest",      config={}, label="Random Forest",      color="#59a14f", time="0:02:00"),
        Variant(name="gradient_boosting",  config={}, label="Gradient Boosting",  color="#edc948", time="0:02:00"),
        Variant(name="knn",                config={}, label="KNN",                color="#b07aa1", time="0:02:00"),
        Variant(name="two_stage",          config={}, label="Two-Stage",          color="#9c755f", time="0:02:00"),
        Variant(name="zero_inflated_ridge",config={}, label="Zero-Inflated Ridge",color="#bab0ab", time="0:02:00"),
        Variant(name="tweedie",            config={}, label="Tweedie",            color="#ff9da7", time="0:02:00"),
        Variant(name="log_transform",      config={}, label="Log-Transform",      color="#8cd17d", time="0:02:00"),
        Variant(name="quantile_rf",        config={}, label="Quantile RF",        color="#af7aa1", time="0:02:00"),
    ],
    subset_top_n=6,
    baseline_label="MLP + Latent (Baseline)",
    baseline_color="#e74c3c",
    run_script="baselines/run_baselines.py",
))
