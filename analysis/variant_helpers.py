from __future__ import annotations

import os
import pickle
import re
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional


def _sanitize_token(token: str) -> str:
    token = token.strip()
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token)
    token = token.strip("_")
    return token or "variant"


def make_run_group(analysis_name: str, timestamp: Optional[str] = None) -> str:
    ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
    return f"{_sanitize_token(analysis_name)}_{ts}"


def make_variant_run_name(
    analysis_name: str,
    variant_name: str,
    timestamp: Optional[str] = None,
) -> str:
    ts = timestamp or time.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{_sanitize_token(analysis_name)}_{_sanitize_token(variant_name)}_{ts}"


def make_variant_filename(analysis_name: str, variant_name: str) -> str:
    return f"{_sanitize_token(analysis_name)}_{_sanitize_token(variant_name)}.pkl"


def make_output_dir(script_file: str, output_dir_arg: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(script_file))
    output_dir = os.path.join(script_dir, output_dir_arg)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_variant_result(
    output_dir: str,
    analysis_name: str,
    variant_name: str,
    result: Dict[str, Any],
) -> str:
    output_path = os.path.join(output_dir, make_variant_filename(analysis_name, variant_name))
    with open(output_path, "wb") as f:
        pickle.dump({variant_name: result}, f)
    return output_path


@contextmanager
def variant_wandb_run(
    *,
    use_wandb: bool,
    wandb_module: Any,
    analysis_name: str,
    variant_name: str,
    run_group: Optional[str],
    tags: Optional[list[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Iterator[None]:
    if use_wandb:
        wandb_module.init(
            project="metabarcoding",
            name=make_variant_run_name(analysis_name, variant_name),
            group=run_group,
            tags=tags or [analysis_name, variant_name, "variant_only"],
            config=config,
            reinit=True,
        )
    try:
        yield
    finally:
        if use_wandb:
            wandb_module.finish()
