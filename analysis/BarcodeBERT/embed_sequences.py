#!/usr/bin/env python
"""
Precompute BarcodeBERT embeddings for all BINs in a barcode data file.

This script should be run once before training with embedding-based neighbor
graphs.  The output is a .npy file containing a dict {bin_uri: embedding_vector}
that can be passed to cfg.embedding_path.

Usage
-----
python embed_sequences.py \\
    --barcode_data  ../../data/barcode_data.tsv \\
    --output        ../../data/barcodebert_embeddings.npy \\
    --batch_size    64

The output file can then be used in training:
    cfg.use_taxonomy = False
    cfg.use_embedding = True
    cfg.embedding_path = "../../data/barcodebert_embeddings.npy"
"""

from __future__ import annotations

import argparse
import logging as log
import os
import sys
import time

import numpy as np
import pandas as pd

# Allow importing barcodebert.py from the same folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Metabarcoding.analysis.BarcodeBERT.wrapper import BarcodeBERTEmbedder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_bin_sequences(barcode_data_path: str) -> dict:
    """
    Load one consensus sequence per BIN from barcode_data_path.

    Expected columns: 'bin_uri', 'seq'.
    When multiple rows share the same bin_uri the first row is used (i.e. the
    data is assumed to already carry a consensus sequence per BIN).

    Returns
    -------
    dict: {bin_uri: sequence_string}
    """
    sep = "\t" if barcode_data_path.endswith(".tsv") else ","
    df = pd.read_csv(barcode_data_path, sep=sep)

    required = {"bin_uri", "seq"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"barcode_data file is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # One consensus seq per BIN
    df_dedup = df.drop_duplicates(subset="bin_uri", keep="first")
    n_total = len(df)
    n_bins = len(df_dedup)
    if n_bins < n_total:
        log.info(f"Deduplicated {n_total} rows → {n_bins} unique BINs.")

    # Drop rows with missing sequences
    df_dedup = df_dedup.dropna(subset=["seq"])
    df_dedup = df_dedup[df_dedup["seq"].str.strip() != ""]
    n_valid = len(df_dedup)
    if n_valid < n_bins:
        log.warning(f"{n_bins - n_valid} BINs have missing/empty sequences and will be skipped.")

    return dict(zip(df_dedup["bin_uri"], df_dedup["seq"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute BarcodeBERT embeddings, grouped by BIN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--barcode_data",
        type=str,
        required=True,
        help="Path to TSV/CSV with 'bin_uri' and 'seq' columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npy file to save {bin_uri: embedding} dict.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Sequences per inference batch.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token length for BarcodeBERT.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (cpu / cuda / mps). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    log.basicConfig(
        level=log.DEBUG if args.verbose else log.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # ------------------------------------------------------------------
    # 1. Load sequences
    # ------------------------------------------------------------------
    log.info(f"Loading barcode data from {args.barcode_data} ...")
    bin_seqs = load_bin_sequences(args.barcode_data)
    log.info(f"Found {len(bin_seqs)} BINs with sequences.")

    # ------------------------------------------------------------------
    # 2. Run BarcodeBERT inference
    # ------------------------------------------------------------------
    embedder = BarcodeBERTEmbedder(
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    log.info("Loading BarcodeBERT model ...")
    embedder.load()

    t0 = time.time()
    log.info("Running inference ...")
    emb_dict = embedder.embed_dict(bin_seqs)
    elapsed = time.time() - t0
    log.info(
        f"Inference complete: {len(emb_dict)} embeddings in {elapsed:.1f}s "
        f"({elapsed / max(1, len(emb_dict)) * 1000:.1f} ms/BIN)."
    )

    # ------------------------------------------------------------------
    # 3. Save embeddings
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    np.save(args.output, emb_dict)
    log.info(f"Embeddings saved to {args.output}  (keys: {len(emb_dict)} BINs)")

    # Quick sanity check
    sample_uri = next(iter(emb_dict))
    emb_dim = emb_dict[sample_uri].shape[0]
    log.info(f"Embedding dimension: {emb_dim}  (sample: {sample_uri})")


if __name__ == "__main__":
    main()
