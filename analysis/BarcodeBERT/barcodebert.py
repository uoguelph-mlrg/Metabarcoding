"""
BarcodeBERT embedding utility.

Provides the BarcodeBERTEmbedder class, which loads the bioscan-ml/BarcodeBERT
model from HuggingFace and computes mean-pooled sequence embeddings.

Mean pooling (recommended by the BarcodeBERT authors) averages the last hidden
state over all non-padding token positions, producing a fixed-size vector for
each input sequence regardless of its length.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class BarcodeBERTEmbedder:
    """
    Wrapper around the BarcodeBERT transformer model.

    Usage
    -----
    embedder = BarcodeBERTEmbedder()
    embedder.load()
    embeddings = embedder.embed(["ATCG...", "GCTA..."])  # np.ndarray [N, 768]
    """

    MODEL_NAME = "bioscan-ml/BarcodeBERT"

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 512,
    ) -> None:
        """
        Args:
            device: torch device string ("cpu", "cuda", "mps"). Auto-detected if None.
            batch_size: Number of sequences per inference batch.
            max_length: Maximum token length passed to the tokenizer.
        """
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

        import torch
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

    def load(self) -> "BarcodeBERTEmbedder":
        """
        Download (or load from HuggingFace cache) and initialise the model.
        Returns self for chaining: embedder = BarcodeBERTEmbedder().load()
        """
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError as exc:
            raise ImportError(
                "The 'transformers' package is required for BarcodeBERT inference. "
                "Install it with: pip install transformers"
            ) from exc

        import logging
        logging.getLogger("transformers").setLevel(logging.WARNING)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True
        )
        self._model = (
            AutoModel.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
            .to(self.device)
            .eval()
        )
        return self

    def embed(self, sequences: List[str]) -> np.ndarray:
        """
        Compute mean-pooled embeddings for a list of DNA sequences.

        The BarcodeBERT KmerTokenizer processes ONE sequence at a time and pads
        all sequences to max_len=660, producing a fixed number of tokens (165).
        We encode each sequence individually, then batch the resulting tensors
        for efficient GPU/MPS inference.

        Args:
            sequences: List of DNA barcode strings (IUPAC characters allowed).

        Returns:
            np.ndarray of shape [len(sequences), hidden_dim] (float32).
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Call .load() before .embed()")

        import torch

        all_embeddings: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                batch = sequences[start : start + self.batch_size]

                # KmerTokenizer is single-sequence: encode each one individually
                # then stack — safe because padding makes all outputs the same length
                batch_input_ids: List[List[int]] = []
                batch_attention_mask: List[List[int]] = []
                for seq in batch:
                    encoded = self._tokenizer(seq, padding=True)
                    batch_input_ids.append(encoded["input_ids"])
                    batch_attention_mask.append(encoded["attention_mask"])

                input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(self.device)
                attention_mask = torch.tensor(batch_attention_mask, dtype=torch.long).to(self.device)

                outputs = self._model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

                # Mean-pool over non-padding token positions
                # last_hidden_state: [B, seq_len, hidden_dim]
                last_hidden = outputs.last_hidden_state
                mask_exp = attention_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
                sum_hidden = (last_hidden * mask_exp).sum(dim=1)  # [B, hidden_dim]
                count = mask_exp.sum(dim=1).clamp(min=1e-9)       # [B, 1]
                mean_pooled = (sum_hidden / count).cpu().numpy()   # [B, hidden_dim]

                all_embeddings.append(mean_pooled.astype(np.float32))

        return np.vstack(all_embeddings)

    def embed_dict(self, bin_seq_dict: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Embed a dict mapping bin_uri -> sequence and return a dict
        mapping bin_uri -> embedding vector.

        Args:
            bin_seq_dict: {bin_uri: sequence}

        Returns:
            {bin_uri: np.ndarray of shape [hidden_dim]}
        """
        uris = list(bin_seq_dict.keys())
        seqs = [bin_seq_dict[u] for u in uris]
        embeddings = self.embed(seqs)
        return {uri: emb for uri, emb in zip(uris, embeddings)}
