from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss:
    """
    Loss function for metabarcoding abundance prediction.
    
    Two modes:
    - "cross_entropy": Sample-level distributional loss.
        - Input: logits [batch_size, n_bins] (raw model outputs before softmax)
        - Target: relative abundances [batch_size, n_bins] (probability distribution, sums to 1)
        - Computes: -sum(target * log_softmax(logits)) per sample, averaged over batch
        - This is the KL divergence up to a constant (entropy of target).

    - "logistic": Bin-level independent loss (BCE with logits).
        - Input: logits [batch_size] (raw model outputs before sigmoid)  
        - Target: relative abundances [batch_size] in [0, 1]
        - Computes: BCEWithLogitsLoss treating each bin independently
        - Note: Despite the name, this is NOT binary classification - it's used
        for regression where targets are continuous in [0, 1].
    """
    
    def __init__(self, task: Literal["cross_entropy", "logistic"] = "cross_entropy"):
        self.task = task
        if task == "cross_entropy":
            # No additional criterion needed, will compute directly in cross_entropy_soft_targets
            pass
        elif task == "logistic":
            # BCEWithLogitsLoss accepts continuous targets in [0,1]: the loss
            # −[y·log σ(z) + (1−y)·log(1−σ(z))] is a valid cross-entropy over
            # a Bernoulli whose probability is y, so it is appropriate even when
            # targets are fractional relative abundances rather than hard 0/1 labels.
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown task {task}")
    
    def cross_entropy_soft_targets(
        self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-entropy loss for soft targets (probability distributions).
        
        Args:
            logits: Raw model outputs (before softmax) of shape [batch_size, n_bins_in_sample]
                Padded positions should have value -inf (will become 0 after softmax)
            targets: Target probability distributions of shape [batch_size, n_bins_in_sample]
                Must sum to 1 along last dim. Padded positions should be 0.
            mask: Optional mask of shape [batch_size, n_bins_in_sample]. 
                1 for valid positions, 0 for padded. If None, inferred from logits.
        
        Returns:
            Scalar loss averaged over the batch.
        """
        # Squeeze middle dimension if present (from dataset's expand_dims)
        if logits.dim() == 3 and logits.size(1) == 1:
            logits = logits.squeeze(1)  # [batch_size, n_bins]
        if targets.dim() == 3 and targets.size(1) == 1:
            targets = targets.squeeze(1)  # [batch_size, n_bins]
        
        # Infer mask from logits if not provided (padded positions have -inf)
        if mask is None:
            mask = (logits > float('-inf')).float()
        
        # Compute log-softmax of logits (numerically stable)
        log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, n_bins]
        
        # Cross-entropy: -sum(target * log_prob) per sample
        log_probs_safe = torch.where(
            mask.bool(),
            log_probs,
            torch.zeros_like(log_probs)
        )
        
        loss_per_sample = -torch.sum(targets * log_probs_safe, dim=-1)  # [batch_size]
        
        # Average over batch
        return loss_per_sample.mean()

    def __call__(
        self, outputs: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.task == "cross_entropy":
            return self.cross_entropy_soft_targets(outputs, targets, mask)
        else:
            return self.criterion(outputs, targets)
