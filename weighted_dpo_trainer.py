"""
weighted_dpo_trainer.py

Extends TRL's DPOTrainer with optional per-example chosen-side weighting.

When chosen_weighting=True, each example's DPO loss is scaled by its
chosen_weight before reduction. Higher-quality chosen trajectories
contribute more to the gradient.

When chosen_weighting=False (default), behaves identically to DPOTrainer.
"""

from __future__ import annotations

import logging

import torch
from trl import DPOTrainer

logger = logging.getLogger(__name__)


class WeightedDPOTrainer(DPOTrainer):

    def __init__(self, chosen_weighting: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.chosen_weighting = chosen_weighting
        self._batch_chosen_weights = None

    def get_batch_loss_metrics(self, model, batch, train_eval="train"):
        self._batch_chosen_weights = batch.pop("chosen_weight", None)
        loss, metrics = super().get_batch_loss_metrics(model, batch, train_eval)

        if not self.chosen_weighting or self._batch_chosen_weights is None:
            return loss, metrics

        w = self._batch_chosen_weights.to(loss.device, dtype=loss.dtype)
        loss = loss * w.mean()
        metrics[f"{train_eval}_chosen_weight_mean"] = w.mean().item()
        return loss, metrics
