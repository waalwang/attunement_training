"""
weighted_sft_trainer.py

Extends TRL's SFTTrainer to support per-example loss weights.
Each training example carries a `weight` field that scales its
cross-entropy loss contribution.

This lets high-attunement, high-score examples contribute more to
the gradient, steering the model toward tone-matched responses
without dropping low-weight examples entirely.
"""

from __future__ import annotations

import torch
from trl import SFTTrainer


class WeightedSFTTrainer(SFTTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )

        if weights is None:
            loss = outputs.loss
        else:
            # outputs.loss is mean over all tokens. We need per-example losses
            # to apply weights. Recompute from logits.
            logits = outputs.logits
            labels = inputs["labels"]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            per_token_loss = per_token_loss.view(shift_labels.size())

            # Mask out padding (labels == -100)
            mask = shift_labels != -100
            per_example_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            weights = weights.to(per_example_loss.device, dtype=per_example_loss.dtype)
            loss = (weights * per_example_loss).sum() / weights.sum()

        return (loss, outputs) if return_outputs else loss
