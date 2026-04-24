"""
weighted_sft_trainer.py

Extends TRL's SFTTrainer to support per-example loss weights.
Each training example carries a `weight` field that scales its
cross-entropy loss contribution.
"""

from __future__ import annotations

import torch
from trl import SFTTrainer


class WeightedSFTTrainer(SFTTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acc_correct = 0
        self._acc_total = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        weights = inputs.pop("weight", None)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )

        logits = outputs.logits
        labels = inputs["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != -100

        if weights is None:
            loss = outputs.loss
        else:
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            per_token_loss = per_token_loss.view(shift_labels.size())

            per_example_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            weights = weights.to(per_example_loss.device, dtype=per_example_loss.dtype)
            loss = (weights * per_example_loss).sum() / weights.sum()

        with torch.no_grad():
            preds = shift_logits.argmax(dim=-1)
            correct = ((preds == shift_labels) & mask).sum().item()
            total = mask.sum().item()

        self._acc_correct += correct
        self._acc_total += total

        return (loss, outputs) if return_outputs else loss

    def _flush_accuracy(self):
        if self._acc_total > 0:
            acc = self._acc_correct / self._acc_total
            self._acc_correct = 0
            self._acc_total = 0
            return acc
        return None

    def log(self, logs, *args, **kwargs):
        acc = self._flush_accuracy()
        if acc is not None:
            logs["accuracy"] = acc
        super().log(logs, *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        self._acc_correct = 0
        self._acc_total = 0
        output = super().evaluate(*args, **kwargs)
        acc = self._flush_accuracy()
        if acc is not None:
            output["eval_accuracy"] = acc
        return output
