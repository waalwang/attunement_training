"""
weighted_sft_trainer.py

Extends TRL's SFTTrainer to support per-example loss weights.
Each training example carries a `weight` field that scales its
cross-entropy loss contribution.
"""

from __future__ import annotations

import logging
import os
import shutil

import torch
from transformers.trainer_utils import sort_checkpoints
from trl import SFTTrainer

logger = logging.getLogger(__name__)


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
        return super().evaluate(*args, **kwargs)

    def _save_checkpoint(self, model, trial=None):
        limit = self.args.save_total_limit
        if limit and limit > 0:
            run_dir = self._get_output_dir(trial=trial)
            keep = max(1, limit - 1)
            best = self.state.best_model_checkpoint
            checkpoints = sort_checkpoints(run_dir, use_mtime=True)
            if len(checkpoints) > keep:
                to_delete = []
                for cp in checkpoints:
                    if len(checkpoints) - len(to_delete) <= keep:
                        break
                    if cp != best:
                        to_delete.append(cp)
                for cp in to_delete:
                    logger.info("Pre-save rotation: removing %s", cp)
                    shutil.rmtree(cp, ignore_errors=True)

        saved_limit = self.args.save_total_limit
        self.args.save_total_limit = None
        try:
            super()._save_checkpoint(model, trial)
        finally:
            self.args.save_total_limit = saved_limit
