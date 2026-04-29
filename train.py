"""
train.py

Weighted SFT training on attunement-scored conversation chains.
Per-example loss weights are computed from tone attunement scores
and per-turn comment scores.

Usage:
    # Cloud full FT, Qwen2.5-7B (recommended):
    python train.py --profile cloud_full

    # Cloud QLoRA:
    python train.py --profile cloud

    # Local prototyping (QLoRA, small batch):
    python train.py --profile local --model-override qwen-1.5b

    # Dry run:
    python train.py --profile cloud_full --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig

from data_loader import load_from_config
from weighted_sft_trainer import WeightedSFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SyncStateStepsCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        state.compute_steps(args, state.max_steps)


class DeleteOptimizerCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            return
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        opt = os.path.join(ckpt, "optimizer.pt")
        if os.path.exists(opt):
            os.remove(opt)
            logger.info("Deleted %s", opt)


def load_config(config_path: str, profile: str, model_override: str | None = None) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    hw = cfg["profiles"][profile]
    cfg["training"].update(hw)
    cfg["_profile"] = profile

    if model_override:
        presets = cfg.get("model_presets", {})
        if model_override not in presets:
            raise ValueError(f"Unknown model preset: {model_override!r}")
        cfg["model"]["name"] = presets[model_override]["name"]
        logger.info(f"Model override: {model_override} -> {cfg['model']['name']}")

    cfg["_full_finetune"] = cfg["training"].get("full_finetune", False)
    return cfg


def build_quantization_config(cfg: dict) -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg["training"]["bf16"] else torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(cfg: dict) -> LoraConfig:
    qlora = cfg["qlora"]
    return LoraConfig(
        r=qlora["r"],
        lora_alpha=qlora["lora_alpha"],
        lora_dropout=qlora["lora_dropout"],
        target_modules=qlora["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    mask = labels != -100
    correct = ((preds[:, :-1] == labels[:, 1:]) & mask[:, 1:]).sum()
    total = mask[:, 1:].sum()
    return {"accuracy": (correct / total).item() if total > 0 else 0.0}


def build_training_args(cfg: dict) -> SFTConfig:
    t = cfg["training"]
    return SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        warmup_ratio=t.get("warmup_ratio", 0.05),
        weight_decay=t["weight_decay"],
        lr_scheduler_type=t["lr_scheduler_type"],
        bf16=t["bf16"],
        fp16=t["fp16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        eval_steps=t["eval_steps"],
        eval_strategy=t.get("eval_strategy", "steps"),
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        report_to=t.get("report_to", "none"),
        run_name=t.get("run_name"),
        max_length=t["max_length"],
        dataset_num_proc=t.get("dataset_num_proc"),
        seed=cfg["data"].get("seed", 42),
        remove_unused_columns=False,
    )


def load_model_and_tokenizer(cfg: dict):
    model_name = cfg["model"]["name"]
    t = cfg["training"]
    full_ft = cfg["_full_finetune"]

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Profile: {cfg['_profile']} | full_finetune={full_ft}")

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if full_ft:
        if t["bf16"]:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif t["fp16"]:
            model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["quantization_config"] = build_quantization_config(cfg)

    attn_impl = t.get("attn_implementation")
    if attn_impl and attn_impl != "eager":
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if not full_ft:
        model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser(description="Weighted SFT training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--profile", default="cloud_full",
                        choices=["local", "cloud", "cloud_full"])
    parser.add_argument("--model-override", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.profile, args.model_override)
    full_ft = cfg["_full_finetune"]

    # --- Data ---
    logger.info("Loading SFT chain dataset...")
    dataset = load_from_config(cfg)

    if args.dry_run:
        for split_name, split in dataset.items():
            logger.info(f"--- {split_name}: {len(split)} examples ---")
            if len(split) > 0:
                sample = split[0]
                logger.info(f"  turns: {len(sample['messages'])}")
                tw = sample["turn_weights"]
                asst_w = [w for w in tw if w > 0]
                logger.info(f"  turn_weights: {len(tw)} turns, "
                            f"{len(asst_w)} assistant, "
                            f"asst mean={sum(asst_w)/len(asst_w):.3f}" if asst_w
                            else f"  turn_weights: {len(tw)} turns, no assistant")
                content = sample["messages"][0]["content"][:120]
                logger.info(f"  first turn: {content}...")

    # --- Model ---
    model, tokenizer = load_model_and_tokenizer(cfg)

    if full_ft:
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Full fine-tune: {total:,} params, all trainable")
    else:
        lora_config = build_lora_config(cfg)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if args.dry_run:
        logger.info("Dry run complete.")
        return

    # --- Train ---
    training_args = build_training_args(cfg)

    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[SyncStateStepsCallback(), DeleteOptimizerCallback()],
    )

    logger.info("Starting weighted SFT training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Training complete. Best checkpoint kept in %s", cfg["training"]["output_dir"])


if __name__ == "__main__":
    main()
