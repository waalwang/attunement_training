"""
data_loader.py

Loads SFT chain parquet files (from extract_sft_chains.py) and converts
them into the format expected by TRL's SFTTrainer + per-example weights
for weighted SFT.

SFTTrainer expects each row to have:
  - messages: list of {"role": ..., "content": ...}

We also compute per-turn `turn_weights` from per-turn attunement scores
and upvote scores, used by WeightedSFTTrainer to scale token-level loss.
"""

from __future__ import annotations

import glob
import json
import logging
import os

import numpy as np
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


# DEPRECATED: no weight floor, negative-attunement turns get zero gradient,
# and no cap on normalized weights. Causes train/eval divergence from
# overfitting to high-weight samples. Use _compute_turn_weights_v2 instead.
def _compute_turn_weights(
    turns: list[dict],
    beta: float,
    weight_mode: str = "gate_amplify",
) -> list[float]:
    """Compute per-turn loss weights.

    weight_mode="gate_amplify" (default / option A):
        Attunement gates: turns with attunement <= 0 get weight=0.
        Surviving turns: weight = attunement * (1 + beta * log1p(upvote))
        Upvote amplifies good turns but cannot rescue bad ones.

    weight_mode="attunement_only" (option B):
        weight = max(0, attunement_score), upvote ignored.
        Cleanest gradient signal; upvote only used for pre-filtering.

    weight_mode="additive" (original):
        weight = max(0, attunement) + beta * log1p(upvote)
        No gating; upvote creates a floor even for negative-attunement turns.

    For user/system turns: weight = 0.0 (masked out by labels=-100 anyway).
    attunement_score absent: treated as 0.5 (neutral).
    """
    weights = []
    for t in turns:
        if t["role"] != "assistant":
            weights.append(0.0)
            continue
        att = t.get("attunement_score")
        att_comp = max(0.0, att) if att is not None else 0.5

        if weight_mode == "attunement_only":
            weights.append(att_comp)
        elif weight_mode == "additive":
            score_comp = np.log1p(max(0, t.get("score", 0)))
            weights.append(att_comp + beta * score_comp)
        else:  # gate_amplify
            if att_comp == 0.0:
                weights.append(0.0)
            else:
                score_comp = np.log1p(max(0, t.get("score", 0)))
                weights.append(att_comp * (1.0 + beta * score_comp))
    return weights


def _compute_turn_weights_v2(
    turns: list[dict],
    beta: float,
    weight_mode: str = "gate_amplify",
    weight_floor: float = 0.1,
) -> list[float]:
    """Compute per-turn loss weights (v2: floor + clamp-ready).

    Same modes as v1, but negative-attunement turns get weight_floor
    instead of 0, so they still receive gradient signal.

    For user/system turns: weight = 0.0 (masked out by labels=-100 anyway).
    attunement_score absent: treated as 0.5 (neutral).
    """
    weights = []
    for t in turns:
        if t["role"] != "assistant":
            weights.append(0.0)
            continue
        att = t.get("attunement_score")
        att_comp = max(weight_floor, att) if att is not None else 0.5

        if weight_mode == "attunement_only":
            weights.append(att_comp)
        elif weight_mode == "additive":
            score_comp = np.log1p(max(0, t.get("score", 0)))
            weights.append(att_comp + beta * score_comp)
        else:  # gate_amplify
            score_comp = np.log1p(max(0, t.get("score", 0)))
            weights.append(att_comp * (1.0 + beta * score_comp))
    return weights


# DEPRECATED: no weight floor or clamp. Use load_sft_dataset_v2 instead.
def load_sft_dataset(
    data_dir: str,
    test_split: float = 0.05,
    seed: int = 42,
    min_chain_depth: int = 3,
    min_total_score: float = 0.0,
    weight_beta: float = 0.3,
    weight_mode: str = "gate_amplify",
) -> DatasetDict:
    """Load SFT chain parquet files into a HuggingFace DatasetDict.

    Args:
        data_dir:        Directory with SFT chain parquet shards.
        test_split:      Fraction held out for eval.
        seed:            Random seed for split.
        min_chain_depth: Drop chains shorter than this.
        min_total_score: Drop chains with total score below this.
        weight_beta:     Controls how much upvote score contributes to weight
                         vs attunement. Higher = more upvote influence.
        weight_mode:     "gate_amplify" (default), "attunement_only", or "additive".
                         See _compute_turn_weights for details.
    Returns:
        DatasetDict with "train" and "test" splits.
        Each row has: messages (list[dict]), turn_weights (list[float]).
    """
    if os.path.isfile(data_dir):
        files = [data_dir]
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files at {data_dir}")

    logger.info(f"Found {len(files)} parquet shard(s) in {data_dir}")

    rows = []
    for fpath in files:
        rows.extend(_load_shard(fpath, weight_beta, weight_mode,
                                min_chain_depth, min_total_score))

    if not rows:
        raise ValueError(f"No rows loaded from {data_dir}")

    # Normalize non-zero turn weights to mean=1 so effective LR is unchanged
    all_nonzero = [w for r in rows for w in r["turn_weights"] if w > 0]
    if all_nonzero:
        arr = np.array(all_nonzero)
        if arr.std() > 1e-8:
            scale = arr.mean()
            for r in rows:
                r["turn_weights"] = [w / scale if w > 0 else 0.0
                                     for w in r["turn_weights"]]
            arr = arr / scale
        else:
            for r in rows:
                r["turn_weights"] = [1.0 if w > 0 else 0.0
                                     for w in r["turn_weights"]]
            arr = np.ones_like(arr)
        logger.info(
            f"Loaded {len(rows)} chains | "
            f"turn weight stats (assistant only): mean={arr.mean():.3f} "
            f"std={arr.std():.3f} min={arr.min():.3f} max={arr.max():.3f}"
        )
    else:
        logger.info(f"Loaded {len(rows)} chains | no assistant turns found")

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=test_split, seed=seed)
    logger.info(f"Split: train={len(split['train'])}, test={len(split['test'])}")
    return split


def _load_shard(
    path: str,
    weight_beta: float,
    weight_mode: str,
    min_chain_depth: int,
    min_total_score: float,
) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    data = table.to_pydict()
    n = table.num_rows

    rows = []
    for i in range(n):
        chain_depth = data["chain_depth"][i]
        total_score = data["total_score"][i]
        if chain_depth < min_chain_depth:
            continue
        if total_score < min_total_score:
            continue

        turns = json.loads(data["turns"][i])
        messages = [{"role": t["role"], "content": t["content"]} for t in turns]
        turn_weights = _compute_turn_weights(turns, weight_beta, weight_mode)

        rows.append({
            "messages": messages,
            "turn_weights": turn_weights,
        })

    logger.info(f"  {os.path.basename(path)}: {n} total, {len(rows)} kept")
    return rows


# DEPRECATED: delegates to load_sft_dataset (v1). Use load_from_config_v2.
def load_from_config(config: dict) -> DatasetDict:
    data_cfg = config["data"]
    return load_sft_dataset(
        data_dir=data_cfg["sft_chain_dir"],
        test_split=data_cfg.get("test_split", 0.05),
        seed=data_cfg.get("seed", 42),
        min_chain_depth=data_cfg.get("min_chain_depth", 3),
        min_total_score=data_cfg.get("min_total_score", 0.0),
        weight_beta=data_cfg.get("weight_beta", 0.3),
        weight_mode=data_cfg.get("weight_mode", "gate_amplify"),
    )


# ---------------------------------------------------------------------------
# v2: weight floor + normalized weight clamp
# ---------------------------------------------------------------------------

def _load_shard_v2(
    path: str,
    weight_beta: float,
    weight_mode: str,
    weight_floor: float,
    min_chain_depth: int,
    min_total_score: float,
) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    data = table.to_pydict()
    n = table.num_rows

    rows = []
    for i in range(n):
        chain_depth = data["chain_depth"][i]
        total_score = data["total_score"][i]
        if chain_depth < min_chain_depth:
            continue
        if total_score < min_total_score:
            continue

        turns = json.loads(data["turns"][i])
        messages = [{"role": t["role"], "content": t["content"]} for t in turns]
        turn_weights = _compute_turn_weights_v2(
            turns, weight_beta, weight_mode, weight_floor,
        )

        rows.append({
            "messages": messages,
            "turn_weights": turn_weights,
        })

    logger.info(f"  {os.path.basename(path)}: {n} total, {len(rows)} kept")
    return rows


def load_sft_dataset_v2(
    data_dir: str,
    test_split: float = 0.05,
    seed: int = 42,
    min_chain_depth: int = 3,
    min_total_score: float = 0.0,
    weight_beta: float = 0.3,
    weight_mode: str = "gate_amplify",
    weight_floor: float = 0.1,
    weight_max: float = 3.0,
) -> DatasetDict:
    """Load SFT chain parquet files into a HuggingFace DatasetDict (v2).

    Changes from v1:
        - weight_floor: minimum weight for assistant turns (prevents zero
          gradient on negative-attunement turns).
        - weight_max: cap on normalized weights (prevents outlier samples
          from dominating gradient updates).
    """
    if os.path.isfile(data_dir):
        files = [data_dir]
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files at {data_dir}")

    logger.info(f"Found {len(files)} parquet shard(s) in {data_dir}")

    rows = []
    for fpath in files:
        rows.extend(_load_shard_v2(fpath, weight_beta, weight_mode,
                                   weight_floor, min_chain_depth,
                                   min_total_score))

    if not rows:
        raise ValueError(f"No rows loaded from {data_dir}")

    # Normalize non-zero turn weights to mean=1, then clamp to weight_max
    all_nonzero = [w for r in rows for w in r["turn_weights"] if w > 0]
    if all_nonzero:
        arr = np.array(all_nonzero)
        if arr.std() > 1e-8:
            scale = arr.mean()
            for r in rows:
                r["turn_weights"] = [
                    min(w / scale, weight_max) if w > 0 else 0.0
                    for w in r["turn_weights"]
                ]
            arr = np.clip(arr / scale, 0, weight_max)
        else:
            for r in rows:
                r["turn_weights"] = [1.0 if w > 0 else 0.0
                                     for w in r["turn_weights"]]
            arr = np.ones_like(arr)
        logger.info(
            f"Loaded {len(rows)} chains | "
            f"turn weight stats (assistant only): mean={arr.mean():.3f} "
            f"std={arr.std():.3f} min={arr.min():.3f} max={arr.max():.3f} "
            f"(floor={weight_floor}, clamp={weight_max})"
        )
    else:
        logger.info(f"Loaded {len(rows)} chains | no assistant turns found")

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=test_split, seed=seed)
    logger.info(f"Split: train={len(split['train'])}, test={len(split['test'])}")
    return split


def load_from_config_v2(config: dict) -> DatasetDict:
    data_cfg = config["data"]
    return load_sft_dataset_v2(
        data_dir=data_cfg["sft_chain_dir"],
        test_split=data_cfg.get("test_split", 0.05),
        seed=data_cfg.get("seed", 42),
        min_chain_depth=data_cfg.get("min_chain_depth", 3),
        min_total_score=data_cfg.get("min_total_score", 0.0),
        weight_beta=data_cfg.get("weight_beta", 0.3),
        weight_mode=data_cfg.get("weight_mode", "gate_amplify"),
        weight_floor=data_cfg.get("weight_floor", 0.1),
        weight_max=data_cfg.get("weight_max", 3.0),
    )
