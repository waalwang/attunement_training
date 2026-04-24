"""
data_loader.py

Loads SFT chain parquet files (from extract_sft_chains.py) and converts
them into the format expected by TRL's SFTTrainer + per-example weights
for weighted SFT.

SFTTrainer expects each row to have:
  - messages: list of {"role": ..., "content": ...}

We also compute a per-example `weight` from the attunement score and
per-turn comment scores, used by WeightedSFTTrainer to scale the loss.
"""

from __future__ import annotations

import glob
import json
import logging
import os

import numpy as np
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


def _compute_weight(
    turns: list[dict],
    attunement_score: float | None,
    beta: float,
) -> float:
    """Compute per-example SFT loss weight.

    weight = attunement_component + beta * score_component

    attunement_component: max(0, attunement_score) -- clipped negative = 0
    score_component: mean(log(1 + turn_score)) across assistant turns
                     log compresses heavy-tailed upvotes so viral punchlines
                     don't dominate

    If attunement_score is None (not yet scored), falls back to score_component
    only with a baseline attunement of 0.5.
    """
    # Score component: log-compressed mean of assistant turn scores
    asst_scores = [t.get("score", 0) for t in turns if t["role"] == "assistant"]
    if asst_scores:
        score_component = np.mean([np.log1p(max(0, s)) for s in asst_scores])
    else:
        score_component = 0.0

    # Attunement component
    if attunement_score is not None:
        attunement_component = max(0.0, attunement_score)
    else:
        attunement_component = 0.5

    return attunement_component + beta * score_component


def load_sft_dataset(
    data_dir: str,
    test_split: float = 0.05,
    seed: int = 42,
    min_chain_depth: int = 3,
    min_total_score: float = 0.0,
    weight_beta: float = 0.3,
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
    Returns:
        DatasetDict with "train" and "test" splits.
        Each row has: messages (list[dict]), weight (float).
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
        rows.extend(_load_shard(fpath, weight_beta,
                                min_chain_depth, min_total_score))

    if not rows:
        raise ValueError(f"No rows loaded from {data_dir}")

    # Normalize weights to mean=1 so effective LR is unchanged
    weights = np.array([r["weight"] for r in rows])
    if weights.std() > 1e-8:
        weights = weights / weights.mean()
    else:
        weights = np.ones_like(weights)
    for r, w in zip(rows, weights):
        r["weight"] = float(w)

    logger.info(
        f"Loaded {len(rows)} chains | "
        f"weight stats: mean={weights.mean():.3f} std={weights.std():.3f} "
        f"min={weights.min():.3f} max={weights.max():.3f}"
    )

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=test_split, seed=seed)
    logger.info(f"Split: train={len(split['train'])}, test={len(split['test'])}")
    return split


def _load_shard(
    path: str,
    weight_beta: float,
    min_chain_depth: int,
    min_total_score: float,
) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    data = table.to_pydict()
    n = table.num_rows
    has_attunement = "attunement_score" in data

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

        attunement = data["attunement_score"][i] if has_attunement else None
        weight = _compute_weight(turns, attunement, weight_beta)

        rows.append({
            "messages": messages,
            "weight": weight,
        })

    logger.info(f"  {os.path.basename(path)}: {n} total, {len(rows)} kept")
    return rows


def load_from_config(config: dict) -> DatasetDict:
    data_cfg = config["data"]
    return load_sft_dataset(
        data_dir=data_cfg["sft_chain_dir"],
        test_split=data_cfg.get("test_split", 0.05),
        seed=data_cfg.get("seed", 42),
        min_chain_depth=data_cfg.get("min_chain_depth", 3),
        min_total_score=data_cfg.get("min_total_score", 0.0),
        weight_beta=data_cfg.get("weight_beta", 0.3),
    )
