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
    attunement_path: str | None = None,
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
        attunement_path: Optional path to attunement-scored parquet. If provided,
                         loads attunement_score per post_id and merges.

    Returns:
        DatasetDict with "train" and "test" splits.
        Each row has: messages (list[dict]), weight (float).
    """
    pattern = os.path.join(data_dir, "*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")

    logger.info(f"Found {len(files)} parquet shard(s) in {data_dir}")

    # Load attunement scores if available
    attunement_map = None
    if attunement_path and os.path.exists(attunement_path):
        attunement_map = _load_attunement_scores(attunement_path)
        logger.info(f"Loaded attunement scores for {len(attunement_map)} posts")

    rows = []
    for fpath in files:
        rows.extend(_load_shard(fpath, attunement_map, weight_beta,
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


def _load_attunement_scores(path: str) -> dict[str, float]:
    """Load post_id -> attunement_score mapping from scored parquet."""
    import pyarrow.parquet as pq
    table = pq.read_table(path, columns=["post_id", "attunement_score"])
    post_ids = table.column("post_id").to_pylist()
    scores = table.column("attunement_score").to_pylist()
    return dict(zip(post_ids, scores))


def _load_shard(
    path: str,
    attunement_map: dict[str, float] | None,
    weight_beta: float,
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

        post_id = data["post_id"][i]
        attunement = attunement_map.get(post_id) if attunement_map else None
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
        attunement_path=data_cfg.get("attunement_path"),
    )
