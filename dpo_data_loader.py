"""
dpo_data_loader.py

Loads DPO pair parquet files (from generate_dpo_rejected.py) and converts
them into the format expected by TRL's DPOTrainer.

DPOTrainer (TRL >= 1.0) expects each row to have:
  - prompt: list of {"role": ..., "content": ...}  (shared prefix)
  - chosen: list of {"role": ..., "content": ...}  (diverging chosen branch)
  - rejected: list of {"role": ..., "content": ...} (diverging rejected branch)

Supplying prompt explicitly avoids TRL's extract_prompt pass and the
tokenization prefix mismatch warning from add_generation_prompt=True.

Optionally computes a per-example chosen_weight for weighted DPO,
derived from attunement/upvote scores on the chosen branch.
"""

from __future__ import annotations

import glob
import json
import logging
import os

import numpy as np
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


def _strip_messages(turns: list[dict]) -> list[dict]:
    return [{"role": t["role"], "content": t["content"]} for t in turns]


def _compute_example_weight(turns: list[dict], beta: float, weight_mode: str) -> float:
    from data_loader import _compute_turn_weights

    weights = _compute_turn_weights(turns, beta, weight_mode)
    asst_w = [w for w in weights if w > 0]
    return float(np.mean(asst_w)) if asst_w else 1.0


def load_dpo_dataset(
    data_dir: str,
    test_split: float = 0.05,
    seed: int = 42,
    min_score_delta: float = 0.0,
    chosen_weighting: bool = False,
    weight_beta: float = 0.3,
    weight_mode: str = "gate_amplify",
    length_matched_only: bool = True,
) -> DatasetDict:
    if os.path.isfile(data_dir):
        files = [data_dir]
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files at {data_dir}")

    logger.info("Found %d DPO parquet shard(s) in %s", len(files), data_dir)

    rows = []
    for fpath in files:
        rows.extend(_load_shard(
            fpath, min_score_delta, chosen_weighting, weight_beta, weight_mode,
            length_matched_only,
        ))

    if not rows:
        raise ValueError(f"No DPO rows loaded from {data_dir}")

    if chosen_weighting:
        weights = np.array([r["chosen_weight"] for r in rows])
        nonzero = weights[weights > 0]
        if len(nonzero) > 0 and nonzero.std() > 1e-8:
            scale = nonzero.mean()
            for r in rows:
                r["chosen_weight"] = r["chosen_weight"] / scale if r["chosen_weight"] > 0 else 1.0
            logger.info(
                "Chosen weights normalized: mean=1.000 std=%.3f min=%.3f max=%.3f",
                (nonzero / scale).std(), (nonzero / scale).min(), (nonzero / scale).max(),
            )
        else:
            for r in rows:
                r["chosen_weight"] = 1.0

    logger.info("Loaded %d DPO pairs", len(rows))

    ds = Dataset.from_list(rows)
    split = ds.train_test_split(test_size=test_split, seed=seed)
    logger.info("Split: train=%d, test=%d", len(split["train"]), len(split["test"]))
    return split


def _load_shard(
    path: str,
    min_score_delta: float,
    chosen_weighting: bool,
    weight_beta: float,
    weight_mode: str,
    length_matched_only: bool = True,
) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    data = table.to_pydict()
    n = table.num_rows

    rows = []
    for i in range(n):
        if data["score_delta"][i] < min_score_delta:
            continue
        if length_matched_only and not data.get("length_matched", [True] * n)[i]:
            continue

        fork = json.loads(data["fork"][i])
        chosen_branch = json.loads(data["chosen_branch"][i])
        rejected_branch = json.loads(data["rejected_branch"][i])

        row = {
            "prompt": _strip_messages(fork),
            "chosen": _strip_messages(chosen_branch),
            "rejected": _strip_messages(rejected_branch),
        }

        if chosen_weighting:
            row["chosen_weight"] = _compute_example_weight(
                chosen_branch, weight_beta, weight_mode,
            )

        rows.append(row)

    logger.info("  %s: %d total, %d kept", os.path.basename(path), n, len(rows))
    return rows


def load_dpo_from_config(config: dict) -> DatasetDict:
    dpo_cfg = config["dpo"]
    return load_dpo_dataset(
        data_dir=dpo_cfg["data_dir"],
        test_split=config["data"].get("test_split", 0.05),
        seed=config["data"].get("seed", 42),
        min_score_delta=dpo_cfg.get("min_score_delta", 0.0),
        chosen_weighting=dpo_cfg.get("chosen_weighting", False),
        weight_beta=dpo_cfg.get("weight_beta", 0.3),
        weight_mode=dpo_cfg.get("weight_mode", "gate_amplify"),
        length_matched_only=dpo_cfg.get("length_matched_only", True),
    )
