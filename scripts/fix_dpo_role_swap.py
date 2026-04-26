#!/usr/bin/env python3
"""
scripts/fix_dpo_role_swap.py

Patch DPO parquet files so that TRL's DPOTrainer sees clean prompt/response
boundaries.

Problem: DPOTrainer tokenizes the prompt with add_generation_prompt=True,
appending an assistant header. When the fork ends on an assistant turn
(even fork depth), the branch starts with a user turn, causing a BPE
token mismatch at the prompt/response boundary.

Fix: For pairs where the fork ends on assistant, swap all user<->assistant
role labels throughout fork + both branches. The fork now ends on "user",
branches start with "assistant", and TRL splits cleanly. DPO loss only
cares about chosen-vs-rejected contrast, not role semantics.

Usage:
    python scripts/fix_dpo_role_swap.py \
        --input data/dpo_unweighted_sft_final.parquet \
        --output data/dpo_unweighted_sft_final_fixed.parquet

    # Dry run (stats only, no output):
    python scripts/fix_dpo_role_swap.py \
        --input data/dpo_unweighted_sft_final.parquet \
        --dry-run

    # In-place (overwrites input):
    python scripts/fix_dpo_role_swap.py \
        --input data/dpo_unweighted_sft_final.parquet \
        --in-place
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROLE_SWAP = {"user": "assistant", "assistant": "user", "system": "system"}


def swap_roles(turns: list[dict]) -> list[dict]:
    return [{**t, "role": ROLE_SWAP.get(t["role"], t["role"])} for t in turns]


def parse_args():
    p = argparse.ArgumentParser(
        description="Swap user/assistant roles in even-fork-depth DPO pairs")
    p.add_argument("--input", required=True, help="Input parquet file")
    p.add_argument("--output", default=None, help="Output parquet file")
    p.add_argument("--in-place", action="store_true",
                   help="Overwrite input file")
    p.add_argument("--dry-run", action="store_true",
                   help="Print stats only, no output")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.dry_run and not args.output and not args.in_place:
        logger.error("Specify --output, --in-place, or --dry-run")
        return

    table = pq.read_table(args.input)
    data = table.to_pydict()
    n = len(data["fork"])
    logger.info("Loaded %d pairs from %s", n, args.input)

    swapped = 0
    already_ok = 0

    for i in range(n):
        fork = json.loads(data["fork"][i])

        if not fork or fork[-1]["role"] != "assistant":
            already_ok += 1
            continue

        chosen = json.loads(data["chosen_branch"][i])
        rejected = json.loads(data["rejected_branch"][i])

        data["fork"][i] = json.dumps(swap_roles(fork), ensure_ascii=False)
        data["chosen_branch"][i] = json.dumps(swap_roles(chosen), ensure_ascii=False)
        data["rejected_branch"][i] = json.dumps(swap_roles(rejected), ensure_ascii=False)
        swapped += 1

    logger.info(
        "Results: %d swapped, %d already ok (%.1f%% swapped)",
        swapped, already_ok, 100 * swapped / max(n, 1),
    )

    if args.dry_run:
        logger.info("Dry run -- no output written")
        return

    out_table = pa.table(data, schema=table.schema)

    output_path = args.output
    if args.in_place:
        output_path = args.input + ".tmp"

    pq.write_table(out_table, output_path)
    logger.info("Wrote %d pairs to %s", n, output_path)

    if args.in_place:
        shutil.move(output_path, args.input)
        logger.info("Replaced %s in-place", args.input)


if __name__ == "__main__":
    main()
