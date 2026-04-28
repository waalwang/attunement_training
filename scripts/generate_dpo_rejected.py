#!/usr/bin/env python3
"""
scripts/generate_dpo_rejected.py

Stage 2 of DPO pipeline: synthetic replacement on rejected branches.

Takes raw DPO pairs (from extract_dpo_pairs.py) and processes rejected
branches:
  1. Keep low-score assistant turns as-is (authentically bad)
  2. Replace high-score assistant turns with SFT-checkpoint-generated text
  3. Length-match rejected branch to chosen branch by adjusting user turns

Uses vLLM for batched offline inference (3-5x faster than HuggingFace
sequential generation on Turing GPUs).

Usage:
    python scripts/generate_dpo_rejected.py \
        --input-dir data/dpo_pairs \
        --checkpoint outputs/sft/checkpoint-XXXX \
        --output-path data/dpo_final.parquet \
        --score-threshold 5 \
        --length-tolerance 0.30
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

USER_GEN_SYSTEM = (
    "You are role-playing as a forum commenter. Write a natural follow-up "
    "response to the conversation. Match the tone and style of the previous "
    "messages. Be concise."
)


def _truncate_at_boundary(text: str, target_chars: int) -> str:
    if len(text) <= target_chars:
        return text
    for i in range(target_chars, max(target_chars // 2, 0), -1):
        if text[i] in '.!?\n':
            return text[:i + 1]
    return text[:target_chars]


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic rejected turns")
    p.add_argument("--input-dir", required=True,
                   help="Directory with DPO pair parquet shards")
    p.add_argument("--checkpoint", required=True,
                   help="Path to SFT checkpoint or model name")
    p.add_argument("--output-path", required=True,
                   help="Output parquet path")
    p.add_argument("--score-threshold", type=int, default=5,
                   help="Replace assistant turns with score >= this")
    p.add_argument("--length-tolerance", type=float, default=0.30,
                   help="Max allowed (rej - cho) / max(rej, cho); pairs where "
                        "rejected is longer beyond this are dropped")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--device", default="cuda:0",
                   help="Device: cuda:0, cuda:1, cpu, etc.")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "auto"],
                   help="Model dtype (default float16 for Turing GPUs)")
    p.add_argument("--keep-unmatched", action="store_true", default=False,
                   help="Keep length-unmatched pairs in output (for debugging)")
    p.add_argument("--no-skip-matched", action="store_true", default=False,
                   help="Disable skipping pairs that already meet length "
                        "tolerance and have no high-score assistant turns")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                   help="Fraction of GPU memory for vLLM (default 0.85)")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N pairs")
    return p.parse_args()


def load_pairs(input_dir: str) -> pa.Table:
    if os.path.isfile(input_dir):
        files = [input_dir]
    else:
        files = sorted(glob.glob(os.path.join(input_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files at {input_dir}")

    tables = [pq.read_table(f) for f in files]
    table = pa.concat_tables(tables)
    logger.info("Loaded %d pairs from %d shard(s)", len(table), len(files))
    return table


def _build_prompt(tokenizer, messages: list[dict], role: str) -> str:
    if role == "user":
        msgs = [{"role": "system", "content": USER_GEN_SYSTEM}] + messages
    else:
        msgs = list(messages)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def _collect_asst_requests(
    pairs: list[dict],
    tokenizer,
    score_threshold: int,
):
    requests = []
    for pair_idx, pair in enumerate(pairs):
        fork = pair["fork"]
        rejected = pair["rejected"]
        for turn_idx, turn in enumerate(rejected):
            if turn["role"] != "assistant":
                continue
            if turn["score"] < score_threshold:
                continue
            context = fork + [
                {"role": t["role"], "content": t["content"]}
                for t in rejected[:turn_idx]
            ]
            prompt = _build_prompt(tokenizer, context, "assistant")
            target_chars = len(turn["content"])
            min_tokens = max(1, int(target_chars * 0.7 / 4))
            requests.append({
                "pair_idx": pair_idx,
                "turn_idx": turn_idx,
                "prompt": prompt,
                "max_new_tokens": max(32, int(target_chars * 1.3 / 3)),
                "min_tokens": min_tokens,
                "target_chars": target_chars,
                "role": "assistant",
            })
    return requests


def _collect_length_match_requests(
    pairs: list[dict],
    tokenizer,
    length_tolerance: float,
):
    requests = []
    for pair_idx, pair in enumerate(pairs):
        fork = pair["fork"]
        rejected = pair["rejected"]
        chosen_chars = pair["chosen_chars"]

        rejected_chars = sum(len(t["content"]) for t in rejected)
        denom = max(chosen_chars, rejected_chars, 1)
        if abs(chosen_chars - rejected_chars) / denom <= length_tolerance:
            continue

        # positive = expand rejected, negative = shrink rejected
        diff = chosen_chars - rejected_chars
        adjustable = [
            i for i, t in enumerate(rejected)
            if t["role"] == "user" or t.get("synthetic", False)
        ]
        if not adjustable:
            continue

        per_turn_extra = diff // len(adjustable)

        for idx in adjustable:
            cur_rej = sum(len(t["content"]) for t in rejected)
            cur_denom = max(chosen_chars, cur_rej, 1)
            if abs(chosen_chars - cur_rej) / cur_denom <= length_tolerance:
                break

            per_turn_target = len(rejected[idx]["content"]) + per_turn_extra
            per_turn_target = max(20, per_turn_target)
            role = rejected[idx]["role"]

            context = fork + [
                {"role": t["role"], "content": t["content"]}
                for t in rejected[:idx]
            ]
            prompt = _build_prompt(tokenizer, context, role)
            min_tokens = max(1, int(per_turn_target * 0.7 / 4))

            requests.append({
                "pair_idx": pair_idx,
                "turn_idx": idx,
                "prompt": prompt,
                "max_new_tokens": max(32, int(per_turn_target * 1.3 / 3)),
                "min_tokens": min_tokens,
                "target_chars": per_turn_target,
                "role": role,
            })
    return requests


def _batch_generate(llm, requests, temperature, top_p):
    if not requests:
        return {}

    from vllm import SamplingParams

    grouped = {}
    for req in requests:
        key = (req.get("min_tokens", 0), req["max_new_tokens"])
        grouped.setdefault(key, []).append(req)

    results = {}
    total_done = 0
    for (min_tok, max_tok), group in sorted(grouped.items()):
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            min_tokens=min_tok,
            max_tokens=max_tok,
        )
        prompts = [r["prompt"] for r in group]
        outputs = llm.generate(prompts, params)

        for req, out in zip(group, outputs):
            text = out.outputs[0].text.strip()
            target = req.get("target_chars")
            if target and len(text) > int(target * 1.3):
                text = _truncate_at_boundary(text, target)
            results[(req["pair_idx"], req["turn_idx"])] = text

        total_done += len(group)
        logger.info(
            "Generated %d/%d (min_tokens=%d, max_tokens=%d, batch=%d)",
            total_done, len(requests), min_tok, max_tok, len(group),
        )

    return results


def main():
    args = parse_args()

    table = load_pairs(args.input_dir)

    data = table.to_pydict()
    n = len(table)
    if args.limit:
        n = min(n, args.limit)

    skip_matched = not args.no_skip_matched

    # Phase 1: parse all pairs, separate skip vs needs-generation
    pairs_to_process = []
    skipped_results = []

    logger.info("Phase 1: scanning %d pairs...", n)
    t0 = time.time()

    for i in range(n):
        fork = json.loads(data["fork"][i])
        chosen = json.loads(data["chosen_branch"][i])
        rejected = json.loads(data["rejected_branch"][i])

        chosen_chars = sum(len(t["content"]) for t in chosen)
        rejected_chars = sum(len(t["content"]) for t in rejected)

        meta = {
            "source": data["source"][i],
            "subreddit": data["subreddit"][i],
            "post_id": data["post_id"][i],
            "post_url": data["post_url"][i],
            "post_score": data["post_score"][i],
            "fork_depth": data["fork_depth"][i],
            "chosen_depth": data["chosen_depth"][i],
            "chosen_score": data["chosen_score"][i],
            "rejected_score": data["rejected_score"][i],
            "score_delta": data["score_delta"][i],
        }

        if skip_matched:
            denom = max(chosen_chars, rejected_chars, 1)
            length_ok = abs(chosen_chars - rejected_chars) / denom <= args.length_tolerance
            no_high_asst = all(
                t.get("score", 0) < args.score_threshold
                for t in rejected if t["role"] == "assistant"
            )
            if length_ok and no_high_asst:
                skipped_results.append({
                    "fork": fork,
                    "chosen_branch": chosen,
                    "rejected_branch": rejected,
                    "chosen_chars": chosen_chars,
                    "rejected_chars": rejected_chars,
                    "asst_replacements": 0,
                    "user_replacements": 0,
                    "length_matched": True,
                    **meta,
                })
                continue

        pairs_to_process.append({
            "fork": fork,
            "chosen": chosen,
            "rejected": [dict(t) for t in rejected],
            "chosen_chars": chosen_chars,
            "meta": meta,
        })

    logger.info(
        "Phase 1 done in %.1fs: %d skipped, %d need generation",
        time.time() - t0, len(skipped_results), len(pairs_to_process),
    )

    if not pairs_to_process:
        results = skipped_results
    else:
        # Load vLLM model
        logger.info("Loading vLLM model from %s...", args.checkpoint)
        t_load = time.time()

        from vllm import LLM
        from transformers import AutoTokenizer

        device_idx = 0
        if ":" in args.device:
            device_idx = int(args.device.split(":")[1])

        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)

        llm = LLM(
            model=args.checkpoint,
            dtype=args.dtype,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint, trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Model loaded in %.1fs", time.time() - t_load)

        # Phase 2: collect and batch assistant replacement requests
        logger.info("Phase 2: assistant turn replacement...")
        t_asst = time.time()

        asst_requests = _collect_asst_requests(
            pairs_to_process, tokenizer, args.score_threshold
        )
        logger.info("Collected %d assistant replacement requests", len(asst_requests))

        asst_results = _batch_generate(
            llm, asst_requests, args.temperature, args.top_p
        )

        # Apply assistant replacements
        for (pair_idx, turn_idx), text in asst_results.items():
            pairs_to_process[pair_idx]["rejected"][turn_idx]["content"] = text
            pairs_to_process[pair_idx]["rejected"][turn_idx]["synthetic"] = True

        logger.info(
            "Phase 2 done in %.1fs: %d replacements",
            time.time() - t_asst, len(asst_results),
        )

        # Phase 3: expand or shrink all available slots for length matching
        logger.info("Phase 3: length matching (expand/shrink user + synthetic asst)...")
        t_user = time.time()

        user_requests = _collect_length_match_requests(
            pairs_to_process, tokenizer, args.length_tolerance
        )
        logger.info("Collected %d length-match requests", len(user_requests))

        user_results = _batch_generate(
            llm, user_requests, args.temperature, args.top_p
        )

        for (pair_idx, turn_idx), text in user_results.items():
            pairs_to_process[pair_idx]["rejected"][turn_idx]["content"] = text
            pairs_to_process[pair_idx]["rejected"][turn_idx]["synthetic"] = True

        logger.info(
            "Phase 3 done in %.1fs: %d replacements",
            time.time() - t_user, len(user_results),
        )

        # Phase 4: assemble results, apply final length check
        logger.info("Phase 4: final length check and assembly...")
        results = list(skipped_results)
        total_dropped_length = 0

        asst_counts = {}
        for req in asst_requests:
            asst_counts[req["pair_idx"]] = asst_counts.get(req["pair_idx"], 0) + 1
        user_counts = {}
        for req in user_requests:
            user_counts[req["pair_idx"]] = user_counts.get(req["pair_idx"], 0) + 1

        for pair_idx, pair in enumerate(pairs_to_process):
            rejected = pair["rejected"]
            chosen_chars = pair["chosen_chars"]
            rejected_chars = sum(len(t["content"]) for t in rejected)
            denom = max(rejected_chars, chosen_chars, 1)
            matched = abs(rejected_chars - chosen_chars) / denom <= args.length_tolerance

            if not matched and not args.keep_unmatched:
                total_dropped_length += 1
                continue

            results.append({
                "fork": pair["fork"],
                "chosen_branch": pair["chosen"],
                "rejected_branch": rejected,
                "chosen_chars": chosen_chars,
                "rejected_chars": rejected_chars,
                "asst_replacements": asst_counts.get(pair_idx, 0),
                "user_replacements": user_counts.get(pair_idx, 0),
                "length_matched": matched,
                **pair["meta"],
            })

        logger.info(
            "Phase 4: %d kept, %d dropped (length unmatched after synthesis)",
            len(results) - len(skipped_results), total_dropped_length,
        )

    elapsed = time.time() - t0
    total_asst = sum(r["asst_replacements"] for r in results)
    total_user = sum(r["user_replacements"] for r in results)
    total_matched = sum(1 for r in results if r["length_matched"])
    logger.info(
        "Done in %.1fs: %d input | kept=%d (skipped=%d, generated=%d) | "
        "asst_replaced=%d user_replaced=%d matched=%d",
        elapsed, n, len(results), len(skipped_results),
        len(results) - len(skipped_results),
        total_asst, total_user, total_matched,
    )

    out_table = pa.table({
        "fork": pa.array(
            [json.dumps(r["fork"], ensure_ascii=False) for r in results],
            type=pa.string(),
        ),
        "chosen_branch": pa.array(
            [json.dumps(r["chosen_branch"], ensure_ascii=False) for r in results],
            type=pa.string(),
        ),
        "rejected_branch": pa.array(
            [json.dumps(r["rejected_branch"], ensure_ascii=False) for r in results],
            type=pa.string(),
        ),
        "source": pa.array([r["source"] for r in results], type=pa.string()),
        "subreddit": pa.array([r["subreddit"] for r in results], type=pa.string()),
        "post_id": pa.array([r["post_id"] for r in results], type=pa.string()),
        "post_url": pa.array([r["post_url"] for r in results], type=pa.string()),
        "post_score": pa.array([r["post_score"] for r in results], type=pa.int64()),
        "fork_depth": pa.array([r["fork_depth"] for r in results], type=pa.int64()),
        "chosen_depth": pa.array([r["chosen_depth"] for r in results], type=pa.int64()),
        "rejected_depth": pa.array(
            [len(r["rejected_branch"]) if isinstance(r["rejected_branch"], list)
             else len(json.loads(r["rejected_branch"])) for r in results],
            type=pa.int64(),
        ),
        "chosen_score": pa.array([r["chosen_score"] for r in results], type=pa.float64()),
        "rejected_score": pa.array([r["rejected_score"] for r in results], type=pa.float64()),
        "score_delta": pa.array([r["score_delta"] for r in results], type=pa.float64()),
        "chosen_chars": pa.array([r["chosen_chars"] for r in results], type=pa.int64()),
        "rejected_chars": pa.array([r["rejected_chars"] for r in results], type=pa.int64()),
        "asst_replacements": pa.array(
            [r["asst_replacements"] for r in results], type=pa.int64(),
        ),
        "user_replacements": pa.array(
            [r["user_replacements"] for r in results], type=pa.int64(),
        ),
        "length_matched": pa.array(
            [r["length_matched"] for r in results], type=pa.bool_(),
        ),
    })

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    pq.write_table(out_table, args.output_path)
    logger.info("Wrote %d pairs to %s", len(results), args.output_path)


if __name__ == "__main__":
    main()
