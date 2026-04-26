#!/usr/bin/env python3
"""
scripts/inspect_dpo_pairs.py

Inspection and stats for DPO pair parquet files produced by
generate_dpo_rejected.py (Stage 2 output).

Usage:
    python scripts/inspect_dpo_pairs.py --data-path data/dpo_unweighted_sft_final.parquet
    python scripts/inspect_dpo_pairs.py --data-path data/dpo_unweighted_sft_final.parquet --samples 5
    python scripts/inspect_dpo_pairs.py --data-path data/dpo_unweighted_sft_final.parquet --source reddit
    python scripts/inspect_dpo_pairs.py --data-path data/dpo_unweighted_sft_final.parquet --subreddit Advice
    python scripts/inspect_dpo_pairs.py --data-path data/dpo_unweighted_sft_final.parquet --synthetic-only
    python scripts/inspect_dpo_pairs.py --data-path data/dpo_unweighted_sft_final.parquet --max-rows 5000
    python scripts/inspect_dpo_pairs.py --data-path data/dpo_unweighted_sft_final.parquet --print-all --samples 1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True,
                   help="Parquet file or directory of shards")
    p.add_argument("--samples", type=int, default=3,
                   help="Number of sample pairs to display")
    p.add_argument("--source", default=None,
                   help="Filter by source (reddit, hacker_news)")
    p.add_argument("--subreddit", default=None)
    p.add_argument("--post-id", default=None,
                   help="Show all pairs for a specific post")
    p.add_argument("--synthetic-only", action="store_true",
                   help="Only show pairs with synthetic replacements")
    p.add_argument("--untouched-only", action="store_true",
                   help="Only show pairs with no synthetic replacements")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--print-all", action="store_true",
                   help="Print full turn content (no truncation)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_pairs(data_path, source=None, subreddit=None, post_id=None,
               synthetic_only=False, untouched_only=False, max_rows=None):
    import pyarrow.parquet as pq

    if os.path.isfile(data_path):
        files = [data_path]
    else:
        files = sorted(glob.glob(os.path.join(data_path, "**", "*.parquet"),
                                 recursive=True))
    if not files:
        print(f"No parquet files at {data_path}")
        sys.exit(1)

    rows = []
    for f in files:
        table = pq.read_table(f)
        data = table.to_pydict()
        for i in range(len(table)):
            row = {col: data[col][i] for col in table.schema.names}
            if source and row.get("source") != source:
                continue
            if subreddit and row.get("subreddit") != subreddit:
                continue
            if post_id and row.get("post_id") != post_id:
                continue

            row["fork"] = json.loads(row["fork"])
            row["chosen_branch"] = json.loads(row["chosen_branch"])
            row["rejected_branch"] = json.loads(row["rejected_branch"])

            touched = row.get("asst_replacements", 0) > 0 or row.get("user_replacements", 0) > 0
            if synthetic_only and not touched:
                continue
            if untouched_only and touched:
                continue

            rows.append(row)
            if max_rows and len(rows) >= max_rows:
                return rows
    return rows


def print_stats(rows):
    n = len(rows)
    print(f"\n{'='*60}")
    print(f"Total DPO pairs: {n:,}")
    print(f"{'='*60}")

    if not rows:
        return

    # Source breakdown
    sources = Counter(r["source"] for r in rows)
    print(f"\nSource breakdown:")
    for src, cnt in sources.most_common():
        print(f"  {src}: {cnt:,} ({100*cnt/n:.1f}%)")

    # Subreddit breakdown (top 20)
    subs = Counter(r.get("subreddit") or "(none)" for r in rows)
    print(f"\nSubreddit breakdown (top 20 of {len(subs)}):")
    for sub, cnt in subs.most_common(20):
        print(f"  {sub}: {cnt:,} ({100*cnt/n:.1f}%)")

    # Unique posts
    post_ids = set(r["post_id"] for r in rows)
    print(f"\nUnique posts: {len(post_ids):,}")
    pairs_per_post = Counter(r["post_id"] for r in rows)
    ppp = list(pairs_per_post.values())
    print(f"Pairs per post: mean={np.mean(ppp):.1f} median={np.median(ppp):.0f} "
          f"max={max(ppp)} p90={np.percentile(ppp, 90):.0f}")

    # Depth stats
    fork_depths = np.array([r["fork_depth"] for r in rows])
    chosen_depths = np.array([r["chosen_depth"] for r in rows])
    rejected_depths = np.array([len(r["rejected_branch"]) for r in rows])

    print(f"\nFork depth (shared prefix turns):")
    print(f"  mean={fork_depths.mean():.1f} median={np.median(fork_depths):.0f} "
          f"min={fork_depths.min()} max={fork_depths.max()}")
    depth_dist = Counter(fork_depths.tolist())
    for d in sorted(depth_dist):
        cnt = depth_dist[d]
        if cnt >= n * 0.01 or d <= 10:
            print(f"    depth {d}: {cnt:,} ({100*cnt/n:.1f}%)")

    print(f"\nChosen branch depth:")
    print(f"  mean={chosen_depths.mean():.1f} median={np.median(chosen_depths):.0f} "
          f"min={chosen_depths.min()} max={chosen_depths.max()}")

    print(f"\nRejected branch depth:")
    print(f"  mean={rejected_depths.mean():.1f} median={np.median(rejected_depths):.0f} "
          f"min={rejected_depths.min()} max={rejected_depths.max()}")

    # Score stats
    chosen_scores = np.array([r["chosen_score"] for r in rows])
    rejected_scores = np.array([r["rejected_score"] for r in rows])
    score_deltas = np.array([r["score_delta"] for r in rows])

    print(f"\nChosen score:")
    print(f"  mean={chosen_scores.mean():.1f} median={np.median(chosen_scores):.0f} "
          f"p10={np.percentile(chosen_scores, 10):.0f} p90={np.percentile(chosen_scores, 90):.0f}")

    print(f"\nRejected score:")
    print(f"  mean={rejected_scores.mean():.1f} median={np.median(rejected_scores):.0f} "
          f"p10={np.percentile(rejected_scores, 10):.0f} p90={np.percentile(rejected_scores, 90):.0f}")

    print(f"\nScore delta (chosen - rejected):")
    print(f"  mean={score_deltas.mean():.1f} median={np.median(score_deltas):.0f} "
          f"p10={np.percentile(score_deltas, 10):.0f} p90={np.percentile(score_deltas, 90):.0f}")

    # Length stats
    chosen_chars = np.array([r["chosen_chars"] for r in rows])
    rejected_chars = np.array([r["rejected_chars"] for r in rows])

    print(f"\nChosen chars:")
    print(f"  mean={chosen_chars.mean():.0f} median={np.median(chosen_chars):.0f} "
          f"p90={np.percentile(chosen_chars, 90):.0f} max={chosen_chars.max()}")

    print(f"\nRejected chars:")
    print(f"  mean={rejected_chars.mean():.0f} median={np.median(rejected_chars):.0f} "
          f"p90={np.percentile(rejected_chars, 90):.0f} max={rejected_chars.max()}")

    # Length balance
    denom = np.maximum(chosen_chars, rejected_chars).astype(float)
    denom = np.where(denom == 0, 1, denom)
    abs_ratio = np.abs(chosen_chars - rejected_chars) / denom
    signed_ratio = (chosen_chars - rejected_chars) / denom

    print(f"\nLength balance |cho - rej| / max(cho, rej):")
    print(f"  mean={abs_ratio.mean():.3f} median={np.median(abs_ratio):.3f} "
          f"p90={np.percentile(abs_ratio, 90):.3f}")
    for t in (0.10, 0.20, 0.30, 0.50):
        cnt = (abs_ratio <= t).sum()
        print(f"  <= {t:.2f}: {cnt:,} ({100*cnt/n:.1f}%)")

    rej_longer = (signed_ratio < 0).sum()
    cho_longer = (signed_ratio > 0).sum()
    print(f"  rejected longer: {rej_longer:,} ({100*rej_longer/n:.1f}%)")
    print(f"  chosen longer:   {cho_longer:,} ({100*cho_longer/n:.1f}%)")

    # Token estimate
    total_chars = int(chosen_chars.sum() + rejected_chars.sum()
                      + sum(sum(len(t["content"]) for t in r["fork"]) for r in rows))
    print(f"\nEstimated total tokens (char/4): {total_chars // 4:,}")
    print(f"Avg tokens/pair: {total_chars // 4 // max(n, 1):,}")

    # Synthesis stats
    print_synthesis_stats(rows)

    # Length matched
    matched = sum(1 for r in rows if r.get("length_matched", False))
    print(f"\nLength matched: {matched:,}/{n:,} ({100*matched/n:.1f}%)")


def print_synthesis_stats(rows):
    n = len(rows)
    asst_reps = np.array([r.get("asst_replacements", 0) for r in rows])
    user_reps = np.array([r.get("user_replacements", 0) for r in rows])

    touched = ((asst_reps > 0) | (user_reps > 0)).sum()
    untouched = n - touched

    print(f"\n{'='*60}")
    print(f"Synthesis stats")
    print(f"{'='*60}")
    print(f"  Untouched (skip-matched): {untouched:,} ({100*untouched/n:.1f}%)")
    print(f"  Touched (any synthesis):  {touched:,} ({100*touched/n:.1f}%)")

    asst_any = (asst_reps > 0).sum()
    user_any = (user_reps > 0).sum()
    both = ((asst_reps > 0) & (user_reps > 0)).sum()
    print(f"\n  Asst replacement only:    {asst_any - both:,}")
    print(f"  User expansion only:      {user_any - both:,}")
    print(f"  Both asst + user:         {both:,}")

    if asst_any > 0:
        asst_nz = asst_reps[asst_reps > 0]
        print(f"\n  Asst replacements per pair (when > 0):")
        print(f"    mean={asst_nz.mean():.1f} median={np.median(asst_nz):.0f} "
              f"max={asst_nz.max()}")
        dist = Counter(asst_nz.tolist())
        for k in sorted(dist):
            cnt = dist[k]
            print(f"      {k}: {cnt:,}")

    if user_any > 0:
        user_nz = user_reps[user_reps > 0]
        print(f"\n  User expansions per pair (when > 0):")
        print(f"    mean={user_nz.mean():.1f} median={np.median(user_nz):.0f} "
              f"max={user_nz.max()}")
        dist = Counter(user_nz.tolist())
        for k in sorted(dist):
            cnt = dist[k]
            print(f"      {k}: {cnt:,}")

    # Synthetic turn flag check (inside JSON)
    synth_turns_total = 0
    synth_asst = 0
    synth_user = 0
    for r in rows:
        for t in r["rejected_branch"]:
            if t.get("synthetic", False):
                synth_turns_total += 1
                if t["role"] == "assistant":
                    synth_asst += 1
                else:
                    synth_user += 1

    print(f"\n  Synthetic turn flags in rejected branches:")
    print(f"    Total synthetic turns:  {synth_turns_total:,}")
    print(f"    Synthetic assistant:    {synth_asst:,}")
    print(f"    Synthetic user:         {synth_user:,}")


def print_samples(rows, num_samples, seed, print_all=False):
    if not rows:
        return

    rng = random.Random(seed)
    samples = rng.sample(rows, min(num_samples, len(rows)))

    for idx, r in enumerate(samples):
        touched = r.get("asst_replacements", 0) > 0 or r.get("user_replacements", 0) > 0
        tag = "[synthetic]" if touched else "[natural]"

        print(f"\n{'='*60}")
        print(f"Sample {idx+1}/{len(samples)}  {tag}")
        print(f"  post_id:       {r['post_id']}")
        print(f"  source:        {r['source']}")
        print(f"  subreddit:     {r.get('subreddit', '')}")
        print(f"  post_url:      {r.get('post_url', '')}")
        print(f"  fork_depth:    {r['fork_depth']}")
        print(f"  chosen_depth:  {r['chosen_depth']}")
        print(f"  rejected_depth:{len(r['rejected_branch'])}")
        print(f"  chosen_score:  {r['chosen_score']}")
        print(f"  rejected_score:{r['rejected_score']}")
        print(f"  score_delta:   {r['score_delta']}")
        print(f"  chosen_chars:  {r['chosen_chars']}")
        print(f"  rejected_chars:{r['rejected_chars']}")
        print(f"  asst_replaced: {r.get('asst_replacements', 0)}")
        print(f"  user_replaced: {r.get('user_replacements', 0)}")
        print(f"  length_matched:{r.get('length_matched', '?')}")

        max_chars = None if print_all else 300

        print(f"\n  -- Fork ({len(r['fork'])} turns) --")
        _print_turns(r["fork"], max_chars)

        print(f"\n  -- Chosen branch ({len(r['chosen_branch'])} turns) --")
        _print_turns(r["chosen_branch"], max_chars, show_score=True)

        print(f"\n  -- Rejected branch ({len(r['rejected_branch'])} turns) --")
        _print_turns(r["rejected_branch"], max_chars, show_score=True,
                     show_synthetic=True)

        print(f"{'='*60}")


def _print_turns(turns, max_chars=None, show_score=False, show_synthetic=False):
    for i, t in enumerate(turns):
        content = t["content"]
        if max_chars and len(content) > max_chars:
            content = content[:max_chars] + "..."
        content = content.replace("\n", "\n      ")

        tags = []
        if show_score and "score" in t:
            tags.append(f"score={t['score']}")
        if show_synthetic and t.get("synthetic", False):
            tags.append("SYNTHETIC")
        tag_str = f" ({', '.join(tags)})" if tags else ""

        print(f"    [{i}] [{t['role']}{tag_str}]")
        print(f"      {content}")
        print()


def main():
    args = parse_args()
    rows = load_pairs(
        args.data_path,
        source=args.source,
        subreddit=args.subreddit,
        post_id=args.post_id,
        synthetic_only=args.synthetic_only,
        untouched_only=args.untouched_only,
        max_rows=args.max_rows,
    )
    print_stats(rows)
    print_samples(rows, args.samples, args.seed, args.print_all)


if __name__ == "__main__":
    main()
