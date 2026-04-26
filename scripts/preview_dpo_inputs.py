#!/usr/bin/env python3
"""
scripts/preview_dpo_inputs.py

Show exactly what the DPO trainer sees for a sample from the fork-based
DPO dataset (generate_dpo_rejected.py output).

Catches silent dataloader / chat-template / truncation bugs before a
training run wastes GPU time.

  What it shows per pair

  - Raw turn counts (fork / chosen_branch / rejected_branch).
  - Synthetic turn markers in rejected branch.
  - The exact string after tokenizer.apply_chat_template for
    fork-only, fork+chosen, fork+rejected.
  - Token counts and truncation flags.
  - A colored loss-mask view: dimmed = prompt (no loss),
    green = response (loss applied).

  Assertions (per pair)

  1. Fork IDs are a strict prefix of fork+chosen IDs.
  2. Fork IDs are a strict prefix of fork+rejected IDs.
  3. Chosen and rejected share identical fork token IDs.
  4. Chosen/rejected response are non-empty after truncation.
  5. EOS present at end (warn-only).
  6. Both branches end on assistant role.
  7. Template is tokenization-stable (warn-only).

  Aggregate (--scan-all)

  - Histograms of fork / chosen / rejected token lengths.
  - Count of pairs hit by --max-length truncation on each side.

Usage:
    python scripts/preview_dpo_inputs.py \
        --data-path data/dpo_unweighted_sft_final.parquet \
        --tokenizer Qwen/Qwen2.5-7B-Instruct \
        --samples 3 \
        --max-length 2048

    # Only synthetic pairs:
    python scripts/preview_dpo_inputs.py \
        --data-path data/dpo_unweighted_sft_final.parquet \
        --tokenizer Qwen/Qwen2.5-7B-Instruct \
        --synthetic-only --samples 2

    # Full audit:
    python scripts/preview_dpo_inputs.py \
        --data-path data/dpo_unweighted_sft_final.parquet \
        --tokenizer Qwen/Qwen2.5-7B-Instruct \
        --scan-all --max-length 2048

    # Specific post:
    python scripts/preview_dpo_inputs.py \
        --data-path data/dpo_unweighted_sft_final.parquet \
        --tokenizer Qwen/Qwen2.5-7B-Instruct \
        --post-id 1abc23

    # Strict mode (exit non-zero on FAIL):
    python scripts/preview_dpo_inputs.py \
        --data-path data/dpo_unweighted_sft_final.parquet \
        --tokenizer Qwen/Qwen2.5-7B-Instruct \
        --strict --scan-all --max-length 2048

    # Save to file:
    python scripts/preview_dpo_inputs.py \
        --data-path data/dpo_unweighted_sft_final.parquet \
        --tokenizer Qwen/Qwen2.5-7B-Instruct \
        --out preview.txt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple


# -----------------------------------------------------------------
# Loading
# -----------------------------------------------------------------

def load_dpo_records(data_path: str) -> List[Dict]:
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
            row["fork"] = json.loads(row["fork"])
            row["chosen_branch"] = json.loads(row["chosen_branch"])
            row["rejected_branch"] = json.loads(row["rejected_branch"])
            rows.append(row)
    return rows


# -----------------------------------------------------------------
# Message helpers
# -----------------------------------------------------------------

def _strip(turns: List[Dict]) -> List[Dict]:
    return [{"role": t["role"], "content": t["content"]} for t in turns]


# -----------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------

RESET = "\033[0m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"


def _color(s: str, code: str, enable: bool) -> str:
    return f"{code}{s}{RESET}" if enable else s


def _section(title: str, fh) -> None:
    print(f"\n{'='*72}", file=fh)
    print(f"  {title}", file=fh)
    print(f"{'='*72}", file=fh)


def _sub(title: str, fh) -> None:
    print(f"\n{'-'*72}", file=fh)
    print(f"  {title}", file=fh)
    print(f"{'-'*72}", file=fh)


# -----------------------------------------------------------------
# Tokenization
# -----------------------------------------------------------------

def _template_tokenize(tokenizer, msgs: List[Dict],
                       add_generation_prompt: bool) -> List[int]:
    out = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=add_generation_prompt
    )
    if isinstance(out, str):
        return tokenizer(out, add_special_tokens=False)["input_ids"]
    if hasattr(out, "ids") and not isinstance(out, (list, tuple)):
        out = out.ids
    if hasattr(out, "keys") and "input_ids" in out.keys():
        out = out["input_ids"]
    if hasattr(out, "tolist") and not isinstance(out, list):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    return [int(x) for x in out]


def _encode_branch(
    tokenizer,
    fork_msgs: List[Dict],
    branch_msgs: List[Dict],
    max_length: Optional[int],
) -> Dict:
    fork_ids = _template_tokenize(tokenizer, fork_msgs,
                                  add_generation_prompt=True)
    full_ids = _template_tokenize(tokenizer, fork_msgs + branch_msgs,
                                  add_generation_prompt=False)

    truncated = False
    if max_length is not None and len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        truncated = True

    fork_str = tokenizer.decode(fork_ids, skip_special_tokens=False)
    full_str = tokenizer.decode(full_ids, skip_special_tokens=False)

    return {
        "fork_str": fork_str,
        "full_str": full_str,
        "fork_ids": fork_ids,
        "full_ids": full_ids,
        "truncated": truncated,
    }


# -----------------------------------------------------------------
# Assertions
# -----------------------------------------------------------------

def _check_prefix(
    fork_ids: List[int],
    full_ids: List[int],
    label: str,
    fh,
) -> Tuple[bool, int]:
    L = min(len(fork_ids), len(full_ids))
    mismatch = next((i for i in range(L) if fork_ids[i] != full_ids[i]), None)
    if mismatch is not None:
        print(f"  [FAIL] {label}: fork not a prefix of full at token {mismatch}",
              file=fh)
        print(f"         fork[{mismatch}]={fork_ids[mismatch]} "
              f"full[{mismatch}]={full_ids[mismatch]}", file=fh)
        return False, len(fork_ids)
    if len(full_ids) < len(fork_ids):
        print(f"  [FAIL] {label}: full truncated below fork length "
              f"({len(full_ids)} < {len(fork_ids)})", file=fh)
        return False, len(fork_ids)
    return True, len(fork_ids)


def _eos_ok(tokenizer, full_ids: List[int]) -> bool:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        return True
    return full_ids[-1] == eos_id


def _render_loss_mask(
    tokenizer,
    full_ids: List[int],
    response_start: int,
    use_color: bool,
    fh,
) -> None:
    prompt_part = tokenizer.decode(full_ids[:response_start],
                                   skip_special_tokens=False)
    response_part = tokenizer.decode(full_ids[response_start:],
                                     skip_special_tokens=False)
    n_prompt = response_start
    n_resp = len(full_ids) - response_start
    print(_color(f"[FORK/PROMPT (masked) -- {n_prompt} tokens]",
                 DIM, use_color), file=fh)
    print(_color(prompt_part, DIM, use_color), file=fh)
    print(_color(f"[BRANCH (loss applied) -- {n_resp} tokens]",
                 GREEN, use_color), file=fh)
    print(_color(response_part, GREEN, use_color), file=fh)


# -----------------------------------------------------------------
# Per-pair preview
# -----------------------------------------------------------------

def preview_pair(
    tokenizer,
    record: Dict,
    max_length: Optional[int],
    use_color: bool,
    fh,
) -> Dict:
    fork_msgs = _strip(record["fork"])
    chosen_msgs = _strip(record["chosen_branch"])
    rejected_msgs = _strip(record["rejected_branch"])

    src = record.get("source", "?")
    sub = record.get("subreddit", "")
    post_id = record.get("post_id", "?")
    delta = record.get("score_delta", 0.0)
    asst_rep = record.get("asst_replacements", 0)
    user_rep = record.get("user_replacements", 0)
    touched = asst_rep > 0 or user_rep > 0
    tag = "synthetic" if touched else "natural"

    _sub(f"[{tag}] source={src}  sub={sub}  post_id={post_id}  "
         f"delta={delta:.1f}", fh)

    print(f"\n  fork turns:       {len(fork_msgs)}", file=fh)
    print(f"  chosen turns:     {len(chosen_msgs)}", file=fh)
    print(f"  rejected turns:   {len(rejected_msgs)}", file=fh)
    if touched:
        print(f"  asst_replaced:    {asst_rep}", file=fh)
        print(f"  user_replaced:    {user_rep}", file=fh)

    # Show synthetic markers in rejected branch
    synth_indices = [i for i, t in enumerate(record["rejected_branch"])
                     if t.get("synthetic", False)]
    if synth_indices:
        print(f"  synthetic turns:  {synth_indices}", file=fh)

    chosen_enc = _encode_branch(tokenizer, fork_msgs, chosen_msgs, max_length)
    rejected_enc = _encode_branch(tokenizer, fork_msgs, rejected_msgs,
                                  max_length)

    print("\n  --- templated fork (shared prefix) ---", file=fh)
    print(chosen_enc["fork_str"], file=fh)
    print("\n  --- templated fork+chosen ---", file=fh)
    print(chosen_enc["full_str"], file=fh)
    print("\n  --- templated fork+rejected ---", file=fh)
    print(rejected_enc["full_str"], file=fh)

    print("\n  --- token counts ---", file=fh)
    print(f"  fork_tokens:     {len(chosen_enc['fork_ids'])}", file=fh)
    print(f"  chosen_tokens:   {len(chosen_enc['full_ids'])}  "
          f"(truncated={chosen_enc['truncated']})", file=fh)
    print(f"  rejected_tokens: {len(rejected_enc['full_ids'])}  "
          f"(truncated={rejected_enc['truncated']})", file=fh)

    print("\n  --- assertions ---", file=fh)
    results = {"checks_passed": 0, "checks_failed": 0, "warnings": []}

    def _mark(ok: bool, msg: str, warn: bool = False) -> None:
        if ok:
            print(f"  [OK]   {msg}", file=fh)
            results["checks_passed"] += 1
        else:
            t = "[WARN]" if warn else "[FAIL]"
            print(f"  {t} {msg}", file=fh)
            if warn:
                results["warnings"].append(msg)
            else:
                results["checks_failed"] += 1

    # 1. Chosen prefix
    c_ok, c_resp_start = _check_prefix(
        chosen_enc["fork_ids"], chosen_enc["full_ids"], "chosen", fh)
    _mark(c_ok, "chosen: fork tokens are a prefix of fork+chosen tokens")

    # 2. Rejected prefix
    r_ok, r_resp_start = _check_prefix(
        rejected_enc["fork_ids"], rejected_enc["full_ids"], "rejected", fh)
    _mark(r_ok, "rejected: fork tokens are a prefix of fork+rejected tokens")

    # 3. Shared fork tokens
    shared_ok = chosen_enc["fork_ids"] == rejected_enc["fork_ids"]
    _mark(shared_ok, "chosen and rejected share identical fork token IDs")

    # 4. Non-empty responses after truncation
    c_resp_len = len(chosen_enc["full_ids"]) - c_resp_start
    r_resp_len = len(rejected_enc["full_ids"]) - r_resp_start
    _mark(c_resp_len > 0,
          f"chosen branch non-empty after truncation (len={c_resp_len})")
    _mark(r_resp_len > 0,
          f"rejected branch non-empty after truncation (len={r_resp_len})")

    # 5. EOS at end
    _mark(_eos_ok(tokenizer, chosen_enc["full_ids"]),
          "chosen ends with EOS token", warn=True)
    _mark(_eos_ok(tokenizer, rejected_enc["full_ids"]),
          "rejected ends with EOS token", warn=True)

    # 6. Branches end on assistant (or user if role-swapped even-fork pair)
    cho_last = chosen_msgs[-1]["role"] if chosen_msgs else "?"
    rej_last = rejected_msgs[-1]["role"] if rejected_msgs else "?"
    _mark(cho_last in ("assistant", "user"),
          f"chosen branch ends on '{cho_last}'", warn=cho_last != "assistant")
    _mark(rej_last in ("assistant", "user"),
          f"rejected branch ends on '{rej_last}'", warn=rej_last != "assistant")

    # 7. Template stability
    render_str = tokenizer.apply_chat_template(
        fork_msgs + chosen_msgs, tokenize=False, add_generation_prompt=False)
    retok_ids = tokenizer(render_str, add_special_tokens=False)["input_ids"]
    stable = retok_ids == chosen_enc["full_ids"] or chosen_enc["truncated"]
    _mark(stable,
          "chat template is tokenization-stable "
          "(tokenize=False -> retokenize matches tokenize=True)",
          warn=True)

    # 8. Role alternation in full conversation
    full_chosen = fork_msgs + chosen_msgs
    full_rejected = fork_msgs + rejected_msgs
    cho_alt = _check_role_alternation(full_chosen)
    rej_alt = _check_role_alternation(full_rejected)
    _mark(cho_alt, "chosen: role alternation is valid (user/assistant)")
    _mark(rej_alt, "rejected: role alternation is valid (user/assistant)")

    # Loss-mask visualization
    print("\n  --- loss mask (chosen) ---", file=fh)
    _render_loss_mask(tokenizer, chosen_enc["full_ids"], c_resp_start,
                      use_color, fh)
    print("\n  --- loss mask (rejected) ---", file=fh)
    _render_loss_mask(tokenizer, rejected_enc["full_ids"], r_resp_start,
                      use_color, fh)

    results.update({
        "fork_tokens": len(chosen_enc["fork_ids"]),
        "chosen_tokens": len(chosen_enc["full_ids"]),
        "rejected_tokens": len(rejected_enc["full_ids"]),
        "chosen_truncated": chosen_enc["truncated"],
        "rejected_truncated": rejected_enc["truncated"],
    })
    return results


def _check_role_alternation(msgs: List[Dict]) -> bool:
    for i in range(1, len(msgs)):
        if msgs[i]["role"] == msgs[i - 1]["role"]:
            return False
    return True


# -----------------------------------------------------------------
# Aggregate histograms
# -----------------------------------------------------------------

def _hist(values: List[int], label: str, fh, bins: int = 10) -> None:
    if not values:
        return
    vmin, vmax = min(values), max(values)
    mean = sum(values) / len(values)
    print(f"\n  {label}: n={len(values)} min={vmin} mean={mean:.0f} max={vmax}",
          file=fh)
    if vmin == vmax:
        return
    step = max((vmax - vmin) // bins, 1)
    counts = [0] * bins
    for v in values:
        idx = min((v - vmin) // step, bins - 1)
        counts[idx] += 1
    peak = max(counts) or 1
    for i, c in enumerate(counts):
        lo = vmin + i * step
        hi = lo + step
        bar = "#" * int(40 * c / peak)
        print(f"    [{lo:>6} .. {hi:>6}) {c:>6}  {bar}", file=fh)


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Sanity-check DPO pair parquets before training")
    p.add_argument("--data-path", required=True,
                   help="Parquet file or directory of shards")
    p.add_argument("--tokenizer", required=True,
                   help="HF tokenizer name or local checkpoint path")
    p.add_argument("--samples", type=int, default=3,
                   help="Number of pairs to preview in full")
    p.add_argument("--max-length", type=int, default=None,
                   help="Truncation length matching your trainer config")
    p.add_argument("--source", default=None)
    p.add_argument("--subreddit", default=None)
    p.add_argument("--post-id", default=None)
    p.add_argument("--synthetic-only", action="store_true",
                   help="Only preview pairs with synthetic replacements")
    p.add_argument("--untouched-only", action="store_true",
                   help="Only preview pairs with no synthesis")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap number of rows loaded")
    p.add_argument("--out", default=None,
                   help="Write output to file instead of stdout")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--scan-all", action="store_true",
                   help="Tokenize every pair for length histograms and "
                        "truncation stats")
    p.add_argument("--strict", action="store_true",
                   help="Exit non-zero if any FAIL assertion fires")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    fh = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    use_color = (not args.no_color) and (args.out is None) and sys.stdout.isatty()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("pip install transformers", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Loading tokenizer: {args.tokenizer}", file=fh)
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True)
        if tokenizer.chat_template is None:
            print("  [WARN] tokenizer has no chat_template -- "
                  "apply_chat_template will use a default.", file=fh)

        print(f"Loading DPO records from: {args.data_path}", file=fh)
        records = load_dpo_records(args.data_path)

        # Filters
        if args.source:
            records = [r for r in records if r.get("source") == args.source]
        if args.subreddit:
            records = [r for r in records
                       if r.get("subreddit") == args.subreddit]
        if args.post_id:
            records = [r for r in records
                       if str(r.get("post_id", "")) == args.post_id]
        if args.synthetic_only:
            records = [r for r in records
                       if r.get("asst_replacements", 0) > 0
                       or r.get("user_replacements", 0) > 0]
        if args.untouched_only:
            records = [r for r in records
                       if r.get("asst_replacements", 0) == 0
                       and r.get("user_replacements", 0) == 0]
        if args.max_rows:
            records = records[:args.max_rows]

        if not records:
            print("No records match filters.", file=fh)
            return
        print(f"  {len(records):,} pairs loaded", file=fh)

        # Preview samples
        if args.post_id:
            picked = records
        else:
            rng = random.Random(args.seed)
            picked = rng.sample(records, min(args.samples, len(records)))

        _section(f"Previewing {len(picked)} pair(s)", fh)
        summaries = []
        for rec in picked:
            summaries.append(preview_pair(
                tokenizer, rec, args.max_length, use_color, fh))

        total_fail = sum(s["checks_failed"] for s in summaries)
        total_pass = sum(s["checks_passed"] for s in summaries)
        all_warnings = [w for s in summaries for w in s["warnings"]]

        _section(f"Preview: {total_pass} OK, {total_fail} FAILED, "
                 f"{len(all_warnings)} warnings", fh)

        if args.strict and total_fail > 0:
            print(f"\n--strict: {total_fail} FAIL(s). "
                  f"Fix data before training.", file=fh)
            sys.exit(2)

        # Full scan
        if args.scan_all:
            _section("Token-length scan across all pairs", fh)
            fork_lens, chosen_lens, rejected_lens = [], [], []
            trunc_c, trunc_r = 0, 0
            scan_fails = 0
            role_alt_fails = 0

            for i, rec in enumerate(records):
                fork_msgs = _strip(rec["fork"])
                chosen_msgs = _strip(rec["chosen_branch"])
                rejected_msgs = _strip(rec["rejected_branch"])

                c_enc = _encode_branch(tokenizer, fork_msgs, chosen_msgs,
                                       args.max_length)
                r_enc = _encode_branch(tokenizer, fork_msgs, rejected_msgs,
                                       args.max_length)

                fork_lens.append(len(c_enc["fork_ids"]))
                chosen_lens.append(len(c_enc["full_ids"]))
                rejected_lens.append(len(r_enc["full_ids"]))
                trunc_c += int(c_enc["truncated"])
                trunc_r += int(r_enc["truncated"])

                # Prefix check
                c_ok = c_enc["fork_ids"] == r_enc["fork_ids"]
                if not c_ok:
                    scan_fails += 1

                # Role alternation
                full_c = fork_msgs + chosen_msgs
                full_r = fork_msgs + rejected_msgs
                if not _check_role_alternation(full_c):
                    role_alt_fails += 1
                if not _check_role_alternation(full_r):
                    role_alt_fails += 1

                if (i + 1) % 10000 == 0:
                    print(f"  scanned {i+1:,}/{len(records):,}...", file=fh)

            _hist(fork_lens, "fork tokens (shared prefix)", fh)
            _hist(chosen_lens, "fork+chosen tokens", fh)
            _hist(rejected_lens, "fork+rejected tokens", fh)

            n = len(records)
            if args.max_length is not None:
                print(f"\n  truncated chosen:   {trunc_c}/{n} "
                      f"({100*trunc_c/n:.1f}%)", file=fh)
                print(f"  truncated rejected: {trunc_r}/{n} "
                      f"({100*trunc_r/n:.1f}%)", file=fh)

            print(f"\n  fork mismatch (chosen vs rejected): {scan_fails}/{n}",
                  file=fh)
            print(f"  role alternation failures: {role_alt_fails}/{2*n} "
                  f"branches", file=fh)

            if args.strict and (scan_fails > 0 or role_alt_fails > 0):
                print(f"\n--strict: scan found {scan_fails} fork mismatches, "
                      f"{role_alt_fails} role alternation failures.", file=fh)
                sys.exit(2)

    finally:
        if fh is not sys.stdout:
            fh.close()


if __name__ == "__main__":
    main()
