#!/usr/bin/env python3
"""
Tone attunement scorer (feature-based).

Scores how well assistant responses attune to user tone using stylometric
features. Measures register similarity -- not semantic similarity -- to
detect and penalize documentary/wikipedia tone in assistant responses.

Features per text:
  - avg sentence length (words)
  - avg word length (chars)
  - contraction rate
  - first/second person pronoun rate
  - question rate
  - exclamation rate
  - hedge word rate
  - informal marker rate
  - punctuation density

Pipeline:
  1. Extract (user, assistant) turn pairs per conversation
  2. Compute feature vectors for each text
  3. Z-normalize features across the full dataset
  4. Cosine similarity between user and assistant feature vectors per pair
  5. Average across pairs -> attunement_score per conversation

Supports two input formats:
  - SFT chains: parquet with `turns` column (from extract_sft_chains.py)
  - DPO pairs:  parquet with `prompt`/`chosen` columns (legacy format)

Auto-detects format from column names.

Usage:
    # SFT chains (new format):
    python scripts/tone_scorer.py \
        --data-dir data/sft_chains \
        --output-path data/sft_chains_scored.parquet

    # DPO pairs (legacy format):
    python scripts/tone_scorer.py \
        --data-dir data/clean_v3/upload \
        --output-path data/clean_v3/scored.parquet
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CONTRACTIONS = re.compile(
    r"\b(?:i'm|i've|i'd|i'll|you're|you've|you'd|you'll|he's|she's|it's|"
    r"we're|we've|we'd|we'll|they're|they've|they'd|they'll|"
    r"isn't|aren't|wasn't|weren't|don't|doesn't|didn't|won't|wouldn't|"
    r"can't|couldn't|shouldn't|mustn't|hasn't|haven't|hadn't|"
    r"that's|there's|here's|what's|who's|how's|let's|"
    r"ain't|gonna|wanna|gotta|kinda|sorta)\b",
    re.IGNORECASE,
)

FIRST_SECOND_PRONOUNS = re.compile(
    r"\b(?:i|me|my|mine|myself|we|us|our|ours|ourselves|"
    r"you|your|yours|yourself|yourselves)\b",
    re.IGNORECASE,
)

HEDGE_WORDS = re.compile(
    r"\b(?:maybe|perhaps|probably|possibly|apparently|basically|"
    r"actually|honestly|frankly|literally|"
    r"kind of|sort of|i think|i guess|i feel like|i mean|"
    r"not sure|not certain|in my opinion|imo)\b",
    re.IGNORECASE,
)

INFORMAL_MARKERS = re.compile(
    r"\b(?:lol|lmao|omg|tbh|imo|imho|idk|smh|fwiw|iirc|"
    r"yeah|yep|yup|nah|nope|haha|heh|hmm|"
    r"gonna|wanna|gotta|kinda|sorta|"
    r"ok|okay|cool|dude|bro|man|guys|"
    r"btw|fyi|tho|tl;dr|af)\b",
    re.IGNORECASE,
)

PUNCTUATION = re.compile(r"[.,;:!?\-\"'()\[\]{}]")


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in parts if s]


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text)


def extract_features(text: str) -> np.ndarray:
    sentences = _split_sentences(text)
    words = _tokenize_words(text)
    n_sentences = max(len(sentences), 1)
    n_words = max(len(words), 1)
    n_chars = max(len(text), 1)

    avg_sent_len = n_words / n_sentences
    avg_word_len = sum(len(w) for w in words) / n_words
    contraction_rate = len(CONTRACTIONS.findall(text)) / n_words
    pronoun_rate = len(FIRST_SECOND_PRONOUNS.findall(text)) / n_words
    question_rate = sum(1 for s in sentences if s.rstrip().endswith("?")) / n_sentences
    exclamation_rate = sum(1 for s in sentences if s.rstrip().endswith("!")) / n_sentences
    hedge_rate = len(HEDGE_WORDS.findall(text)) / n_words
    informal_rate = len(INFORMAL_MARKERS.findall(text)) / n_words
    punct_density = len(PUNCTUATION.findall(text)) / n_chars

    return np.array([
        avg_sent_len,
        avg_word_len,
        contraction_rate,
        pronoun_rate,
        question_rate,
        exclamation_rate,
        hedge_rate,
        informal_rate,
        punct_density,
    ], dtype=np.float64)


# ----------------------------------------------------------------
# Turn pair extraction (format-agnostic)
# ----------------------------------------------------------------

def _pairs_from_turns(turns: list[dict]) -> list[tuple[str, str, int]]:
    """Extract adjacent (user, assistant) pairs with the assistant's index in the turn list."""
    pairs = []
    for i in range(len(turns) - 1):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "assistant":
            pairs.append((turns[i]["content"], turns[i + 1]["content"], i + 1))
    return pairs


def _pairs_from_dpo(prompt_json: str, chosen_json: str) -> list[tuple[str, str, int]]:
    """Extract pairs from DPO prompt+chosen format."""
    turns = json.loads(prompt_json) + json.loads(chosen_json)
    return _pairs_from_turns(turns)


def _pairs_from_sft(turns_json: str) -> list[tuple[str, str, int]]:
    """Extract pairs from SFT chain turns format."""
    turns = json.loads(turns_json)
    return _pairs_from_turns(turns)


# ----------------------------------------------------------------
# Format detection and loading
# ----------------------------------------------------------------

def load_data(data_dir: str) -> tuple[pa.Table, str]:
    """Load parquet files and detect format.

    Returns (table, format) where format is "sft" or "dpo".
    """
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No parquet files in {data_dir}")
    tables = [pq.read_table(f) for f in files]
    table = pa.concat_tables(tables)

    col_names = set(table.schema.names)
    if "turns" in col_names:
        fmt = "sft"
    elif "prompt" in col_names and "chosen" in col_names:
        fmt = "dpo"
    else:
        raise ValueError(
            f"Unrecognized format. Columns: {sorted(col_names)}. "
            f"Expected 'turns' (SFT) or 'prompt'+'chosen' (DPO)."
        )

    return table, fmt


# ----------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------

def compute_scores(table: pa.Table, fmt: str) -> tuple[np.ndarray, list[list[tuple[int, float]]]]:
    """Compute attunement scores at both chain and per-turn level.

    Returns:
        chain_scores: (n_rows,) array -- mean attunement per chain.
        per_turn_scores: list of list of (assistant_turn_index, cosine_score)
            per row, one entry per (user, assistant) pair.
    """
    all_user_feats = []
    all_asst_feats = []
    pair_counts = []
    pair_asst_indices: list[list[int]] = []

    if fmt == "sft":
        turns_col = table.column("turns").to_pylist()
        for turns_json in turns_col:
            pairs = _pairs_from_sft(turns_json)
            pair_counts.append(len(pairs))
            indices = []
            for u, a, asst_idx in pairs:
                all_user_feats.append(extract_features(u))
                all_asst_feats.append(extract_features(a))
                indices.append(asst_idx)
            pair_asst_indices.append(indices)
    else:
        prompts = table.column("prompt").to_pylist()
        chosens = table.column("chosen").to_pylist()
        for prompt_json, chosen_json in zip(prompts, chosens):
            pairs = _pairs_from_dpo(prompt_json, chosen_json)
            pair_counts.append(len(pairs))
            indices = []
            for u, a, asst_idx in pairs:
                all_user_feats.append(extract_features(u))
                all_asst_feats.append(extract_features(a))
                indices.append(asst_idx)
            pair_asst_indices.append(indices)

    n = len(table)
    if not all_user_feats:
        return np.zeros(n, dtype=np.float32), [[] for _ in range(n)]

    user_mat = np.stack(all_user_feats)
    asst_mat = np.stack(all_asst_feats)

    # z-normalize each feature across the full dataset (both roles pooled)
    combined = np.concatenate([user_mat, asst_mat], axis=0)
    mu = combined.mean(axis=0)
    sigma = combined.std(axis=0)
    sigma[sigma < 1e-8] = 1.0

    user_normed = (user_mat - mu) / sigma
    asst_normed = (asst_mat - mu) / sigma

    # cosine similarity per pair
    dot = np.sum(user_normed * asst_normed, axis=1)
    user_norm = np.linalg.norm(user_normed, axis=1)
    asst_norm = np.linalg.norm(asst_normed, axis=1)
    denom = user_norm * asst_norm
    denom[denom < 1e-8] = 1.0
    cosines = dot / denom

    chain_scores = np.zeros(n, dtype=np.float32)
    per_turn_scores: list[list[tuple[int, float]]] = []
    idx = 0
    for i, count in enumerate(pair_counts):
        turn_scores = []
        if count > 0:
            chain_scores[i] = cosines[idx:idx + count].mean()
            for j in range(count):
                turn_scores.append((pair_asst_indices[i][j], float(cosines[idx + j])))
        idx += count
        per_turn_scores.append(turn_scores)

    return chain_scores, per_turn_scores


def _embed_turn_scores(turns_json: str, turn_scores: list[tuple[int, float]]) -> str:
    """Write per-turn attunement scores into assistant turn dicts."""
    turns = json.loads(turns_json)
    for asst_idx, score in turn_scores:
        turns[asst_idx]["attunement_score"] = round(score, 6)
    return json.dumps(turns, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-path", required=True)
    args = p.parse_args()

    print(f"Loading data from {args.data_dir}")
    table, fmt = load_data(args.data_dir)
    print(f"Loaded {len(table)} rows (format: {fmt})")

    chain_scores, per_turn_scores = compute_scores(table, fmt)

    print(f"\nAttunement score stats (chain-level):")
    print(f"  mean:   {chain_scores.mean():.4f}")
    print(f"  std:    {chain_scores.std():.4f}")
    print(f"  min:    {chain_scores.min():.4f}")
    print(f"  max:    {chain_scores.max():.4f}")
    print(f"  median: {np.median(chain_scores):.4f}")

    q = np.percentile(chain_scores, [10, 25, 75, 90])
    print(f"  p10:    {q[0]:.4f}")
    print(f"  p25:    {q[1]:.4f}")
    print(f"  p75:    {q[2]:.4f}")
    print(f"  p90:    {q[3]:.4f}")

    # Embed per-turn attunement into turn dicts (SFT format only)
    if fmt == "sft":
        turns_col = table.column("turns").to_pylist()
        updated = [_embed_turn_scores(tj, ts)
                   for tj, ts in zip(turns_col, per_turn_scores)]
        col_idx = table.schema.get_field_index("turns")
        table = table.set_column(col_idx, "turns", pa.array(updated, type=pa.string()))

    table = table.append_column("attunement_score",
                                pa.array(chain_scores, type=pa.float32()))

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    pq.write_table(table, args.output_path)
    print(f"Wrote scored data to {args.output_path}")


if __name__ == "__main__":
    main()
