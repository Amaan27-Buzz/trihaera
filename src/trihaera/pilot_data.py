"""Tiny reproducible packed-text windows for adapter pilots.

This is a small-data controlled test harness, not TRIHAERA's future
foundation-model training pipeline.
"""

import hashlib
import random
import sys
from array import array

import torch


def pack_text_windows(rows, tokenizer, *, sequence_length: int,
                      max_windows: int, max_rows: int,
                      seed: int = 139) -> tuple[torch.Tensor, dict]:
    """Select example rows deterministically, encode and pack nonoverlapping windows.

    Rows are official train or validation split records (dicts with 'text').
    Selected examples are concatenated with EOS separators; do not treat as
    document-preserving training. No split mixing occurs.
    """
    if not 8 <= sequence_length <= 8192 or max_windows < 1 or max_rows < 1:
        raise ValueError("Invalid sequence length/window/row budget")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    indices = indices[:max_rows]
    all_ids = []
    selected_rows = 0
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer must define an EOS token")
    for index in indices:
        text = rows[index]["text"]
        if not text or not text.strip():
            continue
        tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
        if not tokens:
            continue
        all_ids.extend(tokens)
        all_ids.append(int(eos))
        selected_rows += 1
        if len(all_ids) >= max_windows * sequence_length:
            break
    window_count = min(max_windows, len(all_ids) // sequence_length)
    if window_count < 2:
        raise ValueError("Not enough real text to form at least two windows")
    used_ids = all_ids[:window_count * sequence_length]
    windows = torch.tensor(used_ids, dtype=torch.long).reshape(
        window_count, sequence_length
    )
    # Avoid an undeclared NumPy runtime dependency in CPU-only CI.
    # Use explicit little-endian uint64 tokens for a platform-stable checksum.
    raw = array("Q", used_ids)
    if sys.byteorder != "little":
        raw.byteswap()
    checksum = hashlib.sha256(raw.tobytes()).hexdigest()
    return windows, {
        "num_windows": window_count,
        "sequence_length": sequence_length,
        "selected_nonempty_rows": selected_rows,
        "token_count": window_count * sequence_length,
        "window_sha256": checksum,
        "method": "deterministically sampled rows, EOS-separated, nonoverlap packed",
    }
