"""
winnow.py
=========
Shingling & Winnowing Fingerprints.
PDF Reference: Step 4 — Shingling & Winnowing Fingerprints

Robust to small edits and rewording.
Used by Google and Turnitin under the hood.

DSA Concepts:
  - Sliding window (k-shingles)
  - Rolling minimum (winnowing)
  - Hash table (inverted index)
  - Set intersection for overlap detection

Algorithm:
  1. Tokenize text → overlapping k-word shingles
  2. Hash each shingle (Blake2b)
  3. Winnow: slide window of size w, keep only minimum hash
     (guarantee: any common substring of length >= t is detected)
  4. Build inverted index: hash → [(doc_id, position)]
  5. Match shared fingerprints between submission and reference
"""

import hashlib
from collections import defaultdict


# ── Shingle Generator ─────────────────────────────────────────

def shingle_tokens(text: str, k: int = 5):
    """
    Generate k-word shingles from text.
    Yields (token_position, shingle_string).
    Time: O(n) sliding window
    """
    tokens = text.split()
    for i in range(len(tokens) - k + 1):
        yield i, ' '.join(tokens[i:i + k])


def hash_shingle(s: str) -> int:
    """
    Hash a shingle string using Blake2b.
    PDF Step 4 exact implementation.
    Returns 64-bit integer hash.
    """
    return int(
        hashlib.blake2b(s.encode(), digest_size=8).hexdigest(),
        16
    )


# ── Winnowing ─────────────────────────────────────────────────

def winnow_fingerprints(text: str, k: int = 5, t: int = 9) -> list:
    """
    Compute winnowing fingerprints.
    PDF Step 4 exact implementation.

    Parameters:
        text : normalized document text
        k    : shingle size (words)
        t    : guarantee threshold (any match >= t words is detected)

    Returns:
        List of (token_position, hash_value) — the fingerprints.

    Window size w = t - k + 1
    Time: O(n)
    """
    w = max(1, t - k + 1)

    # Build hash list from all shingles
    hashes = []
    for pos, sh in shingle_tokens(text, k):
        hashes.append((pos, hash_shingle(sh)))

    if len(hashes) < w:
        return hashes  # too short to winnow

    # Sliding window minimum selection
    mins      = []
    last_min  = None

    for i in range(len(hashes) - w + 1):
        window   = hashes[i:i + w]
        mpos, mv = min(window, key=lambda x: x[1])  # minimum hash in window
        if (mpos, mv) != last_min:
            mins.append((mpos, mv))
            last_min = (mpos, mv)

    return mins


# ── Fingerprint Index ─────────────────────────────────────────

def fp_index(texts: dict, k: int = 5, t: int = 9) -> dict:
    """
    Build inverted fingerprint index.
    PDF Step 4 exact implementation.

    Parameters:
        texts : { doc_id: normalized_text }

    Returns:
        { hash_value: [(doc_id, position)] }
    """
    inv = defaultdict(list)
    for doc_id, txt in texts.items():
        for pos, h in winnow_fingerprints(txt, k=k, t=t):
            inv[h].append((doc_id, pos))
    return dict(inv)


# ── Overlap Computation ───────────────────────────────────────

def winnow_overlap(text_a: str, text_b: str,
                   k: int = 5, t: int = 9) -> float:
    """
    Compute winnowing overlap between two documents.
    Returns fraction of submission fingerprints found in reference.
    Range: 0.0 → 1.0
    """
    fps_a = {h for _, h in winnow_fingerprints(text_a, k, t)}
    fps_b = {h for _, h in winnow_fingerprints(text_b, k, t)}

    if not fps_a:
        return 0.0

    shared = fps_a & fps_b
    return len(shared) / len(fps_a)


def print_winnow_results(text_a: str, text_b: str, k: int = 5, t: int = 9):
    """Print winnowing analysis."""
    fps_a = list(winnow_fingerprints(text_a, k, t))
    fps_b = list(winnow_fingerprints(text_b, k, t))
    set_a = {h for _, h in fps_a}
    set_b = {h for _, h in fps_b}
    shared = set_a & set_b
    overlap = len(shared) / max(1, len(set_a))

    print(f"\n{'═'*62}")
    print("  🔍  WINNOWING FINGERPRINT RESULTS")
    print(f"{'═'*62}")
    print(f"  Shingle size (k)      : {k} words")
    print(f"  Guarantee threshold   : {t} words")
    print(f"  Window size (w=t-k+1) : {t-k+1}")
    print(f"\n  Original  fingerprints : {len(fps_a)}")
    print(f"  Submitted fingerprints : {len(fps_b)}")
    print(f"  Shared fingerprints    : {len(shared)}")
    print(f"  Winnow Overlap         : {overlap*100:.1f}%")

    if fps_a:
        print(f"\n  Sample fingerprints (original, first 3):")
        for pos, h in fps_a[:3]:
            tokens = text_a.split()[pos:pos+k]
            print(f"    [{pos}] \"{' '.join(tokens)}\" → hash={h}")

    print(f"{'═'*62}\n")
