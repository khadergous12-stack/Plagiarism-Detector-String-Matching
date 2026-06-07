"""
exact.py
========
Exact Matching — KMP & Rabin-Karp.
PDF Reference: Step 3 — Exact Matching (KMP & Rabin-Karp)

Exact implementation from the PDF:
  - KMP for pattern search within text
  - Rabin-Karp windowed scan across submission vs reference

DSA Concepts:
  - KMP: LPS array, failure function O(n+m)
  - Rabin-Karp: Rolling hash, sliding window O(n+m) avg
  - Hash table for reference window lookup
"""

BASE  = 256
PRIME = 10**9 + 7


# ── KMP ──────────────────────────────────────────────────────

def _build_lps(pattern: str) -> list:
    """
    Build LPS (Longest Prefix Suffix) array.
    LPS[i] = length of longest proper prefix of pattern[0..i]
             that is also a suffix.
    Time: O(m)
    """
    m   = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i]  = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


def kmp_search(text: str, pat: str) -> list:
    """
    KMP pattern search.
    PDF Step 3 exact implementation.
    Returns list of start positions where pat found in text.
    Time: O(n + m)
    """
    lps = _build_lps(pat)
    res = []
    j   = 0
    for i, ch in enumerate(text):
        while j > 0 and ch != pat[j]:
            j = lps[j - 1]
        if ch == pat[j]:
            j += 1
        if j == len(pat):
            res.append(i - j + 1)
            j = lps[j - 1]
    return res


def kmp_sentence_match(original_sents: list,
                        submitted_sents: list,
                        min_len: int = 15) -> list:
    """
    Use KMP to find submitted sentences inside original document.
    Returns list of match dicts.
    """
    original_text = ' '.join(original_sents).lower()
    matches = []

    for s_idx, s_sent in enumerate(submitted_sents):
        pat = s_sent.lower().strip()
        if len(pat) < min_len:
            continue
        positions = kmp_search(original_text, pat)
        if positions:
            matches.append({
                'submitted_idx':      s_idx,
                'submitted_sentence': s_sent,
                'positions':          positions,
                'length':             len(pat),
                'algorithm':          'KMP',
            })

    return matches


# ── Rabin-Karp ────────────────────────────────────────────────

def rabin_karp_windows(text: str, ref: str,
                        w: int = 50) -> list:
    """
    Rabin-Karp windowed scan.
    PDF Step 3 exact implementation.

    Slide window of size w over text, probe reference hash table.
    Returns list of (start_in_text, start_in_ref).
    Time: O(n + m) average
    """
    if len(text) < w or len(ref) < w:
        return []

    powp = pow(BASE, w - 1, PRIME)

    # Pre-hash all windows in reference → hash table
    ref_hashes = {}
    hr = 0
    for i in range(w):
        hr = (hr * BASE + ord(ref[i])) % PRIME
    ref_hashes.setdefault(hr, []).append(0)

    for i in range(w, len(ref)):
        hr = ((hr - ord(ref[i - w]) * powp) % PRIME + PRIME) % PRIME
        hr = (hr * BASE + ord(ref[i])) % PRIME
        ref_hashes.setdefault(hr, []).append(i - w + 1)

    # Slide over text and probe
    out = []
    h   = 0
    for i in range(w):
        h = (h * BASE + ord(text[i])) % PRIME

    def check(i, j):
        return text[i:i + w] == ref[j:j + w]

    if h in ref_hashes:
        for j in ref_hashes[h]:
            if check(0, j):
                out.append((0, j))

    for i in range(w, len(text)):
        h = ((h - ord(text[i - w]) * powp) % PRIME + PRIME) % PRIME
        h = (h * BASE + ord(text[i])) % PRIME
        if h in ref_hashes:
            for j in ref_hashes[h]:
                if check(i - w + 1, j):
                    out.append((i - w + 1, j))

    return out


def rk_sentence_match(original_sents: list,
                       submitted_sents: list,
                       min_len: int = 15) -> list:
    """
    Rabin-Karp sentence-level matching using hash table lookup.
    Hash all original sentences → O(1) probe per submitted sentence.
    """
    orig_hashes = {}
    for idx, sent in enumerate(original_sents):
        key = sent.lower().strip()
        orig_hashes.setdefault(hash(key), []).append((idx, key))

    matches = []
    for s_idx, s_sent in enumerate(submitted_sents):
        pat = s_sent.lower().strip()
        if len(pat) < min_len:
            continue
        ph = hash(pat)
        if ph in orig_hashes:
            for o_idx, o_sent in orig_hashes[ph]:
                if pat == o_sent:
                    matches.append({
                        'submitted_idx':      s_idx,
                        'original_idx':       o_idx,
                        'submitted_sentence': s_sent,
                        'hash_value':         ph,
                        'algorithm':          'RABIN-KARP',
                    })
                    break

    return matches


# ── Display ───────────────────────────────────────────────────

def print_exact_results(kmp_matches: list, rk_matches: list,
                         window_matches: list, total: int):
    """Print exact matching results."""
    sep = '═' * 62
    print(f"\n{sep}")
    print("  🔬  EXACT MATCHING RESULTS")
    print(sep)

    print(f"\n  ── KMP Matches ({len(kmp_matches)}) ──")
    for i, m in enumerate(kmp_matches[:5], 1):
        print(f"  {i}. [{m['submitted_idx']+1}] \"{m['submitted_sentence'][:70]}\"")
        print(f"     Positions: {m['positions'][:3]}  Len: {m['length']}ch")
        if i == 1:
            lps = _build_lps(m['submitted_sentence'][:15].lower())
            print(f"     LPS demo: pattern={list(m['submitted_sentence'][:6].lower())}")
            print(f"     LPS arr : {lps}")

    print(f"\n  ── Rabin-Karp Matches ({len(rk_matches)}) ──")
    for i, m in enumerate(rk_matches[:5], 1):
        print(f"  {i}. [{m['submitted_idx']+1}] \"{m['submitted_sentence'][:70]}\"")
        print(f"     Hash: {m['hash_value']}")

    print(f"\n  ── Window Matches (w=50, count={len(window_matches)}) ──")
    for i, (ts, rs) in enumerate(window_matches[:3], 1):
        print(f"  {i}. text[{ts}:{ts+50}] ↔ ref[{rs}:{rs+50}]")

    print(f"\n  KMP Score   : {round(100*len(kmp_matches)/max(1,total),1)}%")
    print(f"  RK Score    : {round(100*len(rk_matches)/max(1,total),1)}%")
    print(f"  Window Hits : {len(window_matches)}")
    print(f"{sep}\n")
