"""
similarity.py
=============
Paraphrase Hints — N-gram Jaccard & TF-IDF Cosine.
PDF Reference: Step 6 — Paraphrase Hints (n-gram Jaccard & TF-IDF Cosine)

Catches paraphrased content where exact matchers fail.

DSA Concepts:
  - Set operations for Jaccard O(n+m)
  - Sliding window for n-gram generation O(n)
  - Hash maps for term frequency O(n)
  - Vector dot product for cosine similarity O(V)
"""

import math
from collections import Counter


# ── N-gram Set ────────────────────────────────────────────────

def ngram_set(text: str, n: int = 4) -> set:
    """
    Generate set of n-gram strings.
    PDF Step 6 exact implementation.
    Sliding window of size n over tokens.
    """
    toks = text.split()
    return {' '.join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set, b: set) -> float:
    """
    Jaccard similarity = |A∩B| / |A∪B|
    PDF Step 6 exact implementation.
    """
    return len(a & b) / max(1, len(a | b))


def ngram_jaccard(text_a: str, text_b: str, n: int = 4) -> float:
    """N-gram Jaccard similarity between two texts."""
    return jaccard(ngram_set(text_a, n), ngram_set(text_b, n))


# ── TF-IDF Cosine (Manual — no sklearn needed) ────────────────

def _tf(tokens: list) -> dict:
    """Term Frequency: count(t) / total_tokens"""
    total  = max(1, len(tokens))
    counts = Counter(tokens)
    return {t: c / total for t, c in counts.items()}


def _idf(docs: list) -> dict:
    """IDF: log(N / df(t)) + 1  (smoothed)"""
    N   = len(docs)
    idf = {}
    for term in set(t for doc in docs for t in doc):
        df       = sum(1 for doc in docs if term in doc)
        idf[term] = math.log(N / max(1, df)) + 1.0
    return idf


def _tfidf_vec(tokens: list, idf: dict) -> dict:
    """TF-IDF vector for a document."""
    tf = _tf(tokens)
    return {t: tf[t] * idf.get(t, 1.0) for t in tf}


def _cosine(v1: dict, v2: dict) -> float:
    """Cosine similarity between two TF-IDF vectors."""
    common  = set(v1) & set(v2)
    dot     = sum(v1[t] * v2[t] for t in common)
    mag1    = math.sqrt(sum(x**2 for x in v1.values()))
    mag2    = math.sqrt(sum(x**2 for x in v2.values()))
    return dot / (mag1 * mag2) if mag1 and mag2 else 0.0


def tfidf_cosine(a: str, b: str) -> float:
    """
    TF-IDF cosine similarity between two documents.
    PDF Step 6 exact implementation.
    """
    ta = a.split()
    tb = b.split()
    idf = _idf([ta, tb])
    va  = _tfidf_vec(ta, idf)
    vb  = _tfidf_vec(tb, idf)
    return _cosine(va, vb)


# ── Blended Score ─────────────────────────────────────────────

def blended_score(exact_cov: float, winnow_overlap: float,
                   tfidf: float, jacc: float) -> int:
    """
    Blended 0-100 plagiarism score.
    PDF Step 7 exact implementation.

    Weights:
      Exact coverage  : 40%
      Winnow overlap  : 30%
      TF-IDF cosine   : 20%
      N-gram Jaccard  : 10%
    """
    w1, w2, w3, w4 = 0.4, 0.3, 0.2, 0.1
    return min(100, int(round(100 * (w1*exact_cov + w2*winnow_overlap + w3*tfidf + w4*jacc))))


def verdict(score: int) -> str:
    if score >= 75: return "🔴 HIGH PLAGIARISM"
    if score >= 40: return "🟡 MODERATE PLAGIARISM"
    if score >= 15: return "🟠 LOW PLAGIARISM"
    return "🟢 LIKELY ORIGINAL"


def print_similarity_report(scores: dict):
    """Print similarity breakdown."""
    print(f"\n{'═'*55}")
    print("  📊  SIMILARITY ANALYSIS — SCORE BREAKDOWN")
    print(f"{'═'*55}")
    print(f"  Exact / KMP Match    : {scores['exact_pct']:>6.1f}%")
    print(f"  Winnow Fingerprints  : {scores['winnow_pct']:>6.1f}%")
    print(f"  MinHash / LSH        : {scores['lsh_pct']:>6.1f}%")
    print(f"  TF-IDF Cosine        : {scores['tfidf_pct']:>6.1f}%")
    print(f"  N-gram Jaccard (4)   : {scores['ngram_pct']:>6.1f}%")
    print(f"{'─'*55}")
    print(f"  ⚡ FINAL SCORE        : {scores['final_score']:>3}%")
    print(f"  {scores['verdict']}")
    print(f"{'═'*55}\n")
