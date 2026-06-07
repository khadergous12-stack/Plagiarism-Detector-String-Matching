"""
lsh.py
======
MinHash + LSH (Locality Sensitive Hashing).
PDF Reference: Step 5 — MinHash & LSH (Corpus-Scale Candidate Generation)

Scales similarity search to millions of documents
without O(N²) pairwise comparisons.

DSA Concepts:
  - MinHash signatures (randomized hashing)
  - LSH banding technique (hash tables)
  - Set similarity via Jaccard estimation
  - Probability theory: P(collision) = Jaccard(A,B)

Algorithm:
  1. For each document, compute MinHash signature of n=100 hashes
  2. Split signature into bands of r rows each
  3. Documents sharing any band bucket are candidate pairs
  4. Only verify candidates (avoids O(N²) brute force)

Theoretical guarantee:
  With b=20 bands, r=5 rows:
  P(candidate) ≈ 1 - (1 - s^r)^b
  where s = true Jaccard similarity
"""

import random
import hashlib
import struct
from collections import defaultdict


# ── MinHasher ─────────────────────────────────────────────────

class MinHasher:
    """
    MinHash signature computer.
    PDF Step 5 exact implementation.

    Uses n independent hash functions simulated via
    seeded random linear combinations.
    """

    def __init__(self, n: int = 100, seed: int = 42):
        """
        Parameters:
            n    : number of hash functions (signature length)
            seed : random seed for reproducibility
        """
        self.n = n
        random.seed(seed)
        # Generate n (a, b) pairs for hash functions: h(x) = (a*x + b) % PRIME
        PRIME = (1 << 31) - 1
        self.a = [random.randint(1, PRIME) for _ in range(n)]
        self.b = [random.randint(0, PRIME) for _ in range(n)]
        self.PRIME = PRIME

    def _token_hash(self, token: str) -> int:
        """Hash a token string to integer."""
        return struct.unpack('I', hashlib.md5(token.encode()).digest()[:4])[0]

    def signature(self, tokens: set) -> list:
        """
        Compute MinHash signature for a set of tokens.

        For each hash function h_i:
          sig[i] = min over all tokens t of h_i(t)

        Time: O(n * |tokens|)
        Returns list of n integers.
        """
        if not tokens:
            return [self.PRIME] * self.n

        sig = []
        for i in range(self.n):
            min_val = self.PRIME
            for t in tokens:
                h = (self.a[i] * self._token_hash(t) + self.b[i]) % self.PRIME
                if h < min_val:
                    min_val = h
            sig.append(min_val)
        return sig

    def estimate_jaccard(self, sig_a: list, sig_b: list) -> float:
        """
        Estimate Jaccard similarity from two MinHash signatures.
        J(A,B) ≈ fraction of matching signature positions.
        """
        matches = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
        return matches / self.n


# ── LSH Index ─────────────────────────────────────────────────

class LSH:
    """
    LSH (Locality Sensitive Hashing) index.
    PDF Step 5 exact implementation.

    Bands technique:
      - Split n-length signature into b bands of r rows
      - Two docs share a bucket in band b if their
        signature rows b*r..(b+1)*r are identical
      - Any such collision → candidate pair
    """

    def __init__(self, bands: int = 20, rows: int = 5):
        """
        Parameters:
            bands : number of bands (b)
            rows  : rows per band (r)
            n = bands * rows = total signature length
        """
        self.bands  = bands
        self.rows   = rows
        self.tables = [defaultdict(list) for _ in range(bands)]

    def add(self, doc_id: str, sig: list):
        """
        Index a document's MinHash signature.
        Time: O(b)
        """
        for b in range(self.bands):
            band_key = tuple(sig[b * self.rows:(b + 1) * self.rows])
            self.tables[b][band_key].append(doc_id)

    def query(self, sig: list) -> list:
        """
        Find candidate similar documents.
        Returns list of doc_ids that share at least one band bucket.
        Time: O(b)
        """
        candidates = set()
        for b in range(self.bands):
            band_key = tuple(sig[b * self.rows:(b + 1) * self.rows])
            candidates.update(self.tables[b].get(band_key, []))
        return list(candidates)


# ── High-Level Functions ──────────────────────────────────────

def build_lsh_index(corpus: dict, n: int = 100,
                     bands: int = 20, rows: int = 5,
                     k_shingle: int = 3) -> tuple:
    """
    Build MinHash + LSH index over a corpus.

    Parameters:
        corpus : { doc_id: clean_text }

    Returns:
        (mh, lsh, signatures)
    """
    mh   = MinHasher(n=n)
    lsh  = LSH(bands=bands, rows=rows)
    sigs = {}

    for doc_id, text in corpus.items():
        # k-character shingles for token set
        tokens = set(text[i:i + k_shingle]
                     for i in range(len(text) - k_shingle + 1))
        sig = mh.signature(tokens)
        sigs[doc_id] = sig
        lsh.add(doc_id, sig)

    return mh, lsh, sigs


def lsh_similarity(text_a: str, text_b: str,
                    n: int = 100, k_shingle: int = 3) -> float:
    """
    Estimate Jaccard similarity using MinHash.
    Returns estimated similarity 0.0 → 1.0
    """
    mh = MinHasher(n=n)

    def shingles(t):
        return set(t[i:i + k_shingle] for i in range(len(t) - k_shingle + 1))

    sig_a = mh.signature(shingles(text_a))
    sig_b = mh.signature(shingles(text_b))
    return mh.estimate_jaccard(sig_a, sig_b)


def print_lsh_results(text_a: str, text_b: str):
    """Print MinHash + LSH analysis."""
    mh  = MinHasher(n=100)
    lsh = LSH(bands=20, rows=5)

    def shingles(t, k=3):
        return set(t[i:i + k] for i in range(len(t) - k + 1))

    sig_a = mh.signature(shingles(text_a))
    sig_b = mh.signature(shingles(text_b))
    est   = mh.estimate_jaccard(sig_a, sig_b)

    lsh.add('original', sig_a)
    candidates = lsh.query(sig_b)

    print(f"\n{'═'*62}")
    print("  🔐  MINHASH + LSH RESULTS")
    print(f"{'═'*62}")
    print(f"  Signature length (n)  : 100 hash functions")
    print(f"  Bands × Rows          : 20 × 5")
    print(f"  Shingle size (chars)  : 3")
    print(f"\n  Estimated Jaccard     : {est*100:.1f}%")
    print(f"  Original in candidates: {'Yes ✓' if 'original' in candidates else 'No'}")
    print(f"\n  Signature sample (first 8 of 100):")
    print(f"    Original : {sig_a[:8]}")
    print(f"    Submitted: {sig_b[:8]}")
    matching = sum(1 for x,y in zip(sig_a,sig_b) if x==y)
    print(f"\n  Matching positions    : {matching}/100")
    print(f"{'═'*62}\n")
