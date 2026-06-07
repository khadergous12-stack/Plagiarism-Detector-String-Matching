"""
tests/test_detector.py
======================
Unit Tests — Plagiarism Detector v2.
PDF Reference: Step 8 — Unit Tests (Correctness)

10 Tests:
  1. test_kmp_finds_exact_match
  2. test_kmp_lps_array
  3. test_rk_finds_exact_match
  4. test_rabin_karp_window_hits
  5. test_winnow_overlap_high
  6. test_winnow_overlap_low
  7. test_minhash_signature_length
  8. test_lsh_finds_candidate
  9. test_tfidf_cosine_identical
 10. test_blended_score_range
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessor import clean_text, split_sentences, normalize
from src.exact        import kmp_search, _build_lps, rk_sentence_match, rabin_karp_windows
from src.winnow       import winnow_overlap, winnow_fingerprints
from src.lsh          import MinHasher, LSH, lsh_similarity
from src.similarity   import tfidf_cosine, ngram_jaccard, blended_score, jaccard, ngram_set


# ─────────────────────────────────────────────────────────────

def test_kmp_finds_exact_match():
    text    = "machine learning is a powerful technique used in artificial intelligence"
    pattern = "powerful technique used"
    result  = kmp_search(text, pattern)
    assert len(result) > 0, "KMP should find the pattern"
    assert text[result[0]:result[0]+len(pattern)] == pattern
    print("  [PASS] test_kmp_finds_exact_match")


def test_kmp_lps_array():
    pattern = "ababcabab"
    lps     = _build_lps(pattern)
    assert lps == [0,0,1,2,0,1,2,3,4], f"LPS wrong: {lps}"
    assert lps[0] == 0,  "LPS[0] always 0"
    assert len(lps) == len(pattern), "LPS length must equal pattern length"
    print("  [PASS] test_kmp_lps_array")


def test_rk_finds_exact_match():
    orig  = ["machine learning is powerful", "deep learning uses neural networks"]
    subm  = ["deep learning uses neural networks", "a different sentence here"]
    m     = rk_sentence_match(orig, subm)
    assert len(m) >= 1, "RK should find at least one match"
    assert m[0]['submitted_idx'] == 0
    print("  [PASS] test_rk_finds_exact_match")


def test_rabin_karp_window_hits():
    text = "the quick brown fox jumps over the lazy dog and the fox"
    ref  = "the quick brown fox jumps over the lazy"
    hits = rabin_karp_windows(text, ref, w=20)
    assert len(hits) > 0, "Should find overlapping windows"
    print(f"  [PASS] test_rabin_karp_window_hits ({len(hits)} windows)")


def test_winnow_overlap_high():
    text = "machine learning is a subset of artificial intelligence that enables systems to learn"
    dup  = "machine learning is a subset of artificial intelligence that enables systems to learn"
    ov   = winnow_overlap(text, dup)
    assert ov > 0.9, f"Identical texts should have overlap > 0.9, got {ov}"
    print(f"  [PASS] test_winnow_overlap_high (overlap={ov:.2f})")


def test_winnow_overlap_low():
    text  = "machine learning uses neural networks for classification tasks"
    other = "the french revolution began in seventeen eighty nine with the storming"
    ov    = winnow_overlap(text, other)
    assert ov < 0.3, f"Unrelated texts should have low overlap, got {ov}"
    print(f"  [PASS] test_winnow_overlap_low (overlap={ov:.2f})")


def test_minhash_signature_length():
    mh  = MinHasher(n=100)
    sig = mh.signature({"machine","learning","deep","neural"})
    assert len(sig) == 100, f"Signature should have 100 values, got {len(sig)}"
    assert all(isinstance(x, int) for x in sig)
    print("  [PASS] test_minhash_signature_length")


def test_lsh_finds_candidate():
    mh  = MinHasher(n=100)
    lsh = LSH(bands=20, rows=5)
    def sh(t):
        return set(t[i:i+3] for i in range(len(t)-2))
    sig_orig = mh.signature(sh("machine learning deep learning neural"))
    sig_sub  = mh.signature(sh("machine learning deep learning neural"))
    lsh.add("original", sig_orig)
    cands = lsh.query(sig_sub)
    assert "original" in cands, "Identical text should always be a candidate"
    print("  [PASS] test_lsh_finds_candidate")


def test_tfidf_cosine_identical():
    text = "machine learning is a powerful technique in artificial intelligence"
    sim  = tfidf_cosine(text, text)
    assert abs(sim - 1.0) < 1e-6, f"Identical texts cosine should be 1.0, got {sim}"
    sim2 = tfidf_cosine(text, "completely unrelated topic about cooking recipes")
    assert sim2 < 0.5, f"Unrelated texts should have low cosine, got {sim2}"
    print(f"  [PASS] test_tfidf_cosine_identical (identical=1.0, unrelated={sim2:.2f})")


def test_blended_score_range():
    # All zeros → score 0
    s0 = blended_score(0.0, 0.0, 0.0, 0.0)
    assert s0 == 0, f"All zeros should give 0, got {s0}"
    # All ones → score 100
    s1 = blended_score(1.0, 1.0, 1.0, 1.0)
    assert s1 == 100, f"All ones should give 100, got {s1}"
    # Partial
    sp = blended_score(0.5, 0.5, 0.5, 0.5)
    assert 0 <= sp <= 100, f"Score out of range: {sp}"
    print(f"  [PASS] test_blended_score_range (0→{s0}, 0.5→{sp}, 1.0→{s1})")


# ─────────────────────────────────────────────────────────────

def run_all_tests():
    tests = [
        test_kmp_finds_exact_match,
        test_kmp_lps_array,
        test_rk_finds_exact_match,
        test_rabin_karp_window_hits,
        test_winnow_overlap_high,
        test_winnow_overlap_low,
        test_minhash_signature_length,
        test_lsh_finds_candidate,
        test_tfidf_cosine_identical,
        test_blended_score_range,
    ]

    print(f"\n{'═'*55}")
    print("  🧪  RUNNING UNIT TESTS — PLAGIARISM DETECTOR v2")
    print(f"{'═'*55}")

    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}"); failed += 1

    print(f"{'═'*55}")
    print(f"  Results: {passed} passed | {failed} failed | {len(tests)} total")
    print("  ✅  All tests passed!" if failed == 0 else "  ❌  Some tests failed.")
    print(f"{'═'*55}\n")
    return failed == 0


if __name__ == "__main__":
    run_all_tests()
