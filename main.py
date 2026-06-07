"""
main.py
=======
Plagiarism Detector v2 — Full CLI Entry Point
=============================================
All PDF steps in one place.

Usage:
    python main.py              # Interactive menu
    python main.py --auto       # Full auto demo
    python main.py --test       # Unit tests
    python main.py --api        # FastAPI server
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocessor import (read_file, clean_text, split_sentences,
                               normalize, print_preprocessing_report)
from src.exact        import (kmp_sentence_match, rk_sentence_match,
                               rabin_karp_windows, print_exact_results)
from src.winnow       import winnow_overlap, print_winnow_results
from src.lsh          import lsh_similarity, print_lsh_results
from src.similarity   import (ngram_jaccard, tfidf_cosine, blended_score,
                               verdict, ngram_set, jaccard,
                               print_similarity_report)
from src.reporter     import (print_highlighted, save_txt, save_csv,
                               print_final_report)


BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║   PLAGIARISM DETECTOR  v2.0                                  ║
║   DSA Course Project — Full Implementation                   ║
║   KMP | Rabin-Karp | Winnowing | MinHash/LSH | TF-IDF        ║
╚══════════════════════════════════════════════════════════════╝
"""

MENU = """
  ┌────────────────────────────────────────────────────┐
  │  MAIN MENU                                         │
  ├────────────────────────────────────────────────────┤
  │  1. Load & Preprocess Documents                    │
  │  2. Run Naive + KMP Matching                       │
  │  3. Run Rabin-Karp Matching                        │
  │  4. Run Winnowing Fingerprints                     │
  │  5. Run MinHash + LSH Analysis                     │
  │  6. Run Similarity Analysis (TF-IDF + N-gram)      │
  │  7. Show Highlighted Matches                       │
  │  8. Save Full Reports (TXT + CSV)                  │
  │  9. Run Unit Tests                                 │
  │  A. Run Full Auto Demo (all steps)                 │
  │  B. Start FastAPI Server + Dashboard               │
  │  0. Exit                                           │
  └────────────────────────────────────────────────────┘
"""


def run_pipeline(orig_path: str, sub_path: str, verbose: bool = True):
    """Full detection pipeline — all PDF steps."""

    # ── Step 1: Load & Preprocess ─────────────────────────────
    if verbose: print("\n  [1/8] Loading & preprocessing...")
    original_raw  = read_file(orig_path)
    submitted_raw = read_file(sub_path)
    orig_clean    = clean_text(original_raw)
    sub_clean     = clean_text(submitted_raw)
    orig_norm, _  = normalize(original_raw)
    sub_norm,  _  = normalize(submitted_raw)
    orig_sents    = split_sentences(original_raw)
    sub_sents     = split_sentences(submitted_raw)
    if verbose:
        print_preprocessing_report(original_raw, submitted_raw)

    # ── Step 2: KMP Matching ──────────────────────────────────
    if verbose: print("  [2/8] Running KMP...")
    kmp_matches = kmp_sentence_match(orig_sents, sub_sents)

    # ── Step 3: Rabin-Karp ────────────────────────────────────
    if verbose: print("  [3/8] Running Rabin-Karp...")
    rk_matches     = rk_sentence_match(orig_sents, sub_sents)
    window_matches = rabin_karp_windows(sub_norm, orig_norm, w=50)
    if verbose:
        print_exact_results(kmp_matches, rk_matches, window_matches, len(sub_sents))

    # ── Step 4: Winnowing ─────────────────────────────────────
    if verbose: print("  [4/8] Running Winnowing...")
    wov = winnow_overlap(sub_norm, orig_norm)
    if verbose:
        print_winnow_results(orig_norm, sub_norm)

    # ── Step 5: MinHash / LSH ─────────────────────────────────
    if verbose: print("  [5/8] Running MinHash + LSH...")
    lsh_sim = lsh_similarity(sub_clean, orig_clean)
    if verbose:
        print_lsh_results(orig_norm, sub_norm)

    # ── Step 6: Similarity Scores ─────────────────────────────
    if verbose: print("  [6/8] Computing similarity scores...")
    tfidf   = tfidf_cosine(sub_norm, orig_norm)
    ngram   = jaccard(ngram_set(sub_norm, 4), ngram_set(orig_norm, 4))
    kmp_pct = len(kmp_matches) / max(1, len(sub_sents))
    score   = blended_score(kmp_pct, wov, tfidf, ngram)

    scores = {
        'exact_pct':   round(kmp_pct * 100, 1),
        'winnow_pct':  round(wov * 100, 1),
        'lsh_pct':     round(lsh_sim * 100, 1),
        'tfidf_pct':   round(tfidf * 100, 1),
        'ngram_pct':   round(ngram * 100, 1),
        'final_score': score,
        'verdict':     verdict(score),
    }
    if verbose:
        print_similarity_report(scores)

    # ── Step 7: Highlighted Matches ───────────────────────────
    if verbose:
        print("  [7/8] Highlighted matches...")
        print_highlighted(sub_sents, kmp_matches)

    # ── Step 8: Reports ───────────────────────────────────────
    if verbose:
        print("  [8/8] Final report...")
        print_final_report(orig_path, sub_path, kmp_matches,
                           rk_matches, window_matches, scores)

    return {
        'orig_sents':     orig_sents,
        'sub_sents':      sub_sents,
        'kmp_matches':    kmp_matches,
        'rk_matches':     rk_matches,
        'window_matches': window_matches,
        'scores':         scores,
    }


def interactive_menu(orig_path: str, sub_path: str):
    result = None

    while True:
        print(MENU)
        choice = input("  Enter choice: ").strip().upper()

        if choice == "1":
            orig = read_file(orig_path)
            sub  = read_file(sub_path)
            print_preprocessing_report(orig, sub)

        elif choice == "2":
            orig = read_file(orig_path); sub = read_file(sub_path)
            os = split_sentences(orig);  ss = split_sentences(sub)
            on, _ = normalize(orig);     sn, _ = normalize(sub)
            km = kmp_sentence_match(os, ss)
            rm = rk_sentence_match(os, ss)
            wm = rabin_karp_windows(sn, on, 50)
            print_exact_results(km, rm, wm, len(ss))

        elif choice == "3":
            orig = read_file(orig_path); sub = read_file(sub_path)
            os = split_sentences(orig);  ss = split_sentences(sub)
            on, _ = normalize(orig);     sn, _ = normalize(sub)
            rm = rk_sentence_match(os, ss)
            wm = rabin_karp_windows(sn, on, 50)
            print_exact_results([], rm, wm, len(ss))

        elif choice == "4":
            orig = read_file(orig_path); sub = read_file(sub_path)
            on, _ = normalize(orig);     sn, _ = normalize(sub)
            print_winnow_results(on, sn)

        elif choice == "5":
            orig = read_file(orig_path); sub = read_file(sub_path)
            on, _ = normalize(orig);     sn, _ = normalize(sub)
            print_lsh_results(on, sn)

        elif choice == "6":
            orig = read_file(orig_path); sub = read_file(sub_path)
            on, _ = normalize(orig);     sn, _ = normalize(sub)
            t = tfidf_cosine(sn, on)
            n = jaccard(ngram_set(sn, 4), ngram_set(on, 4))
            print(f"\n  TF-IDF Cosine   : {t*100:.1f}%")
            print(f"  N-gram Jaccard  : {n*100:.1f}%\n")

        elif choice == "7":
            if result is None:
                result = run_pipeline(orig_path, sub_path, verbose=False)
            print_highlighted(result['sub_sents'], result['kmp_matches'])

        elif choice == "8":
            if result is None:
                result = run_pipeline(orig_path, sub_path, verbose=False)
            save_txt(orig_path, sub_path, result['kmp_matches'],
                     result['rk_matches'], result['window_matches'],
                     result['scores'])
            save_csv(result['kmp_matches'], result['rk_matches'],
                     result['scores'])
            print("\n  ✓  Reports saved to outputs/ and reports/\n")

        elif choice == "9":
            from tests.test_detector import run_all_tests
            run_all_tests()

        elif choice == "A":
            result = run_pipeline(orig_path, sub_path)
            save_txt(orig_path, sub_path, result['kmp_matches'],
                     result['rk_matches'], result['window_matches'],
                     result['scores'])
            save_csv(result['kmp_matches'], result['rk_matches'],
                     result['scores'])
            print("\n  ✓  Full demo complete. Reports saved.\n")

        elif choice == "B":
            import uvicorn
            print("\n  Starting API server → http://localhost:8000")
            print("  Open dashboard/index.html or http://localhost:8000\n")
            uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

        elif choice == "0":
            print("\n  Goodbye! 👋\n"); break
        else:
            print("  Invalid choice.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto",      action="store_true")
    parser.add_argument("--test",      action="store_true")
    parser.add_argument("--api",       action="store_true")
    parser.add_argument("--original",  default="documents/original.txt")
    parser.add_argument("--submitted", default="documents/submitted.txt")
    args = parser.parse_args()

    print(BANNER)

    try:
        _ = read_file(args.original)
        _ = read_file(args.submitted)
    except FileNotFoundError as e:
        print(f"\n  ❌ {e}\n  Run from project root.\n"); sys.exit(1)

    print(f"  Original  : {args.original}")
    print(f"  Submitted : {args.submitted}\n")

    if args.test:
        from tests.test_detector import run_all_tests
        run_all_tests()
    elif args.api:
        import uvicorn
        print("  Starting server → http://localhost:8000")
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
    elif args.auto:
        result = run_pipeline(args.original, args.submitted)
        save_txt(args.original, args.submitted,
                 result['kmp_matches'], result['rk_matches'],
                 result['window_matches'], result['scores'])
        save_csv(result['kmp_matches'], result['rk_matches'],
                 result['scores'])
        from tests.test_detector import run_all_tests
        run_all_tests()
        print("\n  ✓  All outputs saved.\n")
    else:
        interactive_menu(args.original, args.submitted)


if __name__ == "__main__":
    main()
