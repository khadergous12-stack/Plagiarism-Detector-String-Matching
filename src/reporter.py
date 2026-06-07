"""
reporter.py
===========
Report generation — TXT, CSV, highlighted output.
PDF Reference: Step 7 — Scoring & Evidence Extraction
"""

import os
import csv
from datetime import datetime

OUTPUT_DIR = "outputs"
REPORT_DIR = "reports"


def _ensure():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)


def print_highlighted(submitted_sents: list, kmp_matches: list):
    """Print submitted doc with [COPIED] markers."""
    matched = {m['submitted_idx'] for m in kmp_matches}
    print(f"\n{'═'*65}")
    print("  🎯  SUBMITTED DOCUMENT — HIGHLIGHTED MATCHES")
    print(f"  [COPIED] = detected plagiarised by KMP")
    print(f"{'═'*65}")
    for i, s in enumerate(submitted_sents):
        if i in matched:
            print(f"  [{i+1:>2}] ⚠️  [COPIED] {s}")
        else:
            print(f"  [{i+1:>2}]           {s[:80]}")
    copied   = len(matched)
    original = len(submitted_sents) - copied
    print(f"\n  Copied    : {copied}/{len(submitted_sents)} sentences")
    print(f"  Original  : {original}/{len(submitted_sents)} sentences")
    print(f"{'═'*65}\n")


def save_txt(orig_path, sub_path, kmp_m, rk_m, win_m, scores,
             filename="plagiarism_report.txt"):
    _ensure()
    path = os.path.join(OUTPUT_DIR, filename)
    now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("="*65 + "\n")
        f.write("  PLAGIARISM DETECTOR v2 — FULL REPORT\n")
        f.write(f"  Generated : {now}\n")
        f.write("="*65 + "\n\n")
        f.write(f"  Original  : {orig_path}\n")
        f.write(f"  Submitted : {sub_path}\n\n")
        f.write("SCORE BREAKDOWN:\n")
        f.write(f"  Exact/KMP    : {scores['exact_pct']}%\n")
        f.write(f"  Winnowing    : {scores['winnow_pct']}%\n")
        f.write(f"  MinHash/LSH  : {scores['lsh_pct']}%\n")
        f.write(f"  TF-IDF       : {scores['tfidf_pct']}%\n")
        f.write(f"  N-gram       : {scores['ngram_pct']}%\n")
        f.write(f"  FINAL SCORE  : {scores['final_score']}%\n")
        f.write(f"  VERDICT      : {scores['verdict']}\n\n")
        f.write("KMP MATCHES:\n")
        for i, m in enumerate(kmp_m, 1):
            f.write(f"  {i}. [{m['submitted_idx']+1}] {m['submitted_sentence']}\n")
        f.write("\nRABIN-KARP MATCHES:\n")
        for i, m in enumerate(rk_m, 1):
            f.write(f"  {i}. [{m['submitted_idx']+1}] {m['submitted_sentence']}\n")
        f.write(f"\nWINDOW MATCHES: {len(win_m)} overlapping windows (size=50)\n")
    print(f"  [✓] TXT report → {path}")
    return path


def save_csv(kmp_m, rk_m, scores, filename="matched_content.csv"):
    _ensure()
    path = os.path.join(REPORT_DIR, filename)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['#','Algorithm','Submitted Line','Sentence','Detail'])
        row = 1
        for m in kmp_m:
            w.writerow([row,'KMP',m['submitted_idx']+1,
                        m['submitted_sentence'],f"pos={m['positions'][:3]}"])
            row += 1
        for m in rk_m:
            w.writerow([row,'Rabin-Karp',m['submitted_idx']+1,
                        m['submitted_sentence'],f"hash={m['hash_value']}"])
            row += 1
        w.writerow([])
        w.writerow(['FINAL SCORE', f"{scores['final_score']}%", scores['verdict']])
    print(f"  [✓] CSV report → {path}")
    return path


def print_final_report(orig_path, sub_path, kmp_m, rk_m, win_m, scores):
    """Print complete terminal report."""
    sep = '═'*62
    print(f"\n{sep}")
    print("  📋  PLAGIARISM DETECTION — FINAL REPORT")
    print(sep)
    print(f"  Original  : {orig_path}")
    print(f"  Submitted : {sub_path}")
    print(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(sep)
    print(f"  KMP Matches     : {len(kmp_m)}")
    print(f"  RK Matches      : {len(rk_m)}")
    print(f"  Window Matches  : {len(win_m)}")
    print(f"\n  Exact/KMP       : {scores['exact_pct']}%")
    print(f"  Winnow          : {scores['winnow_pct']}%")
    print(f"  MinHash/LSH     : {scores['lsh_pct']}%")
    print(f"  TF-IDF Cosine   : {scores['tfidf_pct']}%")
    print(f"  N-gram Jaccard  : {scores['ngram_pct']}%")
    print(f"  {'─'*38}")
    print(f"  ⚡ FINAL SCORE  : {scores['final_score']}%")
    print(f"  {scores['verdict']}")
    print(f"{sep}\n")
