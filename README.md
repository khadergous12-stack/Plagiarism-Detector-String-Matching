# 🔍 Plagiarism-Detector-String-Matching

> A **complete, industry-oriented DSA project** implementing all PDF steps — KMP, Rabin-Karp, Winnowing Fingerprints, MinHash+LSH, TF-IDF Cosine, FastAPI REST API, React Dashboard, and 10 Unit Tests.

---

## 📌 Overview

Detects plagiarism across three layers — **exact matches**, **near-duplicate fingerprints**, and **paraphrase hints** — producing a 0–100 blended score with highlighted evidence.

---

## 🧠 DSA Concepts

| Concept | Module | Complexity |
|---|---|---|
| **KMP Algorithm + LPS Array** | `src/exact.py` | O(n+m) |
| **Rabin-Karp Rolling Hash** | `src/exact.py` | O(n+m) avg |
| **Sliding Window (k-shingles)** | `src/winnow.py` | O(n) |
| **Winnowing Fingerprints** | `src/winnow.py` | O(n) |
| **MinHash Signatures** | `src/lsh.py` | O(n·k) |
| **LSH Banding** | `src/lsh.py` | O(b) |
| **Set Operations (Jaccard)** | `src/similarity.py` | O(n+m) |
| **TF-IDF + Cosine Similarity** | `src/similarity.py` | O(V) |
| **Hash Maps** | Throughout | O(1) avg |

---

## ✨ Features — All PDF Steps

| PDF Step | Feature | Status |
|---|---|---|
| Step 2 | Text normalization + index map | ✅ |
| Step 3 | KMP + Rabin-Karp exact matching | ✅ |
| Step 4 | Winnowing fingerprints | ✅ |
| Step 5 | MinHash + LSH (20×5 bands) | ✅ |
| Step 6 | N-gram Jaccard + TF-IDF Cosine | ✅ |
| Step 7 | Blended 0-100 score + evidence | ✅ |
| Step 8 | FastAPI /index /analyze /whatif | ✅ |
| Step 9 | React Dashboard (PlagiaGuard) | ✅ |
| Step 10 | 10 Unit Tests — all passing | ✅ |

---

## 📁 Folder Structure

```
Plagiarism-Detector-v2/
│
├── documents/
│   ├── original.txt        # Reference document
│   ├── submitted.txt       # Document to check
│   └── corpus.txt          # Additional corpus
│
├── src/
│   ├── preprocessor.py     # Normalize, tokenize, split sentences
│   ├── exact.py            # KMP + Rabin-Karp
│   ├── winnow.py           # k-shingling + winnowing
│   ├── lsh.py              # MinHash + LSH
│   ├── similarity.py       # N-gram Jaccard + TF-IDF Cosine
│   └── reporter.py         # Report generation
│
├── tests/
│   └── test_detector.py    # 10 unit tests (all passing)
│
├── dashboard/
│   └── index.html          # React dashboard (PlagiaGuard)
│
├── outputs/                # Generated reports
├── reports/                # CSV matched content
├── docs/                   # Interview prep, architecture
│
├── app.py                  # FastAPI server
├── main.py                 # CLI entry point
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/Plagiarism-Detector-v2.git
cd Plagiarism-Detector-v2
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
python main.py              # Interactive menu
python main.py --auto       # Full demo (all 8 steps)
python main.py --test       # Run 10 unit tests
python main.py --api        # Start FastAPI + dashboard
```

**Dashboard:** After `--api`, open `http://localhost:8000`

---

## 📊 Sample Results

```
SCORE BREAKDOWN:
  Exact/KMP Match    :  60.0%
  Winnow Fingerprints:  56.7%
  MinHash/LSH        :  74.0%
  TF-IDF Cosine      :  87.4%
  N-gram Jaccard (4) :  37.8%
  ─────────────────────────────
  ⚡ FINAL SCORE     :  62%
  🟡 MODERATE PLAGIARISM

  Copied   : 9/15 sentences
  Original : 6/15 sentences
```

### Unit Tests
```
  [PASS] test_kmp_finds_exact_match
  [PASS] test_kmp_lps_array
  [PASS] test_rk_finds_exact_match
  [PASS] test_rabin_karp_window_hits
  [PASS] test_winnow_overlap_high
  [PASS] test_winnow_overlap_low
  [PASS] test_minhash_signature_length
  [PASS] test_lsh_finds_candidate
  [PASS] test_tfidf_cosine_identical
  [PASS] test_blended_score_range
  Results: 10 passed | 0 failed ✅
```

---

## 🌐 API Endpoints

```
POST /index    → index a reference document
POST /analyze  → check submission against corpus
POST /whatif   → add extra doc and re-analyze
GET  /docs_list → list indexed documents
GET  /health   → API health check
GET  /         → React dashboard
```

---

## 📅 Day-wise Commit Plan

| Day | Work | Commit |
|---|---|---|
| Day 1 | Setup, documents, preprocessor | `feat: text preprocessing and normalization` |
| Day 2 | KMP + Rabin-Karp exact matching | `feat: KMP O(n+m) and Rabin-Karp rolling hash` |
| Day 3 | Winnowing fingerprints | `feat: k-shingling and winnowing fingerprints` |
| Day 4 | MinHash + LSH | `feat: MinHash signatures and LSH banding` |
| Day 5 | TF-IDF + blended score + tests | `feat: TF-IDF cosine, blended score, 10 unit tests` |
| Day 6 | FastAPI + dashboard + README | `docs: FastAPI API, React dashboard, full README` |

---

## 👨‍💻 Author
**Gous** — Computer Science Student, DSA Course Project
