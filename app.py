"""
app.py
======
FastAPI Service — /index, /analyze, /whatif.
PDF Reference: Step 8 — FastAPI Service (Analyze, Index, Report)

Endpoints:
  POST /index    → index a document into corpus
  POST /analyze  → check submission against corpus
  GET  /docs_list → list indexed documents
  GET  /health   → health check
  GET  /         → React dashboard

Run:
  python main.py --api
  OR: uvicorn app:app --reload --port 8000
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi               import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses     import FileResponse
from fastapi.staticfiles   import StaticFiles
from pydantic              import BaseModel
from typing                import Optional

from src.preprocessor      import normalize, clean_text, split_sentences
from src.exact             import kmp_sentence_match, rk_sentence_match, rabin_karp_windows
from src.winnow            import winnow_overlap, fp_index
from src.lsh               import lsh_similarity, build_lsh_index
from src.similarity        import ngram_jaccard, tfidf_cosine, blended_score, verdict, ngram_set, jaccard


app = FastAPI(
    title       = "Plagiarism Detector v2 API",
    description = "KMP + Rabin-Karp + Winnowing + MinHash/LSH + TF-IDF",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── In-memory corpus ─────────────────────────────────────────
CORPUS      = {}   # doc_id → raw text
CORPUS_NORM = {}   # doc_id → normalized text


# ── Models ───────────────────────────────────────────────────
class IndexDoc(BaseModel):
    doc_id: str
    text:   str


class AnalyzeReq(BaseModel):
    text:  str
    top_k: int = 5


class WhatIfReq(BaseModel):
    text:     str
    extra_doc: Optional[dict] = None  # {"doc_id":"X","text":"..."}


# ── Startup: load default documents ──────────────────────────
def _load_defaults():
    for name, path in [
        ("original",  "documents/original.txt"),
        ("corpus",    "documents/corpus.txt"),
    ]:
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                raw  = f.read()
                norm, _ = normalize(raw)
                CORPUS[name]      = raw
                CORPUS_NORM[name] = norm


_load_defaults()


# ── /health ──────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "Plagiarism Detector v2",
            "indexed_docs": len(CORPUS)}


# ── / (dashboard) ─────────────────────────────────────────────
@app.get("/")
def root():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "dashboard", "index.html")
    return FileResponse(p)


# ── /index ───────────────────────────────────────────────────
@app.post("/index")
def index_doc(d: IndexDoc):
    """Index a reference document into the corpus."""
    norm, _ = normalize(d.text)
    CORPUS[d.doc_id]      = d.text
    CORPUS_NORM[d.doc_id] = norm
    return {"ok": True, "doc_id": d.doc_id,
            "indexed_docs": len(CORPUS)}


# ── /docs_list ───────────────────────────────────────────────
@app.get("/docs_list")
def docs_list():
    return {"docs": list(CORPUS.keys())}


# ── /analyze ─────────────────────────────────────────────────
@app.post("/analyze")
def analyze(req: AnalyzeReq):
    """
    Analyze submission against indexed corpus.
    PDF Step 8 exact endpoint.
    Returns: overall score, per-doc scores, matched sentences.
    """
    if not CORPUS:
        raise HTTPException(400, "No documents indexed yet. POST to /index first.")

    sub_norm, _ = normalize(req.text)
    sub_sents   = split_sentences(req.text)
    sub_clean   = clean_text(req.text)
    S           = ngram_set(sub_norm, n=4)

    results = []
    for doc_id, ref_norm in CORPUS_NORM.items():
        ref_sents = split_sentences(CORPUS[doc_id])
        ref_clean = clean_text(CORPUS[doc_id])

        # KMP
        kmp_m   = kmp_sentence_match(ref_sents, sub_sents)
        kmp_pct = len(kmp_m) / max(1, len(sub_sents))

        # Rabin-Karp
        rk_m    = rk_sentence_match(ref_sents, sub_sents)
        rk_pct  = len(rk_m)  / max(1, len(sub_sents))

        # Winnowing
        wov     = winnow_overlap(sub_norm, ref_norm)

        # MinHash / LSH
        lsh_sim = lsh_similarity(sub_clean, ref_clean)

        # TF-IDF + N-gram
        tfidf   = tfidf_cosine(sub_norm, ref_norm)
        ngram   = jaccard(S, ngram_set(ref_norm, n=4))

        # Blended score (PDF Step 7)
        score   = blended_score(kmp_pct, wov, tfidf, ngram)

        results.append({
            "doc_id":      doc_id,
            "score":       score,
            "kmp_pct":     round(kmp_pct*100, 1),
            "rk_pct":      round(rk_pct*100, 1),
            "winnow_pct":  round(wov*100, 1),
            "lsh_sim":     round(lsh_sim*100, 1),
            "tfidf":       round(tfidf*100, 1),
            "ngram":       round(ngram*100, 1),
            "kmp_matches": [m['submitted_sentence'] for m in kmp_m[:5]],
        })

    results.sort(key=lambda x: -x["score"])
    overall = int(sum(r["score"] for r in results[:req.top_k])
                  / max(1, min(req.top_k, len(results))))

    return {
        "overall":    overall,
        "verdict":    verdict(overall),
        "candidates": results[:req.top_k],
        "total_sentences": len(sub_sents),
    }


# ── /whatif ──────────────────────────────────────────────────
@app.post("/whatif")
def whatif(req: WhatIfReq):
    """
    What-if: add an extra reference doc and re-analyze.
    PDF Step 8 — /whatif endpoint.
    """
    extra_corpus = dict(CORPUS_NORM)
    if req.extra_doc:
        norm, _ = normalize(req.extra_doc["text"])
        extra_corpus[req.extra_doc["doc_id"]] = norm

    sub_norm, _ = normalize(req.text)
    sub_sents   = split_sentences(req.text)
    S           = ngram_set(sub_norm, n=4)

    results = []
    for doc_id, ref_norm in extra_corpus.items():
        kmp_m    = kmp_sentence_match(split_sentences(CORPUS.get(doc_id,"")), sub_sents)
        kmp_pct  = len(kmp_m) / max(1, len(sub_sents))
        wov      = winnow_overlap(sub_norm, ref_norm)
        tfidf    = tfidf_cosine(sub_norm, ref_norm)
        ngram    = jaccard(S, ngram_set(ref_norm, n=4))
        score    = blended_score(kmp_pct, wov, tfidf, ngram)
        results.append({"doc_id": doc_id, "score": score,
                         "kmp_pct": round(kmp_pct*100,1),
                         "winnow_pct": round(wov*100,1),
                         "tfidf": round(tfidf*100,1)})

    results.sort(key=lambda x: -x["score"])
    overall = int(sum(r["score"] for r in results) / max(1, len(results)))
    return {"overall": overall, "verdict": verdict(overall), "candidates": results}
