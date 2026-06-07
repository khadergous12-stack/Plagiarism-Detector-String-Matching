"""
preprocessor.py
===============
Text Ingestion & Normalization.
PDF Reference: Step 2 — Text Ingestion & Normalization

Converts raw text to normalized form while keeping
a character offset map so we can highlight original text later.

DSA Concepts:
  - String processing O(n)
  - Character-level index mapping (array)
  - Sliding window for sentence splitting
"""

import re
import os


def read_file(path: str) -> str:
    """Read a text file and return its content."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def normalize(text: str) -> tuple:
    """
    Normalize text and return (normalized_str, index_map).

    index_map[i] = position in original text
    Used to highlight matched spans in original document.

    PDF exact implementation from Step 2.
    """
    idx_map = []
    out     = []

    for i, ch in enumerate(text):
        if ch.isspace():
            ch = ' '
        if re.match(r'[A-Za-z0-9 ]', ch):
            out.append(ch.lower())
            idx_map.append(i)

    # Collapse multiple spaces
    s = re.sub(r'\s+', ' ', ''.join(out)).strip()
    return s, idx_map


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_sentences(text: str) -> list:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+|\n', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def tokenize(text: str) -> list:
    """Split cleaned text into word tokens."""
    return clean_text(text).split()


def print_preprocessing_report(original: str, submitted: str):
    """Print preprocessing summary."""
    o_clean = clean_text(original)
    s_clean = clean_text(submitted)
    o_sents = split_sentences(original)
    s_sents = split_sentences(submitted)

    print(f"\n{'═'*62}")
    print("  🧹  TEXT PREPROCESSING REPORT")
    print(f"{'═'*62}")
    print(f"  {'Document':<20} {'Tokens':>8} {'Sentences':>10} {'Chars':>8}")
    print(f"  {'─'*52}")
    print(f"  {'Original':<20} {len(tokenize(original)):>8} {len(o_sents):>10} {len(original):>8}")
    print(f"  {'Submitted':<20} {len(tokenize(submitted)):>8} {len(s_sents):>10} {len(submitted):>8}")
    print(f"\n  Original  (preview): {o_clean[:80]}...")
    print(f"  Submitted (preview): {s_clean[:80]}...")
    print(f"{'═'*62}\n")
