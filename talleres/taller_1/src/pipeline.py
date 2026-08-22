"""
pipeline.py - Run the full NLP pipeline end-to-end.

Usage:
    python pipeline.py

Steps:
    1. Load & explore raw data
    2. Clean with regex
    3. Spell-correct with SymSpell (domain dictionary)
    4. Tokenise (regex, spaCy, NLTK)
    5. POS tagging + NER (spaCy)
    6. Keyword extraction + wordclouds
    7. Graph construction (bipartite + subject-to-subject)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import step01_load_explore as s1
import step02_clean as s2
import step03_spellcheck as s3
import step04_tokenize as s4
import step05_pos_ner as s5
import step06_keywords as s6
import step07_graph as s7
import spacy
from config import SPACY_MODEL


def run_pipeline() -> None:
    total_start = time.time()

    print("\n>>> Loading spaCy model once for all steps...")
    nlp = spacy.load(SPACY_MODEL)
    print(f"    Model loaded: {SPACY_MODEL}\n")

    steps = [
        ("1 - Load & Explore", lambda _: s1.run()),
        ("2 - Clean (regex)",  lambda df: s2.run(df)),
        ("3 - Spell Check",    lambda df: s3.run(df)),
        ("4 - Tokenise",       lambda df: s4.run(df, nlp)),
        ("5 - POS + NER",      lambda df: s5.run(df, nlp)),
        ("6 - Keywords",       lambda df: s6.run(df)),
        ("7 - Graphs",         lambda kw: s7.run(kw)),
    ]

    result = None
    for name, fn in steps:
        print(f"\n{'─'*60}")
        print(f"  Running step: {name}")
        print(f"{'─'*60}")
        t0 = time.time()
        result = fn(result)
        print(f"  Done in {time.time() - t0:.1f}s")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {time.time() - total_start:.1f}s")
    print(f"  Outputs saved to: outputs/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_pipeline()
