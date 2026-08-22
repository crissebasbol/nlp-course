"""
Step 3: Spell Correction with SymSpell
Builds a domain-specific frequency dictionary from the cleaned corpus
(better than a generic dictionary for medical Spanish)
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from symspellpy import SymSpell, Verbosity

from config import DICT_DIR, OUTPUTS_DIR

DICT_PATH = DICT_DIR / "corpus_freq_dict.txt"
MAX_EDIT_DISTANCE = 2
MIN_TOKEN_FREQ = 2   # tokens appearing only once are likely correct proper nouns - skip them


_WORD_RE = re.compile(r"[a-záéíóúüñA-ZÁÉÍÓÚÜÑ]+")


def _tokenise_simple(text: str) -> list[str]:
    return re.findall(r"[a-záéíóúüñ]+", text.lower())


def build_corpus_dictionary(df: pd.DataFrame) -> Path:
    """Build a frequency dictionary from the cleaned corpus itself.

    Only words with frequency >= MIN_TOKEN_FREQ are added. This prevents
    typos (which appear exactly once) from entering the dictionary and being
    treated as valid words by SymSpell.
    """
    counter: Counter = Counter()
    for topic in df["topic_clean"].dropna():
        counter.update(_tokenise_simple(topic))
    trusted = {w: f for w, f in counter.items() if f >= MIN_TOKEN_FREQ}
    with open(DICT_PATH, "w", encoding="utf-8") as f:
        for word, freq in sorted(trusted.items(), key=lambda x: -x[1]):
            f.write(f"{word} {freq}\n")
    print(f"Domain dictionary built: {len(trusted)} trusted tokens "
          f"(of {len(counter)} total, {len(counter)-len(trusted)} hapax excluded) → {DICT_PATH}")
    return DICT_PATH


def build_spellchecker(dict_path: Path = DICT_PATH) -> SymSpell:
    sym = SymSpell(max_dictionary_edit_distance=MAX_EDIT_DISTANCE)
    sym.load_dictionary(str(dict_path), term_index=0, count_index=1)
    return sym


def correct_topic(text: str, sym: SymSpell, freq: Counter) -> tuple[str, list[tuple]]:
    corrections: list[tuple] = []

    def _replace(match: re.Match) -> str:
        original = match.group()
        lower = original.lower()
        if freq[lower] >= MIN_TOKEN_FREQ:
            return original
        suggestions = sym.lookup(lower, Verbosity.CLOSEST, max_edit_distance=1)
        if suggestions and suggestions[0].term != lower:
            best = suggestions[0].term
            corrections.append((lower, best))
            return best
        return original

    corrected = _WORD_RE.sub(_replace, text)
    return corrected, corrections


def apply_spellcheck(df: pd.DataFrame) -> pd.DataFrame:
    freq: Counter = Counter()
    for topic in df["topic_clean"].dropna():
        freq.update(_tokenise_simple(topic))

    sym = build_spellchecker()
    corrected_topics = []
    all_corrections = []

    for topic in df["topic_clean"]:
        if not isinstance(topic, str):
            corrected_topics.append(topic)
            all_corrections.append([])
            continue
        corrected, corrections = correct_topic(topic, sym, freq)
        corrected_topics.append(corrected)
        all_corrections.append(corrections)

    df = df.copy()
    df["topic_spellchecked"] = corrected_topics
    df["_corrections"] = all_corrections
    return df


def print_examples(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("STEP 3: SPELL CHECK (SymSpell, domain dictionary)")
    print("=" * 60)
    changed = df[df["_corrections"].apply(len) > 0]
    print(f"Topics with at least one correction: {len(changed)}")
    print()
    print("Sample corrections (original token → suggested):")
    shown = 0
    for _, row in changed.iterrows():
        for orig, fix in row["_corrections"]:
            print(f"  {orig!r:25} → {fix!r}")
            shown += 1
            if shown >= 15:
                break
        if shown >= 15:
            break
    print()


def save(df: pd.DataFrame) -> pd.DataFrame:
    out_df = df.drop(columns=["_corrections"])
    path = OUTPUTS_DIR / "03_spellchecked.csv"
    out_df.to_csv(path, index=False)
    print(f"Saved → {path}")
    return out_df


def run(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(OUTPUTS_DIR / "02_cleaned.csv")
    build_corpus_dictionary(df)
    df = apply_spellcheck(df)
    print_examples(df)
    return save(df)


if __name__ == "__main__":
    run()
