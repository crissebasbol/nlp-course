"""
Step 5: POS Tagging & Named Entity Recognition (spaCy es_core_news_md)
Tags each token with its part of speech and identifies named entities.
"""

import ast

import pandas as pd
import spacy

from config import MIN_TOKEN_LEN, OUTPUTS_DIR, SPACY_MODEL, SUBJECT_COL

_KEEP_POS = {"NOUN", "ADJ"}
_USEFUL_ENT_TYPES = {"MISC", "ORG", "LOC", "PER"}


def _parse_tokens(cell) -> list[str]:
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        try:
            return ast.literal_eval(cell)
        except Exception:
            return cell.split()
    return []


def tag_doc(text: str, nlp) -> tuple[list[tuple], list[tuple]]:
    """Return (pos_tags, ner_entities) for a text."""
    doc = nlp(text)
    pos_tags = [(t.text, t.pos_, t.lemma_) for t in doc if not t.is_space]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return pos_tags, entities


def extract_keywords_from_doc(text: str, nlp) -> list[str]:
    """Keep lemmas of NOUN/ADJ tokens that are long enough and not stopwords."""
    doc = nlp(text)
    return [
        t.lemma_.lower()
        for t in doc
        if t.pos_ in _KEEP_POS
        and not t.is_stop
        and len(t.lemma_) >= MIN_TOKEN_LEN
        and t.is_alpha
    ]


def process(df: pd.DataFrame, nlp) -> pd.DataFrame:
    pos_tags_col = []
    entities_col = []
    keywords_col = []

    for text in df["topic_spellchecked"]:
        if not isinstance(text, str) or not text.strip():
            pos_tags_col.append([])
            entities_col.append([])
            keywords_col.append([])
            continue
        pos_tags, entities = tag_doc(text, nlp)
        keywords = extract_keywords_from_doc(text, nlp)
        pos_tags_col.append(pos_tags)
        entities_col.append(entities)
        keywords_col.append(keywords)

    df = df.copy()
    df["pos_tags"] = pos_tags_col
    df["entities"] = entities_col
    df["keywords"] = keywords_col
    return df


def _parse_list_col(cell):
    """Parse a list cell that may have been serialised to string by CSV round-trip."""
    if isinstance(cell, list):
        return cell
    if isinstance(cell, str):
        try:
            return ast.literal_eval(cell)
        except Exception:
            return []
    return []


def print_pos_examples(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("STEP 5A: POS TAGGING (spaCy)")
    print("=" * 60)
    print(f"{'TOKEN':<25} {'POS':<8} {'LEMMA'}")
    print("-" * 50)
    for _, row in df.sample(5, random_state=7).iterrows():
        tags = _parse_list_col(row["pos_tags"])
        print(f"\n[{row[SUBJECT_COL]}] {row['topic_spellchecked']}")
        for item in tags[:8]:
            token, pos, lemma = item
            print(f"  {token:<25} {pos:<8} {lemma}")
    print()


def print_ner_examples(df: pd.DataFrame) -> None:
    from collections import Counter as _Counter
    print("=" * 60)
    print("STEP 5B - NER (spaCy es_core_news_md)")
    print("=" * 60)

    df = df.copy()
    df["entities"] = df["entities"].apply(_parse_list_col)
    has_ents = df[df["entities"].apply(len) > 0]
    print(f"Topics with at least one entity: {len(has_ents)}")
    print()

    all_ents: list[tuple] = []
    for ents in df["entities"]:
        all_ents.extend(ents)
    by_type: dict = {}
    for text, label in all_ents:
        by_type.setdefault(label, _Counter())[text] += 1

    for label, counter in sorted(by_type.items()):
        top = counter.most_common(5)
        print(f"  [{label}] " + ", ".join(f"{t!r}({n})" for t, n in top))
    print()

    print("Sample entities per topic:")
    for _, row in has_ents.head(6).iterrows():
        ents_str = ", ".join(f"{t}[{l}]" for t, l in row["entities"])
        print(f"  {row['topic_spellchecked']!r}")
        print(f"    → {ents_str}")
    print()


def save(df: pd.DataFrame) -> pd.DataFrame:
    out = OUTPUTS_DIR / "05_pos_ner.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")
    return df


def run(df: pd.DataFrame = None, nlp=None) -> pd.DataFrame:
    if df is None:
        df = pd.read_csv(OUTPUTS_DIR / "04_tokenised.csv")
    if nlp is None:
        nlp = spacy.load(SPACY_MODEL)
    df = process(df, nlp)
    print_pos_examples(df)
    print_ner_examples(df)
    return save(df)


if __name__ == "__main__":
    run()
