from conftest import RAW_WITH_CONTINUATION

from medec.data import check_no_leakage, number_sentences, parse_sentences, split_by_case, split_free_text


def test_parse_sentences_joins_continuation_lines():
    sentences = parse_sentences(RAW_WITH_CONTINUATION)
    assert len(sentences) == 3
    assert sentences[1] == "His blood pressure is 98/62 mm Hg."
    assert sentences[2].startswith("The causal pathogen")


def test_split_free_text_prose_and_numbered():
    assert split_free_text("He has fever. He takes aspirin. It works.") == ["He has fever.", "He takes aspirin.", "It works."]
    assert len(split_free_text(RAW_WITH_CONTINUATION)) == 3


def test_number_sentences():
    assert number_sentences(["a", "b"]) == "0 a\n1 b"


def test_split_by_case_keeps_cases_together(toy_df):
    splits = split_by_case(toy_df, val_size=0.2, test_size=0.2, seed=0)
    check_no_leakage(splits)
    assert sum(len(s) for s in splits.values()) == len(toy_df)
    holder = [name for name, s in splits.items() if "t0" in set(s["text_id"])][0]
    assert "t1" in set(splits[holder]["text_id"])
