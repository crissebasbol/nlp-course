import pandas as pd
import pytest

RAW_WITH_CONTINUATION = (
    "0 A 53-year-old man has fever.\n"
    "1 His blood pressure is 98/62 mm\n"
    "Hg.\n"
    "2 The causal pathogen was Haemophilus influenzae."
)


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """Two clinical cases; case A appears twice (correct and wrong versions)."""
    rows = [
        ("t0", ["Case A first.", "Middle.", "Wrong drug given."], 1, 2, "Wrong drug given.", "Right drug given."),
        ("t1", ["Case A first.", "Middle.", "Right drug given."], 0, -1, None, None),
        ("t2", ["Case B first.", "Something else."], 1, 1, "Something else.", "Something right."),
        ("t3", ["Case C first.", "Fine."], 0, -1, None, None),
        ("t4", ["Case D first.", "Wrong organism."], 1, 1, "Wrong organism.", "Right organism."),
        ("t5", ["Case E first.", "Ok."], 0, -1, None, None),
    ]
    df = pd.DataFrame(rows, columns=["text_id", "sentences", "error_flag", "error_sentence_id", "error_sentence", "corrected_sentence"])
    df["group"] = df["sentences"].map(lambda s: s[0].lower())
    return df
