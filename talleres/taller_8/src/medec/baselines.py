"""Reference baseline the models must beat.

The majority class of train is "has error", so the majority-class baseline is the
same as this one: flag 1 and the last sentence (where 62.5% of the errors are).
"""

from __future__ import annotations

import pandas as pd

from medec.schemas import Prediction


def predict_last_sentence(df: pd.DataFrame) -> list[Prediction]:
    """Always "the error is in the last sentence" (62.5% of errors in train are there)."""
    return [
        Prediction(text_id=r.text_id, error_flag=1, error_sentence_id=len(r.sentences) - 1)
        for r in df.itertuples()
    ]
