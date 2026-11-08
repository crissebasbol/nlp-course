"""Step 5: quality check of the correction and output formatting."""

from __future__ import annotations

from typing import Any

from rouge_score import rouge_scorer

from medec.data import normalize_text
from medec.schemas import Prediction

_ROUGE_L = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def quality_check_correction(error_sentence: str, correction: str, min_rouge: float = 0.7) -> dict[str, Any]:
    """A correction is doubtful if it is identical to the original or changes too much.

    It is only flagged for error analysis (idea from WangLab); the prediction is kept.
    """
    rouge_l = _ROUGE_L.score(error_sentence, correction)["rougeL"].fmeasure
    identical = normalize_text(error_sentence).lower() == normalize_text(correction).lower()
    reason = "identical" if identical else ("changed_too_much" if rouge_l < min_rouge else None)
    return {"rouge_l": round(rouge_l, 4), "doubtful": reason is not None, "reason": reason}


def format_prediction(text_id: str, flag: int, sentence_id: int, correction: str | None) -> str:
    """One line in the official shared task format: `text_id flag sentence_id "correction"|NA`."""
    if not flag:
        return f"{text_id} 0 -1 NA"
    return f'{text_id} 1 {sentence_id} "{normalize_text(correction or "")}"'


def to_official_lines(preds: list[Prediction]) -> list[str]:
    return [format_prediction(p.text_id, p.error_flag, p.error_sentence_id, p.corrected_sentence) for p in preds]
