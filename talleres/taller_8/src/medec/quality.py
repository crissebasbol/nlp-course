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


_ROUGE_1 = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)


def _words(text: str) -> list[str]:
    return [w.strip(".,;:()[]\"'").lower() for w in normalize_text(text).split() if w.strip(".,;:()[]\"'")]


def compare_with_reference(original: str, correction: str, reference: str) -> dict[str, Any]:
    """How close a correction is to the reference one (only possible on labeled examples).

    key_terms are the words the reference adds to the original sentence (the actual fix, e.g. the
    right organism or drug); key_coverage is the fraction of them that the correction contains.
    """
    orig_words, corr_words = set(_words(original)), set(_words(correction))
    key_terms = [w for w in dict.fromkeys(_words(reference)) if w not in orig_words]
    found = [w for w in key_terms if w in corr_words]
    coverage = len(found) / len(key_terms) if key_terms else None
    if _words(correction) == _words(reference):
        verdict = "exact"
    elif coverage is not None and coverage >= 0.5:
        verdict = "key_terms"
    else:
        verdict = "different"
    return {
        "rouge1": round(_ROUGE_1.score(reference, correction)["rouge1"].fmeasure, 4),
        "key_terms": key_terms,
        "key_found": found,
        "key_coverage": None if coverage is None else round(coverage, 2),
        "verdict": verdict,
    }
