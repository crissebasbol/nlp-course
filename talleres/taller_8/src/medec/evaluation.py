"""Step 6: metrics (Accuracy for A and B, ROUGE-1 and BERTScore for C) and comparison tables."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from medec.schemas import Prediction, predictions_by_id

_ROUGE_1 = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=False)


def _gold_sentence_id(row) -> int:
    return int(row.error_sentence_id) if int(row.error_flag) else -1


def align(gold_df: pd.DataFrame, preds: list[Prediction]) -> list[tuple[Any, Prediction]]:
    """Pairs (gold row, prediction); texts without prediction count as "no error"."""
    by_id = predictions_by_id(preds)
    return [(row, by_id.get(row.text_id, Prediction(row.text_id, 0))) for row in gold_df.itertuples()]


def compute_flag_accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if len(y_true) else 0.0


def compute_sentence_accuracy(y_true, y_pred) -> float:
    """Accuracy over all texts, where texts without error have ID -1 (as in the shared task)."""
    return compute_flag_accuracy(y_true, y_pred)


def compute_bertscore(refs: list[str], cands: list[str], model_type: str) -> list[float]:
    if not refs:
        return []
    from bert_score import score

    _, _, f1 = score(cands, refs, model_type=model_type, lang="en", verbose=False, batch_size=16)
    return f1.tolist()


def compute_correction_scores(gold_df: pd.DataFrame, preds: list[Prediction], bertscore_model: str | None) -> dict[str, float]:
    """ROUGE-1 and BERTScore for subtask C.

    "official" follows the MEDIQA-CORR rule: both NA -> 1, only one NA -> 0, otherwise the metric.
    "pairs" averages only over texts where reference and prediction both have a correction.
    """
    per_text, pair_refs, pair_cands, pair_pos = [], [], [], []
    for row, pred in align(gold_df, preds):
        ref = row.corrected_sentence if int(row.error_flag) else None
        cand = pred.corrected_sentence if pred.error_flag else None
        if ref is None and cand is None:
            per_text.append({"rouge1": 1.0, "bertscore": 1.0})
        elif ref is None or cand is None:
            per_text.append({"rouge1": 0.0, "bertscore": 0.0})
        else:
            pair_pos.append(len(per_text))
            pair_refs.append(ref)
            pair_cands.append(cand)
            per_text.append({"rouge1": _ROUGE_1.score(ref, cand)["rouge1"].fmeasure, "bertscore": np.nan})

    if bertscore_model and pair_refs:
        for pos, value in zip(pair_pos, compute_bertscore(pair_refs, pair_cands, bertscore_model)):
            per_text[pos]["bertscore"] = value
    frame = pd.DataFrame(per_text)
    pairs = frame.iloc[pair_pos] if pair_pos else frame.iloc[0:0]
    result = {
        "rouge1_official": frame["rouge1"].mean(),
        "rouge1_pairs": pairs["rouge1"].mean() if len(pairs) else np.nan,
        "n_pairs": len(pairs),
    }
    if bertscore_model:
        result["bertscore_official"] = frame["bertscore"].mean()
        result["bertscore_pairs"] = pairs["bertscore"].mean() if len(pairs) else np.nan
    return result


def detection_metrics(gold_df: pd.DataFrame, preds: list[Prediction]) -> dict[str, float]:
    pairs = align(gold_df, preds)
    flags_true = [int(r.error_flag) for r, _ in pairs]
    flags_pred = [p.error_flag for _, p in pairs]
    ids_true = [_gold_sentence_id(r) for r, _ in pairs]
    ids_pred = [p.error_sentence_id for _, p in pairs]
    on_errors = [(t, p) for t, p, f in zip(ids_true, ids_pred, flags_true) if f]
    return {
        "flag_acc": compute_flag_accuracy(flags_true, flags_pred),
        "sentence_acc": compute_sentence_accuracy(ids_true, ids_pred),
        "sentence_acc_on_errors": compute_flag_accuracy(*zip(*on_errors)) if on_errors else np.nan,
        "n": len(pairs),
    }


def compare_detectors(gold_df: pd.DataFrame, systems: dict[str, list[Prediction]]) -> pd.DataFrame:
    """Table of Accuracy (A and B) for every system."""
    rows = [{"system": name, **detection_metrics(gold_df, preds)} for name, preds in systems.items()]
    return pd.DataFrame(rows).set_index("system").round(4)


def compare_correctors(gold_df: pd.DataFrame, systems: dict[str, list[Prediction]], bertscore_model: str | None) -> pd.DataFrame:
    """Table of ROUGE-1 and BERTScore (C) for every system that corrects."""
    rows = [{"system": name, **compute_correction_scores(gold_df, preds, bertscore_model)} for name, preds in systems.items()]
    return pd.DataFrame(rows).set_index("system").round(4)


ERROR_TYPE_RULES = [
    ("CausalOrganism", r"organism|pathogen|infect|caused by|culture|bacteri|virus|spider"),
    ("Diagnosis", r"diagnos|suspect|consistent with|most likely"),
    ("Pharmacotherapy", r"\b(prescribed|administered|started on|dose|medication|drug|\d+ ?mg)\b(?!/)"),
    ("Management", r"next step|referred|perform|ordered|scheduled|recommended|undergo|testing|discussed"),
    ("Treatment", r"treat|therap|surgery|surgical|repair|resection|initiated"),
]


def approximate_error_type(sentence: str | None) -> str:
    """Heuristic error type (the MS subset has no type label)."""
    s = (sentence or "").lower()
    for name, pattern in ERROR_TYPE_RULES:
        if re.search(pattern, s):
            return name
    return "Other"


def error_analysis(gold_df: pd.DataFrame, preds: list[Prediction]) -> dict[str, pd.DataFrame]:
    """Sentence accuracy on error texts by position (last or not) and by approximate type."""
    rows = []
    for row, pred in align(gold_df, preds):
        if not int(row.error_flag):
            continue
        rows.append({
            "position": "last" if row.error_sentence_id == len(row.sentences) - 1 else "not_last",
            "error_type": approximate_error_type(row.error_sentence),
            "hit": pred.error_sentence_id == row.error_sentence_id,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    agg = lambda col: frame.groupby(col)["hit"].agg(["mean", "count"]).round(3)  # noqa: E731
    return {"by_position": agg("position"), "by_error_type": agg("error_type")}
