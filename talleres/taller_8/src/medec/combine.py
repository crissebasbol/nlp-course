"""Step 3c: weighted combination of p_bert and s_dspy, and the final decision (subtasks A and B)."""

from __future__ import annotations

from typing import Any

import numpy as np


def combine_scores(p_bert: list[float] | None, s_dspy: list[float] | None, w: float) -> list[float]:
    """Final score per sentence: w * p_bert + (1 - w) * s_dspy.

    If one detector is missing, the other one is used alone.
    """
    if p_bert is None and s_dspy is None:
        raise ValueError("At least one score vector is needed")
    if s_dspy is None:
        return list(map(float, p_bert))
    if p_bert is None:
        return list(map(float, s_dspy))
    if len(p_bert) != len(s_dspy):
        raise ValueError(f"Length mismatch: {len(p_bert)} vs {len(s_dspy)}")
    return (w * np.asarray(p_bert, float) + (1 - w) * np.asarray(s_dspy, float)).tolist()


def decide_error_flag(scores: list[float], threshold: float) -> int:
    """Subtask A: 1 if the most suspicious sentence passes the threshold."""
    return int(len(scores) > 0 and max(scores) > threshold)


def select_error_sentence(scores: list[float], flag: int) -> int:
    """Subtask B: argmax of the scores, or -1 when there is no error."""
    return int(np.argmax(scores)) if flag else -1


def detection_objective(records: list[dict[str, Any]], w: float, threshold: float) -> dict[str, float]:
    """Accuracy of A and B for one (w, threshold) on records with p_bert, s_dspy and gold labels."""
    flag_hits, sentence_hits = 0, 0
    for rec in records:
        scores = combine_scores(rec.get("p_bert"), rec.get("s_dspy"), w)
        flag = decide_error_flag(scores, threshold)
        sid = select_error_sentence(scores, flag)
        flag_hits += flag == rec["error_flag"]
        sentence_hits += sid == (rec["error_sentence_id"] if rec["error_flag"] else -1)
    n = max(len(records), 1)
    return {"flag_acc": flag_hits / n, "sentence_acc": sentence_hits / n}


def tune_weight_and_threshold(
    records: list[dict[str, Any]],
    weights: list[float] | None = None,
    thresholds: list[float] | None = None,
) -> dict[str, Any]:
    """Grid search on validation; maximizes accuracy(A) + accuracy(B).

    w = 1 is BERT alone and w = 0 is DSPy alone, so the same grid compares both detectors.
    """
    weights = weights if weights is not None else [round(x, 2) for x in np.linspace(0, 1, 11)]
    thresholds = thresholds if thresholds is not None else [round(x, 3) for x in np.linspace(0.05, 0.95, 37)]
    grid = []
    for w in weights:
        for t in thresholds:
            res = detection_objective(records, w, t)
            grid.append({"w": w, "threshold": t, **res, "objective": res["flag_acc"] + res["sentence_acc"]})
    best = max(grid, key=lambda g: (g["objective"], g["w"]))
    best_by_w = {}
    for g in grid:
        if g["w"] not in best_by_w or g["objective"] > best_by_w[g["w"]]["objective"]:
            best_by_w[g["w"]] = g
    return {"best": best, "best_by_weight": list(best_by_w.values())}
