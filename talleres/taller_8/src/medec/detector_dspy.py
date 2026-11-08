"""Step 3b: LLM detector (Qwen served by llama.cpp) with a DSPy-optimized prompt."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import dspy
import pandas as pd

from medec.data import number_sentences

logger = logging.getLogger(__name__)


def configure_dspy_lm(api_base: str, model_name: str, temperature: float = 0.0, max_tokens: int = 600,
                      timeout: float = 900, cache: bool = True) -> dspy.LM:
    """Point DSPy at the local OpenAI-compatible server (llama-server)."""
    lm = dspy.LM(
        f"openai/{model_name}",
        api_base=api_base,
        api_key="local",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        cache=cache,
        num_retries=1,
    )
    dspy.configure(lm=lm)
    return lm


class DetectErrorSignature(dspy.Signature):
    """You are a physician reviewing a clinical note. The note is either fully correct or
    contains exactly one medical error (a wrong diagnosis, management step, treatment,
    pharmacotherapy or causal organism). Sentences are numbered from 0. Decide whether the
    note has an error and, if it does, which sentence contains it."""

    numbered_text: str = dspy.InputField(desc="Clinical note, one numbered sentence per line")
    has_error: bool = dspy.OutputField(desc="True if the note contains a medical error")
    error_sentence_id: int = dspy.OutputField(desc="Number of the sentence with the error, or -1 if there is none")
    confidence: float = dspy.OutputField(desc="Confidence in the answer, between 0 and 1")


def build_dspy_detector() -> dspy.Module:
    return dspy.ChainOfThought(DetectErrorSignature)


def to_dspy_example(row: pd.Series | Any) -> dspy.Example:
    flag = bool(int(row.error_flag))
    return dspy.Example(
        numbered_text=number_sentences(list(row.sentences)),
        has_error=flag,
        error_sentence_id=int(row.error_sentence_id) if flag else -1,
    ).with_inputs("numbered_text")


def detection_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """1 if flag and sentence are right, 0.5 if only the flag is right.

    During compilation (trace is not None) only fully correct answers are kept as demos.
    """
    try:
        flag_ok = bool(pred.has_error) == bool(example.has_error)
        pred_id = int(pred.error_sentence_id) if pred.has_error else -1
    except (TypeError, ValueError, AttributeError):
        return 0.0
    sentence_ok = pred_id == int(example.error_sentence_id)
    score = 1.0 if (flag_ok and sentence_ok) else (0.5 if flag_ok else 0.0)
    return float(score == 1.0) if trace is not None else score


def compile_dspy_detector(trainset: list[dspy.Example], max_bootstrapped_demos: int = 3,
                          max_labeled_demos: int = 3) -> dspy.Module:
    """BootstrapFewShot: runs the program on train texts and keeps the right answers as demos."""
    optimizer = dspy.BootstrapFewShot(
        metric=detection_metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
    )
    return optimizer.compile(build_dspy_detector(), trainset=trainset)


def save_compiled_program(program: dspy.Module, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    program.save(str(path))


def load_compiled_program(path: str | Path | None) -> dspy.Module:
    """Compiled program if the file exists; otherwise the zero-shot module."""
    program = build_dspy_detector()
    if path and Path(path).exists():
        program.load(str(path))
    return program


def _clip(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return min(max(float(value), low), high)
    except (TypeError, ValueError):
        return 0.5


def predict_dspy_scores(program: dspy.Module, sentences: list[str]) -> dict[str, Any]:
    """s_dspy: confidence on the chosen sentence and 0 on the others (all 0 when no error)."""
    n = len(sentences)
    scores = [0.0] * n
    try:
        pred = program(numbered_text=number_sentences(sentences))
    except Exception as exc:  # malformed output, timeout: counted as "no error"
        logger.warning("DSPy detector failed: %s", exc)
        return {"scores": scores, "has_error": False, "error_sentence_id": -1, "failed": True, "reasoning": ""}

    has_error = bool(getattr(pred, "has_error", False))
    try:
        sid = int(getattr(pred, "error_sentence_id", -1))
    except (TypeError, ValueError):
        sid = -1
    confidence = _clip(getattr(pred, "confidence", 0.5))
    if has_error and 0 <= sid < n:
        scores[sid] = confidence
    else:
        has_error, sid = False, -1
    return {
        "scores": scores,
        "has_error": has_error,
        "error_sentence_id": sid,
        "confidence": confidence,
        "reasoning": str(getattr(pred, "reasoning", ""))[:2000],
        "failed": False,
    }


def predict_split_dspy(program: dspy.Module, df: pd.DataFrame, num_threads: int = 4,
                       done_ids: set[str] | None = None,
                       on_result: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    """Run the detector over a split in parallel; skip text_ids already processed."""
    done_ids = done_ids or set()
    rows = [r for r in df.itertuples() if r.text_id not in done_ids]

    def run(row) -> dict[str, Any]:
        out = {"text_id": row.text_id, **predict_dspy_scores(program, list(row.sentences))}
        if on_result:
            on_result(out)
        return out

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        return list(pool.map(run, rows))
