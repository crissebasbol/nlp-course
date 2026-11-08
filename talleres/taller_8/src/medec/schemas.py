"""Single prediction format shared by every system, plus JSONL helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Prediction:
    """Output of one system for one text.

    error_sentence_id is -1 and corrected_sentence is None when error_flag is 0.
    """

    text_id: str
    error_flag: int
    error_sentence_id: int = -1
    corrected_sentence: str | None = None
    scores: list[float] = field(default_factory=list)
    doubtful: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.error_flag = int(self.error_flag)
        self.error_sentence_id = int(self.error_sentence_id)
        if not self.error_flag:
            self.error_sentence_id = -1
            self.corrected_sentence = None


def save_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    """Append one record; used by long runs so they can resume after a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_predictions(preds: Iterable[Prediction], path: str | Path) -> Path:
    return save_jsonl((asdict(p) for p in preds), path)


def load_predictions(path: str | Path) -> list[Prediction]:
    return [Prediction(**rec) for rec in load_jsonl(path)]


def predictions_by_id(preds: Iterable[Prediction]) -> dict[str, Prediction]:
    return {p.text_id: p for p in preds}
