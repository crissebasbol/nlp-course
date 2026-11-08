"""Inference service: loads the trained artifacts once and runs the pipeline on one text."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from medec.combine import combine_scores, decide_error_flag, select_error_sentence
from medec.config import get_output_dir
from medec.data import split_free_text
from medec.llm_server import is_llm_ready
from medec.quality import quality_check_correction

logger = logging.getLogger(__name__)

STEPS = [
    ("prepare", "1-2. Preparar oraciones"),
    ("bert", "3a. BiomedBERT"),
    ("dspy", "3b. Qwen + DSPy"),
    ("combine", "3c. Combinar y decidir"),
    ("correct", "4. Corregir"),
    ("quality", "5. Revisar corrección"),
]


def word_diff(before: str, after: str) -> list[dict[str, str]]:
    """Word-level diff for the UI: [{op: equal|delete|insert, text}]."""
    a, b = before.split(), after.split()
    out = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes():
        if op == "equal":
            out.append({"op": "equal", "text": " ".join(a[i1:i2])})
        else:
            if i2 > i1:
                out.append({"op": "delete", "text": " ".join(a[i1:i2])})
            if j2 > j1:
                out.append({"op": "insert", "text": " ".join(b[j1:j2])})
    return out


class MedecService:
    """Holds BERT, the DSPy program and the corrector; each part is optional."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.out = get_output_dir(cfg)
        self.api_base = cfg["models"]["llm_api_base"]
        self.bert = None
        self.dspy_program = None
        self.corrector = None
        self.dspy_compiled = False
        self.combine_cfg = self._load_json(self.out / "artifacts" / "combine.json") or {}
        self.examples = self._load_examples()
        self._load_bert()
        self._load_dspy()
        self._load_corrector()

    @staticmethod
    def _load_json(path: Path) -> dict | None:
        return json.loads(path.read_text()) if path.exists() else None

    def _load_examples(self) -> pd.DataFrame:
        path = self.out / "splits" / "test.parquet"
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_parquet(path)
        df["sentences"] = df["sentences"].map(list)
        return df.set_index("text_id", drop=False)

    def _load_bert(self) -> None:
        model_dir = self.out / "models" / "bert" / "final"
        if not model_dir.exists():
            logger.warning("BERT model not found in %s", model_dir)
            return
        import torch

        from medec.detector_bert import BertScorer

        torch.set_num_threads(max(torch.get_num_threads(), 1))
        self.bert = BertScorer(model_dir, self.cfg["bert"]["max_length"])
        logger.info("BERT loaded on %s", self.bert.device)

    def _load_dspy(self) -> None:
        from medec.detector_dspy import configure_dspy_lm, load_compiled_program

        d, m = self.cfg["dspy"], self.cfg["models"]
        configure_dspy_lm(self.api_base, m["llm_name"], d["temperature"], d["max_tokens"], d["timeout"])
        path = self.out / "models" / "dspy_detector.json"
        self.dspy_compiled = path.exists()
        self.dspy_program = load_compiled_program(path)

    def _load_corrector(self) -> None:
        from medec.corrector import Corrector, load_encoder, load_example_index

        art = self.out / "artifacts"
        if not (art / "fewshot_embeddings.npy").exists():
            logger.warning("Few-shot index not found in %s", art)
            return
        encoder = load_encoder(self.cfg["models"]["embeddings"])
        self.corrector = Corrector(load_example_index(art), encoder, self.api_base, self.cfg["models"]["llm_name"], self.cfg["corrector"])

    # ------------------------------------------------------------------ status
    def status(self) -> dict[str, Any]:
        combined = self.combine_cfg.get("combined", {})
        bert = self.combine_cfg.get("bert", {})
        return {
            "bert": self.bert is not None,
            "llm": is_llm_ready(self.api_base),
            "llm_api_base": self.api_base,
            "dspy_compiled": self.dspy_compiled,
            "corrector": self.corrector is not None,
            "examples": len(self.examples),
            "defaults": {
                "weight": combined.get("w", self.cfg["combine"]["default_weight"]),
                "threshold": combined.get("threshold", self.cfg["combine"]["default_threshold"]),
                "bert_threshold": bert.get("threshold", self.cfg["combine"]["default_threshold"]),
            },
        }

    def example_list(self) -> list[dict[str, Any]]:
        if self.examples.empty:
            return []
        return [
            {"text_id": r.text_id, "error_flag": int(r.error_flag), "n_sentences": len(r.sentences), "preview": r.sentences[0][:90]}
            for r in self.examples.itertuples()
        ]

    def example(self, text_id: str) -> dict[str, Any] | None:
        if text_id not in self.examples.index:
            return None
        r = self.examples.loc[text_id]
        flag = int(r.error_flag)
        return {
            "text_id": text_id,
            "sentences": list(r.sentences),
            "error_flag": flag,
            "error_sentence_id": int(r.error_sentence_id) if flag else -1,
            "corrected_sentence": r.corrected_sentence if flag else None,
        }

    # --------------------------------------------------------------- pipeline
    def analyze(self, payload: dict[str, Any], report: Callable[[str, str, float | None], None]) -> dict[str, Any]:
        """Run the pipeline on one text. `report(step, status, seconds)` updates the job."""

        def timed(step: str, fn: Callable[[], Any]) -> Any:
            report(step, "running", None)
            start = time.time()
            try:
                result = fn()
            except Exception:
                report(step, "error", time.time() - start)
                raise
            report(step, "done", round(time.time() - start, 2))
            return result

        gold = self.example(payload["text_id"]) if payload.get("text_id") else None
        if not payload.get("text") and gold is None:
            raise ValueError(f"No existe el ejemplo {payload.get('text_id')} en el conjunto de prueba")
        sentences = timed("prepare", lambda: gold["sentences"] if gold and not payload.get("text") else split_free_text(payload["text"]))
        if not sentences:
            raise ValueError("El texto no tiene oraciones")

        use_bert = payload.get("use_bert", True) and self.bert is not None
        use_dspy = payload.get("use_dspy", True)
        if not use_bert and not use_dspy:
            raise ValueError("Elige al menos un detector")

        p_bert = timed("bert", lambda: self.bert.predict_bert_scores(sentences)) if use_bert else None
        if not use_bert:
            report("bert", "skipped", None)

        dspy_out = None
        if use_dspy:
            from medec.detector_dspy import predict_dspy_scores

            dspy_out = timed("dspy", lambda: predict_dspy_scores(self.dspy_program, sentences))
        else:
            report("dspy", "skipped", None)

        defaults = self.status()["defaults"]

        def decide():
            if use_bert and use_dspy:
                w = float(payload.get("weight", defaults["weight"]))
                t = float(payload.get("threshold", defaults["threshold"]))
                scores = combine_scores(p_bert, dspy_out["scores"], w)
                flag = decide_error_flag(scores, t)
                return scores, flag, select_error_sentence(scores, flag), {"mode": "combined", "weight": w, "threshold": t}
            if use_bert:
                t = float(payload.get("threshold", defaults["bert_threshold"]))
                flag = decide_error_flag(p_bert, t)
                return p_bert, flag, select_error_sentence(p_bert, flag), {"mode": "bert", "threshold": t}
            return dspy_out["scores"], int(dspy_out["has_error"]), dspy_out["error_sentence_id"], {"mode": "dspy"}

        scores, flag, sid, decision = timed("combine", decide)

        correction, qc, examples = None, None, []
        if flag and payload.get("correct", True) and self.corrector is not None:
            res = timed("correct", lambda: self.corrector.correct(sentences, sid))
            correction, examples = res["corrected_sentence"], res["examples"]
            qc = timed("quality", lambda: quality_check_correction(sentences[sid], correction, self.cfg["quality"]["min_rouge"]))
        else:
            report("correct", "skipped", None)
            report("quality", "skipped", None)

        return {
            "sentences": sentences,
            "p_bert": p_bert,
            "s_dspy": dspy_out["scores"] if dspy_out else None,
            "dspy": {k: dspy_out.get(k) for k in ("has_error", "error_sentence_id", "confidence", "reasoning", "failed")} if dspy_out else None,
            "scores": [round(float(s), 4) for s in scores],
            "error_flag": flag,
            "error_sentence_id": sid,
            "decision": decision,
            "corrected_sentence": correction,
            "diff": word_diff(sentences[sid], correction) if correction else None,
            "quality": qc,
            "few_shot_examples": examples,
            "gold": gold,
        }


class JobManager:
    """Runs one analysis at a time (the LLM serves one request at a time on CPU)."""

    def __init__(self, service: MedecService, max_jobs: int = 200):
        self.service = service
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.max_jobs = max_jobs

    def submit(self, payload: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id, "status": "queued", "created": time.time(), "result": None, "error": None,
            "steps": [{"id": sid, "label": label, "status": "pending", "seconds": None} for sid, label in STEPS],
        }
        with self.lock:
            if len(self.jobs) >= self.max_jobs:
                oldest = min(self.jobs.values(), key=lambda j: j["created"])
                self.jobs.pop(oldest["id"], None)
            self.jobs[job_id] = job
        self.pool.submit(self._run, job, payload)
        return job_id

    def _run(self, job: dict[str, Any], payload: dict[str, Any]) -> None:
        def report(step: str, status: str, seconds: float | None) -> None:
            for s in job["steps"]:
                if s["id"] == step:
                    s["status"], s["seconds"] = status, seconds

        job["status"] = "running"
        start = time.time()
        try:
            job["result"] = self.service.analyze(payload, report)
            job["status"] = "done"
        except Exception as exc:  # shown in the UI
            logger.exception("Job %s failed", job["id"])
            job["status"], job["error"] = "error", str(exc)
        job["seconds"] = round(time.time() - start, 2)

    def get(self, job_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        queued_before = sum(1 for j in self.jobs.values() if j["status"] == "queued" and j["created"] < job["created"])
        return {**job, "queue_position": queued_before}
