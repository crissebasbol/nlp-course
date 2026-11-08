"""External reference: Claude Opus 5.5 solves A, B and C in one call (MEDEC-style prompt).

It is not part of the pipeline; it only shows how far a large commercial LLM goes
without training. Uses the Message Batches API (50% cheaper) and structured outputs.
Only the MS subset (public) is sent to the API.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from medec.data import number_sentences
from medec.schemas import Prediction

# Instruction adapted from prompt P#1 of the MEDEC paper.
INSTRUCTION = (
    "The following is a medical narrative about a patient. You are a skilled medical doctor "
    "reviewing the clinical text. The text is either correct or contains one error. The text "
    "has one sentence per line. Each line starts with the sentence ID, followed by a space and "
    "then the sentence to check. Check every sentence of the text. If the text is correct, set "
    "has_error to false, error_sentence_id to -1 and corrected_sentence to an empty string. If "
    "the text has a medical error related to treatment, management, cause, or diagnosis, set "
    "has_error to true, error_sentence_id to the ID of the sentence containing the error, and "
    "corrected_sentence to a corrected version of that sentence."
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "has_error": {"type": "boolean"},
        "error_sentence_id": {"type": "integer"},
        "corrected_sentence": {"type": "string"},
    },
    "required": ["has_error", "error_sentence_id", "corrected_sentence"],
    "additionalProperties": False,
}


def load_api_key(env_file: str | Path | None = None) -> str:
    """Read ANTHROPIC_API_KEY from .env or the environment; never hardcoded."""
    from dotenv import load_dotenv

    load_dotenv(env_file)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (put it in taller_8/.env)")
    return key


def format_example(row) -> str:
    """One solved train example (prompt P#2 of MEDEC)."""
    answer = {
        "has_error": bool(row.error_flag),
        "error_sentence_id": int(row.error_sentence_id) if row.error_flag else -1,
        "corrected_sentence": row.corrected_sentence if row.error_flag else "",
    }
    return f"Example text:\n{number_sentences(list(row.sentences))}\nExample answer: {json.dumps(answer)}"


def build_claude_request(text_id: str, sentences: list[str], cfg: dict[str, Any], example: str | None = None) -> Request:
    """One batch request: instruction (+ optional example), the numbered text and the JSON schema."""
    system = INSTRUCTION + (f"\n\n{example}" if example else "")
    return Request(
        custom_id=text_id,
        params=MessageCreateParamsNonStreaming(
            model=cfg["model"],
            max_tokens=cfg["max_tokens"],
            system=system,
            messages=[{"role": "user", "content": number_sentences(sentences)}],
            output_config={"effort": cfg["effort"], "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        ),
    )


def parse_claude_message(msg, n_sentences: int) -> dict[str, Any]:
    """Answer of one message (thinking blocks are skipped); invalid sentence IDs count as "no error"."""
    out: dict[str, Any] = {
        "stop_reason": msg.stop_reason,
        "usage": {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens},
        "answer": None,
    }
    if msg.stop_reason == "refusal":
        return out
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        ans = json.loads(text)
    except json.JSONDecodeError:
        out["raw"] = text
        return out
    sid = int(ans.get("error_sentence_id", -1))
    flag = bool(ans.get("has_error")) and 0 <= sid < n_sentences
    out["answer"] = {
        "has_error": flag,
        "error_sentence_id": sid if flag else -1,
        "corrected_sentence": (ans.get("corrected_sentence") or None) if flag else None,
    }
    return out


def ask_claude(client: anthropic.Anthropic, sentences: list[str], cfg: dict[str, Any], example: str | None = None,
               price_in: float = 4.0, price_out: float = 20.0) -> dict[str, Any]:
    """One synchronous call (used by the web interface). Same prompt and schema as the batch run."""
    params = build_claude_request("ui", sentences, cfg, example)["params"]
    start = time.time()
    msg = client.messages.create(**params)
    out = parse_claude_message(msg, len(sentences))
    out["seconds"] = round(time.time() - start, 2)
    u = out["usage"]
    out["cost_usd"] = round((u["input_tokens"] * price_in + u["output_tokens"] * price_out) / 1e6, 4)
    out["model"] = msg.model
    return out


def count_prompt_tokens(client: anthropic.Anthropic, request: Request) -> int:
    params = request["params"]
    return client.messages.count_tokens(model=params["model"], system=params["system"], messages=params["messages"]).input_tokens


def submit_claude_batch(client: anthropic.Anthropic, requests: list[Request]) -> str:
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def wait_for_batch(client: anthropic.Anthropic, batch_id: str, poll_s: int = 60) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            return
        counts = batch.request_counts
        print(f"Lote {batch_id}: {batch.processing_status} (procesando {counts.processing}, listos {counts.succeeded})")
        time.sleep(poll_s)


def collect_claude_results(client: anthropic.Anthropic, batch_id: str) -> list[dict[str, Any]]:
    """Results keyed by custom_id (they arrive in any order); refusals and errors are recorded."""
    out = []
    for item in client.messages.batches.results(batch_id):
        if item.result.type == "succeeded":
            out.append(message_to_record(item.custom_id, item.result.message))
        else:
            out.append({"text_id": item.custom_id, "status": item.result.type})
    return out


def message_to_record(text_id: str, msg) -> dict[str, Any]:
    """Same record for a batch result and a synchronous call; refusals and bad JSON are recorded."""
    rec: dict[str, Any] = {"text_id": text_id, "status": "succeeded", "stop_reason": msg.stop_reason,
                           "usage": {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}}
    if msg.stop_reason == "refusal":
        rec["answer"] = None
        return rec
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        rec["answer"] = json.loads(text)
    except json.JSONDecodeError:
        rec["answer"], rec["raw"] = None, text
    return rec


def run_claude_sync(client: anthropic.Anthropic, requests: list[Request], num_threads: int = 8) -> list[dict[str, Any]]:
    """Normal (non-batch) calls in parallel: full price, but a predictable time of a few minutes."""
    from concurrent.futures import ThreadPoolExecutor

    def call(req: Request) -> dict[str, Any]:
        try:
            return message_to_record(req["custom_id"], client.messages.create(**req["params"]))
        except anthropic.APIError as exc:
            return {"text_id": req["custom_id"], "status": "errored", "error": str(exc)[:300]}

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        return list(pool.map(call, requests))


def to_prediction_format(results: list[dict[str, Any]], df: pd.DataFrame) -> list[Prediction]:
    """Same format as the pipeline; failed or refused answers count as "no error"."""
    n_sentences = dict(zip(df["text_id"], df["sentences"].map(len)))
    preds = []
    for rec in results:
        ans = rec.get("answer") or {}
        sid = int(ans.get("error_sentence_id", -1))
        flag = int(bool(ans.get("has_error")) and 0 <= sid < n_sentences.get(rec["text_id"], 0))
        preds.append(Prediction(
            text_id=rec["text_id"], error_flag=flag, error_sentence_id=sid,
            corrected_sentence=ans.get("corrected_sentence") or None,
            extra={"status": rec["status"], "stop_reason": rec.get("stop_reason")},
        ))
    return preds
