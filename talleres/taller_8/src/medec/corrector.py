"""Step 4: rewrite the wrong sentence with Qwen, using similar train examples as few-shot."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medec.llm_server import chat_completion

SYSTEM_PROMPT = (
    "You are a physician correcting clinical notes. You receive a clinical note and one "
    "sentence of it that contains a medical error (a wrong diagnosis, management step, "
    "treatment, pharmacotherapy or causal organism). Rewrite that sentence fixing the error. "
    "Change only the wrong medical concept and keep the rest of the sentence as it is. "
    "Answer with the corrected sentence only, without explanations or quotes."
)


def load_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def build_example_index(train_df: pd.DataFrame, encoder) -> dict[str, Any]:
    """Embeddings of the wrong sentences of train, to find similar corrections later."""
    errors = train_df[train_df["error_flag"] == 1]
    examples = pd.DataFrame({
        "text_id": errors["text_id"].values,
        "error_sentence": errors["error_sentence"].values,
        "corrected_sentence": errors["corrected_sentence"].values,
    }).reset_index(drop=True)
    emb = encoder.encode(examples["error_sentence"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    return {"examples": examples, "embeddings": np.asarray(emb, dtype=np.float32)}


def save_example_index(index: dict[str, Any], folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    index["examples"].to_parquet(folder / "fewshot_examples.parquet", index=False)
    np.save(folder / "fewshot_embeddings.npy", index["embeddings"])


def load_example_index(folder: str | Path) -> dict[str, Any]:
    folder = Path(folder)
    return {
        "examples": pd.read_parquet(folder / "fewshot_examples.parquet"),
        "embeddings": np.load(folder / "fewshot_embeddings.npy"),
    }


def select_few_shot_examples(index: dict[str, Any], query: str, encoder, k: int = 3) -> list[dict[str, str]]:
    """The k train corrections whose wrong sentence is most similar to the query."""
    q = encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = index["embeddings"] @ q
    top = np.argsort(-sims)[:k]
    return [
        {**index["examples"].iloc[i][["error_sentence", "corrected_sentence"]].to_dict(), "similarity": float(sims[i])}
        for i in top
    ]


def build_correction_prompt(numbered_text: str, error_sentence: str, examples: list[dict[str, str]]) -> list[dict[str, str]]:
    """Chat messages: system instructions, examples as previous turns, then the query."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        messages.append({"role": "user", "content": f"Sentence with the error:\n{ex['error_sentence']}"})
        messages.append({"role": "assistant", "content": ex["corrected_sentence"]})
    messages.append({
        "role": "user",
        "content": f"Clinical note:\n{numbered_text}\n\nSentence with the error:\n{error_sentence}",
    })
    return messages


def generate_correction(messages: list[dict[str, str]], api_base: str, model: str,
                        temperature: float = 0.0, max_tokens: int = 160, timeout: float = 900) -> str:
    return chat_completion(api_base, model, messages, temperature, max_tokens, timeout)


PREFIXES = re.compile(r"^(corrected sentence|correction|answer|sentence)\s*[:\-]\s*", re.IGNORECASE)


def extract_corrected_sentence(output: str) -> str:
    """Keep only the sentence: drop prefixes, quotes, leading IDs and extra lines."""
    lines = [l.strip() for l in str(output).strip().split("\n") if l.strip()]
    if not lines:
        return ""
    text = PREFIXES.sub("", lines[0])
    text = re.sub(r"^\d+\s+(?=[A-Z])", "", text)
    return text.strip().strip('"').strip("'").strip("*").strip()


class Corrector:
    """Bundles the encoder, the example index and the LLM settings."""

    def __init__(self, index: dict[str, Any], encoder, api_base: str, model: str, params: dict[str, Any]):
        self.index, self.encoder = index, encoder
        self.api_base, self.model, self.params = api_base, model, params

    def correct(self, sentences: list[str], error_sentence_id: int) -> dict[str, Any]:
        from medec.data import number_sentences

        error_sentence = sentences[error_sentence_id]
        examples = select_few_shot_examples(self.index, error_sentence, self.encoder, self.params.get("k_examples", 3))
        messages = build_correction_prompt(number_sentences(sentences), error_sentence, examples)
        raw = generate_correction(
            messages, self.api_base, self.model,
            self.params.get("temperature", 0.0), self.params.get("max_tokens", 160), self.params.get("timeout", 900),
        )
        return {"corrected_sentence": extract_corrected_sentence(raw) or error_sentence, "raw": raw, "examples": examples}
