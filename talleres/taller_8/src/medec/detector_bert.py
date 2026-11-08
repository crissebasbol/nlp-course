"""Step 3a: BiomedBERT with a 2-class head, fine-tuned on sentence pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn


def get_device() -> torch.device:
    if torch.cuda.is_available():  # GPU NVIDIA (Colab, servidores con GPU)
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # GPU Apple (M1, M2)
        return torch.device("mps")
    return torch.device("cpu") # Sin GPU (Servidores de despliegue)


def load_detector(model_name: str | Path):
    """BiomedBERT (or a fine-tuned checkpoint) with a sequence classification head."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return model, tokenizer


def _weighted_trainer_class():
    from transformers import Trainer

    class WeightedLossTrainer(Trainer):
        """Trainer whose cross-entropy gives more weight to the rare "error" class."""

        def __init__(self, *args, class_weights: list[float] | None = None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            weight = None
            if self.class_weights is not None:
                weight = torch.tensor(self.class_weights, dtype=outputs.logits.dtype, device=outputs.logits.device)
            loss = nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
            return (loss, outputs) if return_outputs else loss

    return WeightedLossTrainer


def train_detector(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    model_name: str,
    output_dir: str | Path,
    params: dict[str, Any],
    seed: int = 42,
) -> Path:
    """Fine-tune BiomedBERT and save the best checkpoint in output_dir/final."""
    from transformers import DataCollatorWithPadding, TrainingArguments

    from medec.data import to_hf_dataset
    from medec.pairs import compute_class_weights, encode_batch

    model, tokenizer = load_detector(model_name)
    max_length = params.get("max_length", 512)
    columns = ["sentence", "context", "label"]

    def prepare(df: pd.DataFrame):
        ds = to_hf_dataset(df[columns].to_dict("records"))
        ds = ds.map(lambda b: encode_batch(b, tokenizer, max_length), batched=True, remove_columns=["sentence", "context"])
        return ds.rename_column("label", "labels")

    train_ds, val_ds = prepare(train_pairs), prepare(val_pairs)
    output_dir = Path(output_dir)
    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=params.get("epochs", 2),
        per_device_train_batch_size=params.get("batch_size", 16),
        per_device_eval_batch_size=params.get("eval_batch_size", 32),
        learning_rate=float(params.get("learning_rate", 2e-5)),
        weight_decay=params.get("weight_decay", 0.01),
        warmup_steps=params.get("warmup_ratio", 0.1),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    trainer = _weighted_trainer_class()(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        class_weights=compute_class_weights(train_pairs["label"]),
    )
    trainer.train()
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return final_dir


class BertScorer:
    """Loads the fine-tuned detector once and scores the sentences of a text."""

    def __init__(self, model_dir: str | Path, max_length: int = 512, device: torch.device | None = None):
        self.model, self.tokenizer = load_detector(model_dir)
        self.device = device or get_device()
        self.model.to(self.device).eval()
        self.max_length = max_length

    @torch.inference_mode()
    def score_pairs(self, sentences: list[str], contexts: list[str], batch_size: int = 32) -> np.ndarray:
        probs = []
        for start in range(0, len(sentences), batch_size):
            enc = self.tokenizer(
                sentences[start : start + batch_size],
                contexts[start : start + batch_size],
                truncation="only_second",
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(probs) if probs else np.array([])

    def predict_bert_scores(self, sentences: list[str]) -> list[float]:
        """p_bert: probability that each sentence is the error (vector of length n)."""
        context = " ".join(sentences)
        return self.score_pairs(sentences, [context] * len(sentences)).round(6).tolist()


def predict_split_scores(scorer: BertScorer, df: pd.DataFrame, batch_size: int = 32) -> list[dict[str, Any]]:
    """p_bert for every text of a split, batching sentences across texts."""
    from medec.pairs import build_pairs_frame

    pairs = build_pairs_frame(df)
    pairs["p"] = scorer.score_pairs(pairs["sentence"].tolist(), pairs["context"].tolist(), batch_size)
    return [
        {"text_id": text_id, "scores": group.sort_values("sentence_id")["p"].round(6).tolist()}
        for text_id, group in pairs.groupby("text_id", sort=False)
    ]
