"""MedQA "answer sheet" without any LLM (test split).

BERT picks sentence k. The most similar MedQA question is retrieved (reusing medec.medqa_experiment).
  - k contains the correct answer          -> no error
  - k contains another option of the question -> error; correction = replace it by the answer
  - k contains no option                   -> undecided (counted as error, no correction)
Measures how much of the task the retrieved answer alone solves (leakage, not medical skill).
Usage: python medqa_sin_llm.py <cache_dir> [bertscore_model]
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]  # taller_8/
sys.path.insert(0, str(ROOT / "src"))
from medec.corrector import load_encoder  # noqa: E402
from medec.data import load_split  # noqa: E402
from medec.evaluation import compute_correction_scores, detection_metrics  # noqa: E402
from medec.medqa_experiment import build_medqa_index, fallback_correction, load_medqa, retrieve_similar_questions  # noqa: E402
from medec.schemas import Prediction, load_predictions  # noqa: E402

CACHE = Path(sys.argv[1])
CACHE.mkdir(parents=True, exist_ok=True)
BERTSCORE = sys.argv[2] if len(sys.argv) > 2 else None

encoder = load_encoder("NeuML/pubmedbert-base-embeddings")
# All MedQA splits: the MS texts may come from any of them
medqa = pd.concat([load_medqa(s).assign(medqa_split=s) for s in ("train", "test")], ignore_index=True)
idx_path = CACHE / "medqa_all_embeddings.npy"
if idx_path.exists():
    index = np.load(idx_path)
else:
    index = build_medqa_index(medqa["question"].tolist(), encoder)
    np.save(idx_path, index)
print(f"MedQA: {len(medqa)} preguntas ({medqa.medqa_split.value_counts().to_dict()})")

df = load_split(ROOT / "outputs" / "splits", "test")
bert = {r["text_id"]: r["scores"] for r in map(json.loads, open(ROOT / "outputs" / "scores" / "bert_test.jsonl"))}


def matched_option(sentence: str, options: dict) -> str | None:
    """Longest option text contained in the sentence (avoids 'X' matching inside 'X and Y')."""
    low = sentence.lower()
    found = [o for o in options.values() if o and o.lower() in low]
    return max(found, key=len) if found else None


preds, rows = [], []
for r in df.itertuples():
    sents = list(r.sentences)
    k = int(np.argmax(bert[r.text_id]))
    hit = retrieve_similar_questions(" ".join(sents), index, medqa, encoder)[0]
    opt = matched_option(sents[k], hit["options"])
    if opt is not None and opt == hit["answer"]:
        case, pred = "contiene la respuesta", Prediction(r.text_id, 0)
    elif opt is not None:
        case = "contiene otra opción"
        pred = Prediction(r.text_id, 1, k, fallback_correction(sents[k], [hit]))
    else:
        case, pred = "ninguna opción", Prediction(r.text_id, 1, k)
    preds.append(pred)
    rows.append({"text_id": r.text_id, "error_flag": int(r.error_flag), "case": case, "similarity": hit["similarity"]})

info = pd.DataFrame(rows)
print(f"\nSimilitud con la pregunta recuperada: mediana {info.similarity.median():.3f}, "
      f"p10 {info.similarity.quantile(.1):.3f}")
print("\nQué encontró la regla en la oración de BERT (filas) según la etiqueta real (columnas):")
print(pd.crosstab(info.case, info.error_flag.map({0: "sin error", 1: "con error"}), margins=True))

y = dict(zip(df.text_id, df.error_flag))
print(f"\n{'Sistema (test)':38s} {'A':>6s} {'B':>6s} {'B err':>6s} {'%marc':>6s} {'correctos':>9s}")
systems = {
    "BiomedBERT solo": load_predictions(ROOT / "outputs/predictions/bert_test.jsonl"),
    "Pipeline final actual": load_predictions(ROOT / "outputs/predictions/pipeline_test.jsonl"),
    "MedQA sin LLM (BERT + hoja de resp.)": preds,
    "Claude Opus 5.5": load_predictions(ROOT / "outputs/predictions/claude_test.jsonl"),
}
for label, p in systems.items():
    m = detection_metrics(df, p)
    tn = sum(1 for x in p if not x.error_flag and y.get(x.text_id) == 0)
    print(f"{label:38s} {m['flag_acc']:6.3f} {m['sentence_acc']:6.3f} {m['sentence_acc_on_errors']:6.3f} "
          f"{np.mean([x.error_flag for x in p]):6.0%} {tn:>5d}/144")

print("\nCorrección (C):")
for label in ("Pipeline final actual", "MedQA sin LLM (BERT + hoja de resp.)", "Claude Opus 5.5"):
    c = compute_correction_scores(df, systems[label], BERTSCORE)
    extra = f" BERTScore oficial={c['bertscore_official']:.3f}" if BERTSCORE else ""
    print(f"  {label:38s} ROUGE-1 oficial={c['rouge1_official']:.3f}{extra}")

out = CACHE / "medqa_sin_llm_test.jsonl"
with open(out, "w") as fh:
    for p, row in zip(preds, rows):
        fh.write(json.dumps({**p.__dict__, **row}) + "\n")
print(f"\nPredicciones -> {out}")
