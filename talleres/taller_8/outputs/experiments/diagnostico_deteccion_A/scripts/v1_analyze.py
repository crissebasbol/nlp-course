"""Option A: threshold tuned on val, evaluated on test with the project's own metrics."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[4]  # taller_8/
sys.path.insert(0, str(ROOT / "src"))
from medec.data import load_split  # noqa: E402
from medec.evaluation import compute_correction_scores, detection_metrics  # noqa: E402
from medec.schemas import Prediction, load_predictions  # noqa: E402

V1 = Path(sys.argv[1])
BERTSCORE = sys.argv[2] if len(sys.argv) > 2 else None


def load(split):
    df = load_split(ROOT / "outputs" / "splits", split)
    v = {r["text_id"]: r for r in map(json.loads, open(V1 / f"v1_{split}.jsonl"))}
    df = df[df.text_id.isin(v)]
    return df, v


def predict(df, v, t):
    """Error if P(correct) < t; the sentence is BERT's pick."""
    return [Prediction(r.text_id, int(v[r.text_id]["p_correct"] < t), v[r.text_id]["k"]) for r in df.itertuples()]


val_df, val_v = load("val")
test_df, test_v = load("test")
print(f"val: {len(val_df)} textos | test: {len(test_df)} textos")
for name, df, v in (("val", val_df, val_v), ("test", test_df, test_v)):
    y = df.error_flag.values
    s = np.array([1 - v[t]["p_correct"] for t in df.text_id])
    secs = np.mean([v[t]["seconds"] for t in df.text_id])
    toks = np.mean([v[t]["prompt_tokens"] for t in df.text_id])
    print(f"AUC {name}: {roc_auc_score(y, s):.3f} | {secs:.2f} s por texto en el Mac | {toks:.0f} tokens de prompt")

# Threshold on val: maximize A + B (same objective as the combination)
grid = sorted(set(np.round(np.linspace(0.01, 0.99, 99), 2)))
best = max(grid, key=lambda t: sum(detection_metrics(val_df, predict(val_df, val_v, t))[k] for k in ("flag_acc", "sentence_acc")))
print(f"\numbral elegido en val: P(correcta) < {best}")
for t in (best, 0.5):
    m = detection_metrics(val_df, predict(val_df, val_v, t))
    print(f"  val (umbral {t}): A={m['flag_acc']:.3f} B={m['sentence_acc']:.3f}")

# Test: option A vs the current systems
preds = predict(test_df, test_v, best)
rows = []
for label, p in (
    ("BiomedBERT solo", load_predictions(ROOT / "outputs/predictions/bert_test.jsonl")),
    ("Pipeline final actual", load_predictions(ROOT / "outputs/predictions/pipeline_test.jsonl")),
    (f"Opción A (umbral {best} de val)", preds),
    ("Opción A (umbral fijo 0.5)", predict(test_df, test_v, 0.5)),
    ("Claude Opus 5.5", load_predictions(ROOT / "outputs/predictions/claude_test.jsonl")),
):
    m = detection_metrics(test_df, p)
    flagged = np.mean([x.error_flag for x in p])
    y = dict(zip(test_df.text_id, test_df.error_flag))
    tn = sum(1 for x in p if not x.error_flag and y.get(x.text_id) == 0)
    rows.append((label, m["flag_acc"], m["sentence_acc"], m["sentence_acc_on_errors"], flagged, tn))
print(f"\nTEST ({len(test_df)} textos, {(test_df.error_flag == 0).sum()} correctos)")
print(f"{'Sistema':34s} {'A':>6s} {'B':>6s} {'B err':>6s} {'%marc':>6s} {'correctos reconocidos':>22s}")
for label, a, b, be, fl, tn in rows:
    print(f"{label:34s} {a:6.3f} {b:6.3f} {be:6.3f} {fl:6.0%} {tn:>22d}")

# Correction: reuse the existing corrections where option A flags the same sentence
existing = {}
for name in ("pipeline", "pipeline_v1"):
    for p in load_predictions(ROOT / f"outputs/predictions/{name}_test.jsonl"):
        if p.corrected_sentence:
            existing.setdefault((p.text_id, p.error_sentence_id), p.corrected_sentence)
with_corr, missing = [], 0
for p in preds:
    if p.error_flag:
        c = existing.get((p.text_id, p.error_sentence_id))
        missing += c is None
        with_corr.append(Prediction(p.text_id, 1, p.error_sentence_id, corrected_sentence=c))
    else:
        with_corr.append(p)
print(f"\nCorrección: {sum(x.error_flag for x in preds)} textos marcados, {missing} sin corrección guardada (cuentan como corrección vacía)")
for label, p in (("Pipeline final actual", load_predictions(ROOT / "outputs/predictions/pipeline_test.jsonl")),
                 ("Opción A + corrector actual", with_corr)):
    c = compute_correction_scores(test_df, p, BERTSCORE)
    extra = f" BERTScore oficial={c['bertscore_official']:.3f}" if BERTSCORE else ""
    print(f"  {label:30s} ROUGE-1 oficial={c['rouge1_official']:.3f}{extra}")
