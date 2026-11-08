"""Step 5: compare the current BERT with the MedQA-pretrained ones.

For each model: AUC for A (max p_bert), B on error texts (argmax), and A/B with a threshold tuned on val
(same objective A + B as the pipeline). Test is only reported with the val threshold.
Usage: python 3_analyze.py <exp_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[4]  # taller_8/
sys.path.insert(0, str(ROOT / "src"))
from medec.combine import decide_error_flag, select_error_sentence  # noqa: E402
from medec.data import load_split  # noqa: E402
from medec.evaluation import detection_metrics  # noqa: E402
from medec.schemas import Prediction  # noqa: E402

EXP = Path(sys.argv[1])
MODELS = {
    "BERT actual (solo MEDEC)": lambda s: ROOT / "outputs" / "scores" / f"bert_{s}.jsonl",
    "Etapa 1: solo MedQA filtrado": lambda s: EXP / f"bert_stage1_{s}.jsonl",
    "Etapa 2: MedQA filtrado + MEDEC": lambda s: EXP / f"bert_stage2_{s}.jsonl",
}
data = {s: load_split(ROOT / "outputs" / "splits", s) for s in ("val", "test")}


def load(path):
    return {r["text_id"]: r["scores"] for r in map(json.loads, open(path))} if path.exists() else None


def predict(df, sc, t):
    out = []
    for r in df.itertuples():
        flag = decide_error_flag(sc[r.text_id], t)
        out.append(Prediction(r.text_id, flag, select_error_sentence(sc[r.text_id], flag)))
    return out


grid = [round(x, 3) for x in np.linspace(0.02, 0.98, 49)]
print(f"{'Modelo':34s} {'AUC val':>8s} {'AUC test':>9s} {'B err val':>10s} {'umbral':>7s} {'A test':>7s} {'B test':>7s} {'%marc':>6s} {'correctos':>10s}")
for name, path in MODELS.items():
    sc = {s: load(path(s)) for s in data}
    if sc["val"] is None or sc["test"] is None:
        print(f"{name:34s} (sin puntajes todavía)")
        continue
    aucs, b_err = {}, None
    for s, df in data.items():
        mx = [max(sc[s][t]) for t in df.text_id]
        aucs[s] = roc_auc_score(df.error_flag, mx)
        if s == "val":
            errs = df[df.error_flag == 1]
            b_err = np.mean([int(np.argmax(sc[s][r.text_id])) == r.error_sentence_id for r in errs.itertuples()])
    best = max(grid, key=lambda t: sum(detection_metrics(data["val"], predict(data["val"], sc["val"], t))[k]
                                       for k in ("flag_acc", "sentence_acc")))
    p = predict(data["test"], sc["test"], best)
    m = detection_metrics(data["test"], p)
    y = dict(zip(data["test"].text_id, data["test"].error_flag))
    tn = sum(1 for x in p if not x.error_flag and y[x.text_id] == 0)
    print(f"{name:34s} {aucs['val']:8.3f} {aucs['test']:9.3f} {b_err:10.3f} {best:7.3f} {m['flag_acc']:7.3f} "
          f"{m['sentence_acc']:7.3f} {np.mean([x.error_flag for x in p]):6.0%} {tn:>6d}/144")
print("\nReferencias en test: pipeline final A 0.566 / B 0.547; Claude A 0.948 / B 0.942")
