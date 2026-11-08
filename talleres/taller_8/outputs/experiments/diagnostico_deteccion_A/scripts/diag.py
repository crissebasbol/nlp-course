"""Diagnostics on val (no LLM): (1) does BERT point to the answer slot in correct texts?
(2) does Qwen's confidence separate texts with and without error?"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[4]  # taller_8/
sys.path.insert(0, str(ROOT / "src"))
from medec.data import load_split  # noqa: E402

OUT = ROOT / "outputs"
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"
df = load_split(OUT / "splits", SPLIT)


def load(name):
    return {r["text_id"]: r for r in map(json.loads, open(OUT / "scores" / f"{name}_{SPLIT}.jsonl"))}


bert, dspy, dzs = load("bert"), load("dspy"), load("dspy_zeroshot")

print(f"===== {SPLIT}: {len(df)} textos, {int(df.error_flag.sum())} con error, {int((df.error_flag == 0).sum())} sin error")

# ---------------------------------------------------------------- 1. BERT and the answer slot
print("\n=== PRUEBA 1: BERT en textos sin error, frente a su pareja con error")
sizes = df.groupby("group").size()
print("textos por caso:", sizes.value_counts().sort_index().to_dict())
same_slot, diff_only_slot, n_pairs, pos_info = 0, 0, 0, []
for g, sub in df.groupby("group"):
    errs, oks = sub[sub.error_flag == 1], sub[sub.error_flag == 0]
    if len(oks) == 0 or len(errs) == 0:
        continue
    ok = oks.iloc[0]
    for _, er in errs.iterrows():
        if len(er.sentences) != len(ok.sentences):
            continue
        slot = int(er.error_sentence_id)
        diffs = [i for i, (a, b) in enumerate(zip(er.sentences, ok.sentences)) if a.strip() != b.strip()]
        n_pairs += 1
        diff_only_slot += diffs == [slot]
        pick_ok = int(np.argmax(bert[ok.text_id]["scores"]))
        pick_err = int(np.argmax(bert[er.text_id]["scores"]))
        same_slot += pick_ok == slot
        pos_info.append((slot, len(ok.sentences), pick_ok == slot, pick_err == slot))
print(f"parejas comparables (mismo número de oraciones): {n_pairs}")
print(f"  las dos versiones difieren SOLO en la oración del error: {diff_only_slot}/{n_pairs} ({diff_only_slot / n_pairs:.0%})")
print(f"  BERT en el texto CON error elige esa oración: {sum(p[3] for p in pos_info)}/{n_pairs} ({np.mean([p[3] for p in pos_info]):.0%})")
print(f"  BERT en el texto SIN error elige la misma oración (la de la respuesta): {same_slot}/{n_pairs} ({same_slot / n_pairs:.0%})")
last = [p for p in pos_info if p[0] == p[1] - 1]
print(f"  la oración clave es la última del texto en {len(last)}/{n_pairs} parejas")

# How confident is BERT in each kind of text?
mx = pd.DataFrame([{"flag": int(r.error_flag), "max": max(bert[r.text_id]["scores"]),
                    "gap": sorted(bert[r.text_id]["scores"])[-1] - sorted(bert[r.text_id]["scores"])[-2]}
                   for r in df.itertuples()])
print("\nprobabilidad máxima de BERT (mediana): con error "
      f"{mx[mx.flag == 1]['max'].median():.3f} | sin error {mx[mx.flag == 0]['max'].median():.3f}")
print(f"AUC de BERT para A (máx p_bert): {roc_auc_score(mx.flag, mx['max']):.3f}  (0,5 = azar, 1 = perfecto)")
print(f"AUC de BERT para A (diferencia 1.ª - 2.ª): {roc_auc_score(mx.flag, mx['gap']):.3f}")

# ---------------------------------------------------------------- 2. Qwen
print("\n=== PRUEBA 2: ¿la confianza de Qwen separa textos con y sin error?")
for name, sc in (("compilado", dspy), ("sin compilar", dzs)):
    rows = []
    for r in df.itertuples():
        d = sc.get(r.text_id)
        if d is None:
            continue
        conf = d.get("confidence")
        conf = float(conf) if conf is not None else np.nan
        signed = conf if d["has_error"] else 1 - conf  # P(error) implied by the answer
        rows.append({"flag": int(r.error_flag), "has_error": int(bool(d["has_error"])), "conf": conf,
                     "p_err": signed, "failed": bool(d.get("failed"))})
    q = pd.DataFrame(rows)
    tp = ((q.flag == 1) & (q.has_error == 1)).sum()
    tn = ((q.flag == 0) & (q.has_error == 0)).sum()
    print(f"\n-- Qwen {name}: {len(q)} textos, {int(q.failed.sum())} fallos de formato")
    print(f"  dice 'hay error': {q.has_error.mean():.0%} | detecta {tp}/{(q.flag == 1).sum()} con error "
          f"({tp / (q.flag == 1).sum():.0%}) | reconoce {tn}/{(q.flag == 0).sum()} correctos ({tn / (q.flag == 0).sum():.0%})")
    print("  confianza declarada, distribución:", q.conf.round(2).value_counts().head(6).to_dict())
    for f, lab in ((1, "con error"), (0, "sin error")):
        s = q[q.flag == f]
        print(f"  {lab}: confianza media {s.conf.mean():.3f} | cuando dice 'hay error' {s[s.has_error == 1].conf.mean():.3f}")
    valid = q.dropna(subset=["p_err"])
    print(f"  AUC de Qwen para A (P(error) implícita): {roc_auc_score(valid.flag, valid.p_err):.3f}")

# Combined signal, just to see if anything separates
both = [(int(r.error_flag), max(bert[r.text_id]["scores"]), float(dspy[r.text_id].get("confidence") or 0) * (1 if dspy[r.text_id]["has_error"] else -1))
        for r in df.itertuples() if r.text_id in dspy]
b = np.array(both)
print(f"\nAUC de A sumando BERT (máx) + Qwen (confianza con signo), sin ajustar: {roc_auc_score(b[:, 0], b[:, 1] + b[:, 2]):.3f}")

# ---------------------------------------------------------------- DSPy demos
if SPLIT == "val":
    print("\n=== Ejemplos que DSPy eligió para el prompt")
    prog = json.loads((OUT / "models" / "dspy_detector.json").read_text())
    for key, pred in prog.items():
        if not isinstance(pred, dict) or "demos" not in pred:
            continue
        for d in pred["demos"]:
            print(f"  has_error={d.get('has_error')} | error_sentence_id={d.get('error_sentence_id')} | "
                  f"confidence={d.get('confidence')} | con razonamiento={'reasoning' in d}")
