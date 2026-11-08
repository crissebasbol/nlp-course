"""Analysis of prueba 3. AUC with bootstrap CI; accuracy with a threshold chosen by 5-fold CV."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[4]  # taller_8/
sys.path.insert(0, str(ROOT / "src"))
from medec.quality import compare_with_reference  # noqa: E402

recs = [json.loads(line) for line in open(sys.argv[1])]
y = np.array([r["error_flag"] for r in recs])
print(f"{len(recs)} textos ({y.sum()} con error, {(1 - y).sum()} sin error)")
print(f"BERT eligió la oración del error en {sum(r['k_is_gold'] for r in recs)}/{y.sum()} textos con error")

# Embedding similarity between the blind answer and the original sentence
from sentence_transformers import SentenceTransformer  # noqa: E402

enc = SentenceTransformer("NeuML/pubmedbert-base-embeddings")
a = enc.encode([r["target"] for r in recs], normalize_embeddings=True)
b = enc.encode([r["blind"] for r in recs], normalize_embeddings=True)
cos = (a * b).sum(1)

signals = {
    "V1 directa: P(No es correcta)": np.array([1 - r["p_correct_direct"] for r in recs]),
    "V2 ciega + juez: P(distinta)": np.array([1 - r["p_same_blind"] for r in recs]),
    "V2b ciega + embeddings: 1 - coseno": 1 - cos,
}
signals["V1 + V2 (promedio)"] = (signals["V1 directa: P(No es correcta)"] + signals["V2 ciega + juez: P(distinta)"]) / 2

rng = np.random.default_rng(0)


def auc_ci(score):
    vals = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            vals.append(roc_auc_score(y[idx], score[idx]))
    return np.percentile(vals, [2.5, 97.5])


def cv_accuracy(score):
    """Threshold picked on 4/5 of the sample, accuracy measured on the other 1/5."""
    hits = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(score, y):
        ths = np.unique(score[tr])
        best = max(ths, key=lambda t: ((score[tr] >= t) == y[tr]).mean())
        hits[te] = (score[te] >= best) == y[te]
    return hits.mean()


print(f"\n{'Señal':40s} {'AUC':>6s}  {'IC 95 %':>13s}  {'Acc A (CV)':>10s}")
for name, s in signals.items():
    lo, hi = auc_ci(s)
    print(f"{name:40s} {roc_auc_score(y, s):6.3f}  [{lo:.2f}, {hi:.2f}]  {cv_accuracy(s):10.3f}")
ref_label = "Referencia: siempre 'hay error'"
print(f"{ref_label:40s} {'0.500':>6s}  {'':13s}  {y.mean():10.3f}")

# Simple fixed rules (no tuning)
for name, pred in (("V1 regla fija: P(correcta) < 0.5", np.array([r["p_correct_direct"] < 0.5 for r in recs])),
                   ("V2 regla fija: P(igual) < 0.5", np.array([r["p_same_blind"] < 0.5 for r in recs]))):
    tp, tn = (pred & (y == 1)).sum(), (~pred & (y == 0)).sum()
    print(f"{name}: A = {(pred == y).mean():.3f} | detecta {tp}/{y.sum()} con error | reconoce {tn}/{(1 - y).sum()} correctos | marca {pred.mean():.0%}")

# Correction preview: does the blind answer contain the key change of the reference?
pipe = {r["text_id"]: r for r in map(json.loads, open(ROOT / "outputs" / "predictions" / "pipeline_val.jsonl"))}
blind_v, pipe_v = [], []
for r in recs:
    if not r["k_is_gold"]:
        continue
    blind_v.append(compare_with_reference(r["target"], r["blind"], r["gold_correction"])["verdict"])
    p = pipe.get(r["text_id"])
    if p and p.get("corrected_sentence") and p["error_sentence_id"] == r["gold_sid"]:
        pipe_v.append(compare_with_reference(r["target"], p["corrected_sentence"], r["gold_correction"])["verdict"])
ok = lambda v: sum(x in ("exact", "key_terms") for x in v)  # noqa: E731
print(f"\nCorrección (textos con error donde BERT acertó la oración):")
print(f"  respuesta ciega contiene el cambio clave: {ok(blind_v)}/{len(blind_v)} ({ok(blind_v) / max(len(blind_v), 1):.0%})")
print(f"  corrector actual contiene el cambio clave: {ok(pipe_v)}/{len(pipe_v)} ({ok(pipe_v) / max(len(pipe_v), 1):.0%})")

print("\nEjemplos:")
for r in recs[:6]:
    print(f"- flag={r['error_flag']} | original: {r['target'][:90]}\n  ciega   : {r['blind'][:90]}\n  P(correcta)={r['p_correct_direct']:.2f} P(igual)={r['p_same_blind']:.2f}"
          + (f"\n  referencia: {r['gold_correction'][:90]}" if r["gold_correction"] else ""))
