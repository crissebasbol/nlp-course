"""Step 1: find every MedQA question used to build any MEDEC text (train, val or test) and remove it.

Embeddings do not work here: all clinical vignettes look alike (the 2nd most similar question already
has cosine ~0.87). The MEDEC texts are rewrites, so they keep exact words (ages, lab values, phrases):
TF-IDF over word 1-3 grams separates them clearly (source match > 0.6, unrelated questions < 0.2).
Any MedQA question with similarity >= CUTOFF to any MEDEC text is removed.
Usage: python 1_filter_medqa.py <exp_dir> [cutoff=0.2]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[4]  # taller_8/
sys.path.insert(0, str(ROOT / "src"))
from medec.data import load_split  # noqa: E402
from medec.medqa_experiment import load_medqa  # noqa: E402

EXP = Path(sys.argv[1])
CUTOFF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
EXP.mkdir(parents=True, exist_ok=True)

medqa = pd.concat([load_medqa(s).assign(medqa_split=s) for s in ("train", "test")], ignore_index=True)
ms = pd.concat([load_split(ROOT / "outputs/splits", s).assign(split=s) for s in ("train", "val", "test")], ignore_index=True)
ms_text = [" ".join(s) for s in ms.sentences]

vec = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, token_pattern=r"(?u)\b[\w.]+\b")
vec.fit(list(medqa.question) + ms_text)
S = (vec.transform(ms_text) @ vec.transform(medqa.question).T).toarray()  # (MEDEC texts, MedQA questions)

top = S.max(1)
second = np.sort(S, 1)[:, -2]
print(f"MEDEC: {len(ms)} textos | MedQA: {len(medqa)} preguntas")
print("Mejor coincidencia de cada texto MEDEC, histograma [0, .1, .2, .4, .6, 1]:",
      np.histogram(top, bins=[0, .1, .2, .4, .6, 1])[0].tolist())
print(f"Segunda coincidencia: mediana {np.median(second):.3f}")
print(f"Textos MEDEC con su pregunta fuente en MedQA (>= {CUTOFF}): {(top >= CUTOFF).sum()} / {len(ms)}")
for s in ("train", "val", "test"):
    m = (ms.split == s).values
    print(f"  {s}: {(top[m] >= CUTOFF).sum()} / {m.sum()}")

mx = S.max(0)
remove = np.where(mx >= CUTOFF)[0]
keep = medqa.drop(index=remove)
print(f"\nUmbral {CUTOFF}: se excluyen {len(remove)} preguntas MedQA; quedan {len(keep)}.")

print("\nZona dudosa (mejor coincidencia entre 0.2 y 0.6), para revisar a ojo:")
for i in np.where((top >= 0.2) & (top < 0.6))[0][:5]:
    j = S[i].argmax()
    print(f"- {top[i]:.2f} MEDEC: {ms_text[i][:100]}\n       MedQA: {medqa.question[j][:100]}")

keep[["question", "options", "answer", "medqa_split"]].to_json(EXP / "medqa_filtered.jsonl", orient="records", lines=True)
json.dump({"method": "tfidf word 1-3 grams", "cutoff": CUTOFF, "removed": int(len(remove)), "kept": int(len(keep)),
           "medec_with_source": int((top >= CUTOFF).sum()), "medec_total": int(len(ms))},
          open(EXP / "filter_report.json", "w"), indent=2)
print("->", EXP / "medqa_filtered.jsonl")
