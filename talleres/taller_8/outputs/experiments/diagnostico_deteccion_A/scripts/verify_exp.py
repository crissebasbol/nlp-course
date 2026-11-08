"""Prueba 3 (val only): can Qwen verify the sentence that BERT picks?

V1 direct : full text, "is sentence k correct? Yes/No"            -> P(No)
V2 blind  : sentence k masked, Qwen writes it, then judges if it matches the original -> P(different)
Results are appended to a jsonl so the run can be resumed.
"""
import json
import math
import sys
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[4]  # taller_8/
sys.path.insert(0, str(ROOT / "src"))
from medec.data import load_split, number_sentences  # noqa: E402

API = "http://localhost:8001/v1/chat/completions"
OUT = Path(sys.argv[1])
N_PER_CLASS = int(sys.argv[2]) if len(sys.argv) > 2 else 50

df = load_split(ROOT / "outputs" / "splits", "val")
bert = {r["text_id"]: r["scores"] for r in map(json.loads, open(ROOT / "outputs" / "scores" / "bert_val.jsonl"))}
import pandas as pd  # noqa: E402

sample = pd.concat([df[df.error_flag == f].sample(N_PER_CLASS, random_state=42) for f in (0, 1)]).sample(frac=1, random_state=42)

done = {json.loads(line)["text_id"] for line in open(OUT)} if OUT.exists() else set()
client = httpx.Client(timeout=300)


def chat(messages, max_tokens, logprobs=False):
    body = {"model": "qwen", "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if logprobs:
        body.update(logprobs=True, top_logprobs=10)
    r = client.post(API, json=body)
    r.raise_for_status()
    return r.json()["choices"][0]


def p_yes(choice) -> float:
    """P(Yes) / (P(Yes) + P(No)) from the first token's top logprobs."""
    yes = no = 0.0
    for t in choice["logprobs"]["content"][0]["top_logprobs"]:
        tok = t["token"].strip().lower()
        if tok == "yes":
            yes += math.exp(t["logprob"])
        elif tok == "no":
            no += math.exp(t["logprob"])
    return yes / (yes + no) if yes + no > 0 else 0.5


SYSTEM = "You are an expert physician reviewing clinical case descriptions."

for r in sample.itertuples():
    if r.text_id in done:
        continue
    t0 = time.time()
    sents = list(r.sentences)
    k = int(np.argmax(bert[r.text_id]))
    target = sents[k]

    # V1: direct verification of sentence k
    direct = chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Clinical case (one sentence per line, with its ID):\n{number_sentences(sents)}\n\n"
                                    f"Consider sentence {k}: \"{target}\"\n"
                                    "Is this sentence medically correct and consistent with the rest of the case "
                                    "(diagnosis, cause, management, treatment)? Answer Yes or No."},
    ], 1, logprobs=True)

    # V2: blind answer, then judge
    masked = [s if i != k else "[MISSING]" for i, s in enumerate(sents)]
    blind = chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Clinical case. One sentence was removed and replaced by [MISSING]:\n"
                                    + "\n".join(masked) + "\n\n"
                                    "The missing sentence states a diagnosis, cause, or management/treatment decision. "
                                    "Write the most likely missing sentence as one short sentence in the same style. "
                                    "Output only the sentence."},
    ], 80)["message"]["content"].strip()
    judge = chat([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Two statements about the same patient:\nA: {target}\nB: {blind}\n\n"
                                    "Do A and B state the same diagnosis, cause, or treatment? "
                                    "Synonyms and different wording count as the same. Answer Yes or No."},
    ], 1, logprobs=True)

    rec = {
        "text_id": r.text_id, "error_flag": int(r.error_flag), "gold_sid": int(r.error_sentence_id),
        "gold_correction": r.corrected_sentence if r.error_flag else None,
        "k": k, "target": target, "k_is_gold": bool(r.error_flag) and k == int(r.error_sentence_id),
        "p_correct_direct": round(p_yes(direct), 4), "blind": blind,
        "p_same_blind": round(p_yes(judge), 4), "seconds": round(time.time() - t0, 2),
    }
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"{r.text_id} flag={rec['error_flag']} P(correcta)={rec['p_correct_direct']:.2f} "
          f"P(igual)={rec['p_same_blind']:.2f} {rec['seconds']}s", flush=True)
print("FIN")
