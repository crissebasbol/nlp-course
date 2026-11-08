"""Option A on full val and test (experiment only, project code untouched).

BERT picks sentence k (argmax p_bert); Qwen answers "is sentence k correct? Yes/No" and we keep
P(Yes) from the logprobs. Results go to the scratchpad; resumable.
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

OUT_DIR = Path(sys.argv[1])
API = "http://localhost:8001/v1/chat/completions"
SYSTEM = "You are an expert physician reviewing clinical case descriptions."
client = httpx.Client(timeout=300)


def p_yes(choice) -> float:
    yes = no = 0.0
    for t in choice["logprobs"]["content"][0]["top_logprobs"]:
        tok = t["token"].strip().lower()
        if tok == "yes":
            yes += math.exp(t["logprob"])
        elif tok == "no":
            no += math.exp(t["logprob"])
    return yes / (yes + no) if yes + no > 0 else 0.5


for split in ("val", "test"):
    df = load_split(ROOT / "outputs" / "splits", split)
    bert = {r["text_id"]: r["scores"] for r in map(json.loads, open(ROOT / "outputs" / "scores" / f"bert_{split}.jsonl"))}
    out = OUT_DIR / f"v1_{split}.jsonl"
    done = {json.loads(line)["text_id"] for line in open(out)} if out.exists() else set()
    t_split = time.time()
    for n, r in enumerate(df.itertuples()):
        if r.text_id in done:
            continue
        sents = list(r.sentences)
        k = int(np.argmax(bert[r.text_id]))
        t0 = time.time()
        resp = client.post(API, json={
            "model": "qwen", "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 10,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Clinical case (one sentence per line, with its ID):\n{number_sentences(sents)}\n\n"
                                            f"Consider sentence {k}: \"{sents[k]}\"\n"
                                            "Is this sentence medically correct and consistent with the rest of the case "
                                            "(diagnosis, cause, management, treatment)? Answer Yes or No."},
            ],
        })
        resp.raise_for_status()
        rec = {"text_id": r.text_id, "k": k, "p_correct": round(p_yes(resp.json()["choices"][0]), 5),
               "prompt_tokens": resp.json()["usage"]["prompt_tokens"], "seconds": round(time.time() - t0, 2)}
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        if n % 50 == 0:
            print(f"{split}: {n}/{len(df)} ({time.time() - t_split:.0f}s)", flush=True)
    print(f"{split}: listo en {time.time() - t_split:.0f}s", flush=True)
print("FIN")
