"""Steps 2-4: MedQA examples -> stage 1 (MedQA) -> stage 2 (MEDEC) -> scores on val and test.

Reuses medec.detector_bert.train_detector unchanged (stage 2 starts from the stage-1 checkpoint).
Models go to outputs/models/bert/medqa_stage{1,2} (git-ignored); the current BERT is not touched.
Each stage is skipped if its output exists, so the script can be re-run.
Usage: python 2_train_and_score.py <exp_dir>
"""
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]  # taller_8/
sys.path.insert(0, str(ROOT / "src"))
from medec.config import load_config  # noqa: E402
from medec.data import load_split  # noqa: E402
from medec.detector_bert import BertScorer, get_device, predict_split_scores, train_detector  # noqa: E402
from medec.pairs import build_pairs_frame  # noqa: E402

EXP = Path(sys.argv[1])
cfg = load_config()
MODELS = ROOT / "outputs" / "models" / "bert"
STAGE1, STAGE2 = MODELS / "medqa_stage1", MODELS / "medqa_stage2"
random.seed(42)


def medqa_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Per question: (question sentence + correct option) -> 0, (question sentence + one wrong option) -> 1."""
    rows = []
    for q in df.itertuples():
        parts = re.split(r"(?<=[.!?])\s+", q.question.strip())
        vignette, ask = " ".join(parts[:-1]), parts[-1]
        wrong = [o for o in q.options.values() if o != q.answer]
        for option, label in ((q.answer, 0), (random.choice(wrong), 1)):
            sentence = f"{ask} {option}."
            rows.append({"sentence": sentence, "context": f"{vignette} {sentence}", "label": label})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stage 1: MedQA
if not (STAGE1 / "final").exists():
    medqa = pd.read_json(EXP / "medqa_filtered.jsonl", lines=True)
    medqa = medqa.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val = int(len(medqa) * 0.05)
    tr, va = medqa_pairs(medqa.iloc[n_val:]), medqa_pairs(medqa.iloc[:n_val])
    print(f"Etapa 1 (MedQA filtrado): {len(tr)} ejemplos de train, {len(va)} de validación | {get_device()}", flush=True)
    params = {**cfg["bert"], "epochs": 1, "batch_size": 8, "eval_batch_size": 16}  # long vignettes: smaller batches for MPS memory
    train_detector(tr, va, cfg["models"]["detector"], STAGE1, params, cfg["seed"])
    print("Etapa 1 lista", flush=True)

# ---------------------------------------------------------------- stage 2: MEDEC (same as the current BERT)
splits = ROOT / "outputs" / "splits"
if not (STAGE2 / "final").exists():
    train_pairs = build_pairs_frame(load_split(splits, "train"))
    val_pairs = build_pairs_frame(load_split(splits, "val"))
    print(f"Etapa 2 (MEDEC): {len(train_pairs)} pares de train", flush=True)
    train_detector(train_pairs, val_pairs, str(STAGE1 / "final"), STAGE2, cfg["bert"], cfg["seed"])
    print("Etapa 2 lista", flush=True)

# ---------------------------------------------------------------- scores (stage 1 alone and stage 2)
for name, model_dir in (("stage1", STAGE1 / "final"), ("stage2", STAGE2 / "final")):
    scorer = BertScorer(model_dir, cfg["bert"]["max_length"])
    for split in ("val", "test"):
        out = EXP / f"bert_{name}_{split}.jsonl"
        if out.exists():
            continue
        res = predict_split_scores(scorer, load_split(splits, split), cfg["bert"]["eval_batch_size"])
        with open(out, "w") as fh:
            for r in res:
                fh.write(json.dumps(r) + "\n")
        print(f"puntajes {name} {split} -> {out}", flush=True)
print("FIN")
