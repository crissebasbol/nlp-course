"""Step 3b (alternative): an LLM verifies the sentence picked by BERT (subtask A) and proposes a fix.

The LLM is chosen with --verifier, one of the names under `verifier.lms` in the config
(e.g. haiku, sonnet, qwen). Claude models cost money: try first with --limit.

Writes:
  scores/verifier_<name>_<split>.jsonl        raw answers (resumable)
  predictions/verified_<name>_<split>.jsonl   A from the LLM, B from BERT, C from the same LLM
Then `run_correct.py --system verified_<name> --output-name verified_<name>_qwen` corrects with Qwen instead.

--compile: BootstrapFewShot on train texts (alternating correct / wrong sentences), saved in
models/verifier_<name>.json and reported under the name <name>_compiled.
"""

import json

import numpy as np
import pandas as pd

from _common import parse_args

from medec.config import output_path
from medec.data import load_split
from medec.schemas import Prediction, append_jsonl, load_jsonl, save_predictions
from medec.verifier import (
    balanced_order, build_verifier, compile_verifier, make_verifier_lm, to_verifier_example, verify_split,
)


def extra(parser):
    parser.add_argument("--verifier", default=None, help="Name under verifier.lms (default: verifier.default)")
    parser.add_argument("--compile", action="store_true", help="Compile with BootstrapFewShot on train (few demos)")


def compile_program(program, cfg, name):
    """Demos from train texts; sentence k is BERT's pick, as at inference time."""
    from medec.detector_bert import BertScorer, predict_split_scores, resolve_bert_dir

    # BootstrapFewShot and load() return the program without its LM: it is set again afterwards
    lm = program.predictors()[0].lm
    path = output_path(cfg, "models", f"verifier_{name}.json")
    if path.exists():
        program.load(str(path))
        program.set_lm(lm)
        print(f"Programa compilado cargado de {path}")
        return program
    vcfg = cfg["verifier"]
    train = load_split(output_path(cfg, "splits", "x").parent, "train")
    n = vcfg.get("compile_train_size", 60) // 2
    train = pd.concat([train[train.error_flag == f].sample(n, random_state=cfg["seed"]) for f in (0, 1)])
    scorer = BertScorer(resolve_bert_dir(cfg), cfg["bert"]["max_length"])
    bert = {r["text_id"]: r["scores"] for r in predict_split_scores(scorer, train, cfg["bert"]["eval_batch_size"])}
    examples = [to_verifier_example(list(r.sentences), int(np.argmax(bert[r.text_id])), r.error_flag, r.error_sentence_id,
                                    r.corrected_sentence) for r in train.itertuples()]
    program = compile_verifier(program, balanced_order(examples), vcfg.get("max_demos", 3))
    program.set_lm(lm)
    program.save(str(path))
    demos = program.predictors()[0].demos
    print(f"Compilado con {len(demos)} demos (is_correct: {[d.get('is_correct') for d in demos]}) -> {path}")
    return program


def main() -> None:
    args, cfg = parse_args(__doc__, splits=True, extra=extra)
    vcfg = cfg["verifier"]
    base = args.verifier or vcfg["default"]
    model = vcfg["lms"][base]
    program = build_verifier(make_verifier_lm(model, vcfg, cfg), vcfg.get("chain_of_thought", False) if model == "local" else False)
    name = f"{base}_compiled" if args.compile else base
    if args.compile:
        program = compile_program(program, cfg, base)
    print(f"Verificador: {name} ({model})")

    splits_dir = output_path(cfg, "splits", "x").parent
    for split in args.splits:
        df = load_split(splits_dir, split)
        if args.limit:
            df = df.head(args.limit)
        bert = {r["text_id"]: r["scores"] for r in load_jsonl(output_path(cfg, "scores", f"bert_{split}.jsonl"))}
        raw_path = output_path(cfg, "scores", f"verifier_{name}_{split}.jsonl")
        done = {r["text_id"] for r in load_jsonl(raw_path)} if raw_path.exists() else set()
        items = [{"text_id": r.text_id, "sentences": list(r.sentences), "k": int(np.argmax(bert[r.text_id]))}
                 for r in df.itertuples() if r.text_id not in done]
        verify_split(program, items, vcfg.get("num_threads", 8), on_result=lambda res: append_jsonl(res, raw_path))

        wanted = set(df.text_id)
        results = [r for r in load_jsonl(raw_path) if r["text_id"] in wanted]
        preds = []
        for r in results:
            # A failed answer falls back to BERT's behaviour (flag the text), and is counted below
            flag = int(r["is_correct"] is not True)
            correction = r["corrected_sentence"] or None if flag else None
            preds.append(Prediction(r["text_id"], flag, r["k"], correction, extra={"failed": r["failed"]}))
        suffix = f"_limit{args.limit}" if args.limit else ""
        save_predictions(preds, output_path(cfg, "predictions", f"verified_{name}{suffix}_{split}.jsonl"))
        failed = sum(r["failed"] for r in results)
        flagged = sum(p.error_flag for p in preds)
        secs = np.mean([r["seconds"] for r in results]) if results else 0
        print(f"{split}: {len(results)} textos | {failed} fallidos | marcados con error {flagged} ({flagged / max(len(preds), 1):.0%}) "
              f"| {secs:.1f} s por llamada")
        if args.limit:
            gold = dict(zip(df.text_id, df.error_flag))
            hits = sum(p.error_flag == gold[p.text_id] for p in preds)
            print(f"  muestra: {hits}/{len(preds)} aciertos en A (orientativo)")
            for r in results[:3]:
                print("  ", json.dumps({k: r[k] for k in ("text_id", "k", "is_correct", "corrected_sentence", "seconds")}, ensure_ascii=False)[:300])


if __name__ == "__main__":
    main()
