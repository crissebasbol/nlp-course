"""Step 3c: tune w and the threshold on val, then build the detection predictions (A and B).

Systems written to predictions/:
  bert            BERT alone (w = 1, threshold tuned on val)
  dspy, dspy_zeroshot   the LLM's own decision (has_error, sentence)
  combined        w * p_bert + (1 - w) * s_dspy with the best (w, threshold) of val
  hybrid          the combination decides the flag (A), BERT alone picks the sentence (B);
                  w and the threshold are tuned again for this rule. Used by the final pipeline.
"""

import json

from _common import parse_args

from medec.combine import combine_scores, decide_error_flag, locator_scores, select_error_sentence, tune_weight_and_threshold
from medec.config import output_path
from medec.data import load_split
from medec.schemas import Prediction, load_jsonl, save_predictions


def load_scores(cfg, name, split):
    path = output_path(cfg, "scores", f"{name}_{split}.jsonl")
    return {r["text_id"]: r for r in load_jsonl(path)} if path.exists() else {}


def records_for(cfg, split):
    df = load_split(output_path(cfg, "splits", "x").parent, split)
    bert, dspy = load_scores(cfg, "bert", split), load_scores(cfg, "dspy", split)
    recs = []
    for r in df.itertuples():
        recs.append({
            "text_id": r.text_id, "error_flag": int(r.error_flag), "error_sentence_id": int(r.error_sentence_id),
            "p_bert": bert[r.text_id]["scores"] if r.text_id in bert else None,
            "s_dspy": dspy[r.text_id]["scores"] if r.text_id in dspy else None,
        })
    return recs, bert, dspy


def decide(records, w, threshold, sentence_from="combined"):
    preds = []
    for rec in records:
        if rec["p_bert"] is None and rec["s_dspy"] is None:
            continue
        scores = combine_scores(rec["p_bert"], rec["s_dspy"], w)
        flag = decide_error_flag(scores, threshold)
        sid = select_error_sentence(locator_scores(scores, rec["p_bert"], sentence_from), flag)
        preds.append(Prediction(rec["text_id"], flag, sid, scores=[round(s, 4) for s in scores]))
    return preds


def llm_decision(scores):
    return [Prediction(tid, int(r["has_error"]), r["error_sentence_id"], scores=r["scores"]) for tid, r in scores.items()]


def main() -> None:
    args, cfg = parse_args(__doc__, splits=True)
    val_records, _, _ = records_for(cfg, "val")
    complete = [r for r in val_records if r["p_bert"] is not None and r["s_dspy"] is not None]
    bert_only = [{**r, "s_dspy": None} for r in val_records if r["p_bert"] is not None]

    tuning = {}
    if complete:
        tuning["combined"] = tune_weight_and_threshold(complete)
        tuning["hybrid"] = tune_weight_and_threshold(complete, sentence_from="bert")
    if bert_only:
        tuning["bert"] = tune_weight_and_threshold(bert_only, weights=[1.0])
    artifacts = {k: v["best"] for k, v in tuning.items()}
    if "combined" in tuning:
        artifacts["combined_by_weight"] = tuning["combined"]["best_by_weight"]
    path = output_path(cfg, "artifacts", "combine.json")
    path.write_text(json.dumps(artifacts, indent=2))
    for k, v in artifacts.items():
        if k != "combined_by_weight":
            print(f"{k}: w={v['w']} umbral={v['threshold']} | val A={v['flag_acc']:.3f} B={v['sentence_acc']:.3f}")

    for split in args.splits:
        records, _, _ = records_for(cfg, split)
        if "bert" in artifacts:
            b = artifacts["bert"]
            save_predictions(decide([{**r, "s_dspy": None} for r in records if r["p_bert"]], 1.0, b["threshold"]),
                             output_path(cfg, "predictions", f"bert_{split}.jsonl"))
        if "combined" in artifacts:
            c = artifacts["combined"]
            both = [r for r in records if r["p_bert"] is not None and r["s_dspy"] is not None]
            save_predictions(decide(both, c["w"], c["threshold"]), output_path(cfg, "predictions", f"combined_{split}.jsonl"))
        if "hybrid" in artifacts:
            h = artifacts["hybrid"]
            save_predictions(decide(both, h["w"], h["threshold"], sentence_from="bert"),
                             output_path(cfg, "predictions", f"hybrid_{split}.jsonl"))
        for name in ("dspy", "dspy_zeroshot"):
            scores = load_scores(cfg, name, split)
            if scores:
                save_predictions(llm_decision(scores), output_path(cfg, "predictions", f"{name}_{split}.jsonl"))
        print(f"{split}: predicciones de detección guardadas")


if __name__ == "__main__":
    main()
