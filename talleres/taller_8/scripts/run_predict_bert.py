"""Step 3a (inference): p_bert for each sentence of the val/test texts."""

from _common import parse_args

from medec.config import output_path
from medec.data import load_split
from medec.detector_bert import BertScorer, predict_split_scores
from medec.schemas import save_jsonl


def main() -> None:
    args, cfg = parse_args(__doc__, splits=True)
    splits_dir = output_path(cfg, "splits", "x").parent
    scorer = BertScorer(output_path(cfg, "models", "bert", "final", "x").parent, cfg["bert"]["max_length"])
    for split in args.splits:
        df = load_split(splits_dir, split)
        if args.limit:
            df = df.head(args.limit)
        records = predict_split_scores(scorer, df, cfg["bert"]["eval_batch_size"])
        path = save_jsonl(records, output_path(cfg, "scores", f"bert_{split}.jsonl"))
        print(f"{split}: {len(records)} textos -> {path}")


if __name__ == "__main__":
    main()
