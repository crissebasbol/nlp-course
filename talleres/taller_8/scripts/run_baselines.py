"""Baseline: always "error in the last sentence".

The majority class in train is "has error" (56%), so the majority-class baseline
and the last-sentence baseline give the same predictions; only one is kept.
"""

from _common import parse_args

from medec.baselines import predict_last_sentence
from medec.config import output_path
from medec.data import load_split
from medec.schemas import save_predictions


def main() -> None:
    args, cfg = parse_args(__doc__, splits=True)
    splits_dir = output_path(cfg, "splits", "x").parent
    for split in args.splits:
        df = load_split(splits_dir, split)
        save_predictions(predict_last_sentence(df), output_path(cfg, "predictions", f"baseline_last_sentence_{split}.jsonl"))
        print(f"{split}: baseline guardado")


if __name__ == "__main__":
    main()
