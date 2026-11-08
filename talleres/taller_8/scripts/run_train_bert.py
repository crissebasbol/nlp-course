"""Step 3a (training): fine-tune BiomedBERT on (sentence, text) pairs. Run on a GPU (Mac MPS, CUDA)."""

from _common import parse_args

from medec.config import output_path
from medec.data import load_split
from medec.detector_bert import get_device, train_detector
from medec.pairs import build_pairs_frame


def main() -> None:
    args, cfg = parse_args(__doc__)
    splits_dir = output_path(cfg, "splits", "x").parent
    train, val = load_split(splits_dir, "train"), load_split(splits_dir, "val")
    if args.limit:
        train, val = train.head(args.limit), val.head(max(args.limit // 4, 4))
    train_pairs, val_pairs = build_pairs_frame(train), build_pairs_frame(val)
    print(f"Dispositivo: {get_device()} | pares train: {len(train_pairs)} | pares val: {len(val_pairs)}")
    print(f"Proporción de oraciones con error (train): {train_pairs['label'].mean():.3f}")
    final = train_detector(
        train_pairs, val_pairs, cfg["models"]["detector"], output_path(cfg, "models", "bert", "x").parent, cfg["bert"], cfg["seed"]
    )
    print("Modelo guardado en", final)


if __name__ == "__main__":
    main()
