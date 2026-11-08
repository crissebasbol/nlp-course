"""Step 1: read the CSV, rebuild sentences and save train/val/test split by clinical case."""

from _common import parse_args

from medec.config import output_path, resolve_path
from medec.data import check_no_leakage, load_mediqa_dataset, normalize_text, save_splits, split_by_case


def main() -> None:
    args, cfg = parse_args(__doc__)
    df = load_mediqa_dataset(resolve_path(cfg["paths"]["data_csv"]))

    errors = df[df["error_flag"] == 1]
    match = errors.apply(lambda r: normalize_text(r.sentences[r.error_sentence_id]) == normalize_text(r.error_sentence), axis=1)
    print(f"Textos: {len(df)} | casos: {df['group'].nunique()} | oraciones con error alineadas: {match.mean():.2%}")

    splits = split_by_case(df, cfg["split"]["val_size"], cfg["split"]["test_size"], cfg["seed"])
    check_no_leakage(splits)
    save_splits(splits, output_path(cfg, "splits", "x").parent)
    for name, part in splits.items():
        print(f"{name:5s}: {len(part):4d} textos | {part['group'].nunique():4d} casos | con error {part['error_flag'].mean():.1%}")


if __name__ == "__main__":
    main()
