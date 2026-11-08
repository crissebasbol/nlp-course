from medec.pairs import build_sentence_pairs, compute_class_weights


def test_pairs_label_only_error_sentence(toy_df):
    pairs = build_sentence_pairs(toy_df.iloc[0])
    assert [p["label"] for p in pairs] == [0, 0, 1]
    assert all(p["context"] == "Case A first. Middle. Wrong drug given." for p in pairs)


def test_pairs_without_error_are_all_zero(toy_df):
    assert sum(p["label"] for p in build_sentence_pairs(toy_df.iloc[1])) == 0


def test_class_weights_favor_rare_class():
    w0, w1 = compute_class_weights([0] * 9 + [1])
    assert w1 > w0
