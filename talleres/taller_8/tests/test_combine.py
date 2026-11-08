import pytest

from medec.combine import combine_scores, decide_error_flag, select_error_sentence, tune_weight_and_threshold


def test_combine_example_from_the_design():
    # 0.7 * 0.85 + 0.3 * 0.80 = 0.835
    assert combine_scores([0.1, 0.85], [0.0, 0.8], 0.7)[1] == pytest.approx(0.835)


def test_combine_with_one_detector():
    assert combine_scores([0.2, 0.9], None, 0.7) == [0.2, 0.9]
    assert combine_scores(None, [0.0, 0.6], 0.7) == [0.0, 0.6]


def test_decision():
    assert decide_error_flag([0.1, 0.6], 0.5) == 1
    assert select_error_sentence([0.1, 0.6], 1) == 1
    assert decide_error_flag([0.1, 0.4], 0.5) == 0
    assert select_error_sentence([0.1, 0.4], 0) == -1


def test_tuning_finds_perfect_setting():
    records = [
        {"p_bert": [0.1, 0.9], "s_dspy": [0.0, 0.0], "error_flag": 1, "error_sentence_id": 1},
        {"p_bert": [0.2, 0.1], "s_dspy": [0.0, 0.0], "error_flag": 0, "error_sentence_id": -1},
    ]
    best = tune_weight_and_threshold(records)["best"]
    assert best["flag_acc"] == 1 and best["sentence_acc"] == 1
