import pytest

from medec.evaluation import compute_correction_scores, detection_metrics
from medec.schemas import Prediction


def test_detection_metrics(toy_df):
    preds = [
        Prediction("t0", 1, 2), Prediction("t1", 0), Prediction("t2", 1, 0),
        Prediction("t3", 1, 1), Prediction("t4", 1, 1), Prediction("t5", 0),
    ]
    m = detection_metrics(toy_df, preds)
    assert m["flag_acc"] == pytest.approx(5 / 6)
    assert m["sentence_acc"] == pytest.approx(4 / 6)
    assert m["sentence_acc_on_errors"] == pytest.approx(2 / 3)


def test_missing_prediction_counts_as_no_error(toy_df):
    assert detection_metrics(toy_df, [])["flag_acc"] == pytest.approx(3 / 6)


def test_correction_official_rules(toy_df):
    preds = [Prediction("t0", 1, 2, "Right drug given."), Prediction("t1", 0), Prediction("t3", 1, 1, "x")]
    res = compute_correction_scores(toy_df, preds, bertscore_model=None)
    # t0 perfect, t1 and t5 both NA -> 1; t2, t4 missing -> 0; t3 predicted but no reference -> 0
    assert res["rouge1_official"] == pytest.approx(3 / 6)
    assert res["rouge1_pairs"] == pytest.approx(1.0)
