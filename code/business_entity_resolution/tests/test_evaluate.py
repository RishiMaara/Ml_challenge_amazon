"""Tests for src.evaluate: the metric must match the README definition exactly."""
import polars as pl
import pytest

from src.evaluate import blocking_report, entity_scores, evaluate, read_id_list_tsv, write_id_list_tsv


def pairs(d):
    rows = [(k, v) for k, vs in d.items() for v in vs]
    return pl.DataFrame(rows, schema={"s1_id": pl.Utf8, "match_id": pl.Utf8}, orient="row")


def ids(*xs):
    return pl.DataFrame({"s1_id": list(xs)})


def test_readme_example():
    pred = pairs({"S1-00001": ["S2-00047", "S2-00193", "S3-00812"]})
    truth = pairs({"S1-00001": ["S2-00047", "S3-00812"]})
    assert evaluate(pred, truth, ids("S1-00001"))["f05"] == pytest.approx(0.714, abs=1e-3)


def test_singletons_and_empty_predictions():
    truth = pairs({"A": ["S2-1"]})
    s1 = ids("A", "B")  # B is a singleton (no truth rows)
    sc = entity_scores(pairs({}), truth, s1).sort("s1_id")
    assert sc["f"].to_list() == [0.0, 1.0]                      # A missed -> 0, B correct empty -> 1
    sc = entity_scores(pairs({"B": ["S3-9"]}), truth, s1).sort("s1_id")
    assert sc["f"].to_list() == [0.0, 0.0]                      # B false merge -> 0
    assert evaluate(pairs({"A": ["S2-1"]}), truth, s1)["f05"] == 1.0


def test_precision_weighted_more_than_recall():
    truth = pairs({"A": ["1", "2", "3", "4"]})
    half_recall = evaluate(pairs({"A": ["1", "2"]}), truth, ids("A"))["f05"]          # P=1, R=.5
    half_precision = evaluate(pairs({"A": ["1", "2", "3", "4", "x", "y", "z", "w"]}), truth, ids("A"))["f05"]  # P=.5, R=1
    assert half_recall == pytest.approx(0.8333, abs=1e-3) and half_precision == pytest.approx(0.5556, abs=1e-3)


def test_duplicates_and_out_of_set_ids_ignored():
    truth = pairs({"A": ["1"]})
    pred = pairs({"A": ["1", "1"], "Z": ["9"]})
    assert evaluate(pred, truth, ids("A"))["f05"] == 1.0


def test_blocking_report():
    truth = pairs({"A": ["1", "2"], "B": ["3"]})
    cands = pairs({"A": ["1", "2", "x"], "B": ["y"], "C": ["z"]})
    rep = blocking_report(cands, truth, ids("A", "B", "C"))
    assert rep["pair_recall"] == pytest.approx(2 / 3)
    assert rep["entity_full_recall"] == pytest.approx(0.5)
    # oracle: A perfect (1), B nothing recoverable (0), C singleton predicted empty (1)
    assert rep["oracle_f05"] == pytest.approx(2 / 3)


def test_tsv_roundtrip(tmp_path):
    p = pairs({"S1-1": ["S2-5", "S3-7"], "S1-2": []})
    f = tmp_path / "m.tsv"
    write_id_list_tsv(p, ["S1-1", "S1-2", "S1-3"], f)
    text = f.read_text(encoding="utf-8").splitlines()
    assert text == ["source1_entity_id\tmatched_entity_ids", "S1-1\tS2-5,S3-7", "S1-2\t", "S1-3\t"]
    back = read_id_list_tsv(f).sort("match_id")
    assert back.rows() == [("S1-1", "S2-5"), ("S1-1", "S3-7")]
