import polars as pl

from src.decision import expected_f, threshold


def frame(rows):
    return pl.DataFrame(rows, schema={"s1": pl.UInt32, "m": pl.UInt32, "src": pl.UInt8, "p": pl.Float32}, orient="row")


def test_one_owner_and_threshold():
    pred = frame([(1, 10, 2, 0.9), (2, 10, 2, 0.6), (2, 11, 3, 0.8)])
    got = threshold(pred, 0.5).sort("s1")
    assert got.select("s1", "m").rows() == [(1, 10), (2, 11)]  # record 10 goes to its best S1 only


def test_expected_f_prefers_empty_for_weak_singletons():
    pred = frame([(1, 10, 2, 0.15), (1, 11, 3, 0.10)])
    assert expected_f(pred).height == 0


def test_expected_f_takes_confident_prefix_and_drops_weak_tail():
    pred = frame([(1, 10, 2, 0.99), (1, 11, 3, 0.97), (1, 12, 2, 0.20)])
    assert sorted(expected_f(pred)["m"].to_list()) == [10, 11]


def test_expected_f_keeps_single_strong_match():
    pred = frame([(1, 10, 2, 0.95)])
    assert expected_f(pred)["m"].to_list() == [10]
