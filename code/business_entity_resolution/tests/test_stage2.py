import polars as pl

from src.stage2 import GROUP_COLS, group_features


def test_group_features_compare_to_best_other_candidates():
    part = pl.DataFrame({"s1": [1, 1, 1], "m": [10, 11, 12], "src": [2, 3, 2], "p1": [0.95, 0.40, 0.05]},
                        schema={"s1": pl.UInt32, "m": pl.UInt32, "src": pl.UInt8, "p1": pl.Float32})
    recs = pl.DataFrame({
        "m": [10, 11, 12], "src": [2, 3, 2],
        "name_core": ["acme tools", "acme tools", "zeta foods"], "name_key": ["acme tools", "acme tools", "foods zeta"],
        "name_skel": ["akm tls", "akm tls", "jt fds"], "addr_norm": ["12 main st", "", "99 oak rd"],
        "addr_house_no": ["12", "", "99"]}, schema_overrides={"m": pl.UInt32, "src": pl.UInt8})
    g = group_features(part, recs)
    assert g.columns == GROUP_COLS and g.height == 3
    # row 2 (m=11, empty address, weak p1) is a near-copy of the confident anchor m=10
    assert g["g_anchor1_p"][1] > 0.9 and g["g_name_tset"][1] == 100 and g["g_name_key_eq"][1] == 1
    # row 3 (m=12) is unlike both anchors
    assert g["g_name_tset"][2] < 50 and g["g_name_key_eq"][2] == 0
    # the best candidate's anchor is the second-best (it never compares to itself)
    assert abs(g["g_anchor1_p"][0] - 0.40) < 1e-6 and g["g_p_rank"][0] == 1
    assert g["g_n_conf"].to_list() == [0, 1, 1]
