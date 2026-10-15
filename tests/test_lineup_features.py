import numpy as np
import pandas as pd

import feature_store as fs
from tests.test_feature_store_pit import _key, _synthetic_gamelogs, _tracking

G = ["lineup"]


def _starters_from(gl):
    """Five highest-minute players per team-game start (deterministic stand-in for box-score starters)."""
    g = gl.sort_values("MIN", ascending=False).groupby(["TEAM_ID", "GAME_ID"]).head(5)
    out = g[["PLAYER_ID", "TEAM_ID", "GAME_DATE"]].copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    return out


def _build(gl, st, **kw):
    return fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False, groups=G,
                                  starters=st, **kw)


def test_lineup_values_and_role_change():
    gl = _synthetic_gamelogs()
    st = _starters_from(gl)
    df = _build(gl, st)
    s = set(zip(st["PLAYER_ID"], pd.to_datetime(st["GAME_DATE"])))
    exp = [float((p, d) in s) for p, d in zip(df["PLAYER_ID"], df["GAME_DATE"])]
    assert df["starts_tonight"].tolist() == exp
    m = df["start_rate_roll"].notna()
    assert np.allclose(df.loc[m, "role_change"], df.loc[m, "starts_tonight"] - df.loc[m, "start_rate_roll"])
    assert df["team_usual_starters_out"].dropna().between(0, 5).all()


def test_lineup_no_lookahead_and_live_parity():
    gl = _synthetic_gamelogs()
    st = _starters_from(gl)
    last = pd.to_datetime(gl["GAME_DATE"]).max()
    hist = _key(_build(gl, st))
    tonight, past = gl[pd.to_datetime(gl["GAME_DATE"]) == last], gl[pd.to_datetime(gl["GAME_DATE"]) < last]
    st_past = st[pd.to_datetime(st["GAME_DATE"]) < last]
    st_tonight = st[pd.to_datetime(st["GAME_DATE"]) == last]
    live_starters = {}
    for gid, tid in tonight[["GAME_ID", "TEAM_ID"]].drop_duplicates().itertuples(index=False):
        live_starters.setdefault(gid, set()).update(st_tonight.loc[st_tonight["TEAM_ID"] == tid, "PLAYER_ID"])
    played = set(tonight["PLAYER_ID"])
    absent = {int(t): set(gl.loc[gl["TEAM_ID"] == t, "PLAYER_ID"]) - played for t in tonight["TEAM_ID"].unique()}
    games = tonight[["GAME_ID", "TEAM_ID"]].drop_duplicates()
    opp = games.merge(games, on="GAME_ID")
    opp = opp[opp["TEAM_ID_x"] != opp["TEAM_ID_y"]].set_index(["GAME_ID", "TEAM_ID_x"])["TEAM_ID_y"]
    extra = pd.DataFrame({
        "PLAYER_ID": tonight["PLAYER_ID"].values, "PLAYER_NAME": tonight["PLAYER_NAME"].values,
        "TEAM_ID": tonight["TEAM_ID"].values,
        "OPP_TEAM_ID": [opp[(g, t)] for g, t in zip(tonight["GAME_ID"], tonight["TEAM_ID"])],
        "GAME_ID": tonight["GAME_ID"].values, "GAME_DATE": tonight["GAME_DATE"].values,
        "SEASON": tonight["SEASON"].values,
        "is_home": tonight["MATCHUP"].str.contains("vs.", regex=False).astype(int).values, "is_playoffs": 0,
    })
    live = _build(past, st_past, extra_rows=extra, absent_override=absent, live_starters=live_starters)
    live = _key(live[live["_live"]])
    cols = fs.FEATURE_GROUPS["lineup"]
    pd.testing.assert_frame_equal(hist.loc[live.index, cols], live[cols], check_dtype=False)


def test_unconfirmed_live_game_has_nan_lineup_features():
    gl = _synthetic_gamelogs(n_days=20)
    last = pd.to_datetime(gl["GAME_DATE"]).max()
    tonight, past = gl[pd.to_datetime(gl["GAME_DATE"]) == last], gl[pd.to_datetime(gl["GAME_DATE"]) < last]
    extra = tonight[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE", "SEASON"]].assign(
        OPP_TEAM_ID=tonight["TEAM_ID"].map({1: 2, 2: 1, 3: 4, 4: 3}), is_home=0, is_playoffs=0)
    live = _build(past, _starters_from(past), extra_rows=extra, live_starters={})
    live = live[live["_live"]]
    assert live["starts_tonight"].isna().all() and live["team_usual_starters_out"].isna().all()
