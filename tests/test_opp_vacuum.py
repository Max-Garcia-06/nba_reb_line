import pandas as pd

import feature_store as fs
from tests.test_feature_store_pit import _key, _synthetic_gamelogs, _tracking

G = ["opp_vacuum"]


def _build(gl, **kw):
    return fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False, groups=G, **kw)


def test_opp_vacuum_mirrors_opponent_vacuum():
    df = _build(_synthetic_gamelogs())
    tv = df.drop_duplicates(["TEAM_ID", "GAME_ID"]).set_index(["TEAM_ID", "GAME_ID"])["vacuum_reb"]
    for r in df.sample(50, random_state=0).itertuples():
        assert r.opp_vacuum_reb == tv.get((r.OPP_TEAM_ID, r.GAME_ID), 0.0)


def test_opp_vacuum_live_parity_one_sided_slate():
    """Live slate with markets for ONE team only: opponent vacuum must still match history."""
    gl = _synthetic_gamelogs()
    last = pd.to_datetime(gl["GAME_DATE"]).max()
    hist = _key(_build(gl))
    tonight, past = gl[pd.to_datetime(gl["GAME_DATE"]) == last], gl[pd.to_datetime(gl["GAME_DATE"]) < last]
    played = set(tonight["PLAYER_ID"])
    absent = {int(t): set(gl.loc[gl["TEAM_ID"] == t, "PLAYER_ID"]) - played for t in tonight["TEAM_ID"].unique()}
    games = tonight[["GAME_ID", "TEAM_ID"]].drop_duplicates()
    opp = games.merge(games, on="GAME_ID")
    opp = opp[opp["TEAM_ID_x"] != opp["TEAM_ID_y"]].set_index(["GAME_ID", "TEAM_ID_x"])["TEAM_ID_y"]
    one_side = tonight[tonight["TEAM_ID"] == tonight["TEAM_ID"].iloc[0]]
    extra = pd.DataFrame({
        "PLAYER_ID": one_side["PLAYER_ID"].values, "PLAYER_NAME": one_side["PLAYER_NAME"].values,
        "TEAM_ID": one_side["TEAM_ID"].values,
        "OPP_TEAM_ID": [opp[(g, t)] for g, t in zip(one_side["GAME_ID"], one_side["TEAM_ID"])],
        "GAME_ID": one_side["GAME_ID"].values, "GAME_DATE": one_side["GAME_DATE"].values,
        "SEASON": one_side["SEASON"].values,
        "is_home": one_side["MATCHUP"].str.contains("vs.", regex=False).astype(int).values, "is_playoffs": 0,
    })
    live = _build(past, extra_rows=extra, absent_override=absent)
    live = _key(live[live["_live"]])
    cols = fs.FEATURE_GROUPS["opp_vacuum"]
    assert live["opp_vacuum_n"].sum() > 0 or hist.loc[live.index, "opp_vacuum_n"].sum() == 0
    pd.testing.assert_frame_equal(hist.loc[live.index, cols], live[cols], check_dtype=False)
