"""Teammate vacuum must use only what was knowable before tip, the same way in history and live."""

from __future__ import annotations

import pandas as pd

import feature_store as fs
from tests.test_feature_store_pit import FEATS, _build, _key, _synthetic_gamelogs


def _team1_game(gl, day):
    d = pd.Timestamp("2024-10-22") + pd.Timedelta(days=day)
    g = gl[(pd.to_datetime(gl["GAME_DATE"]) == d) & (gl["TEAM_ID"] == 1)]
    return g["GAME_ID"].iloc[0], d


def _vac(df, gid, team=1):
    return df[(df["GAME_ID"] == gid) & (df["TEAM_ID"] == team)][["vacuum_n", "vacuum_reb"]].iloc[0]


def test_covered_game_counts_only_reported_absences():
    gl = _synthetic_gamelogs()
    gid, d = _team1_game(gl, 40)
    # Star 100 sits tonight without being on the injury report (late scratch).
    gl = gl[~((gl["GAME_ID"] == gid) & (gl["PLAYER_ID"] == 100))]
    box = _build(gl)
    assert _vac(box, gid)["vacuum_n"] >= 1  # box-score history sees the scratch

    reported = _build(gl, pregame_out={(gid, 1): set()})
    assert _vac(reported, gid)["vacuum_n"] < _vac(box, gid)["vacuum_n"]  # report did not know

    listed = _build(gl, pregame_out={(gid, 1): {100}})
    assert _vac(listed, gid)["vacuum_n"] == _vac(box, gid)["vacuum_n"]


def test_uncovered_game_falls_back_to_box_scores():
    gl = _synthetic_gamelogs()
    gid, _ = _team1_game(gl, 40)
    other, _ = _team1_game(gl, 42)
    a = _build(gl, pregame_out={(other, 1): set()})
    b = _build(gl)
    assert (_vac(a, gid) == _vac(b, gid)).all()


def _with_trade(gl):
    """Player 101 leaves team 1 after day 39 and plays for team 3 from day 46."""
    gl = gl.copy()
    d = pd.to_datetime(gl["GAME_DATE"])
    gone = (gl["PLAYER_ID"] == 101) & (d > pd.Timestamp("2024-10-22") + pd.Timedelta(days=39))
    gl = gl[~gone]
    later = gl[(gl["TEAM_ID"] == 3) & (d.loc[gl.index] >= pd.Timestamp("2024-10-22") + pd.Timedelta(days=46))]
    moved = later.drop_duplicates("GAME_ID").assign(PLAYER_ID=101, PLAYER_NAME="P101")
    return pd.concat([gl, moved], ignore_index=True)


def test_traded_player_is_absent_for_old_team_in_covered_history():
    gl = _with_trade(_synthetic_gamelogs())
    gid, _ = _team1_game(gl, 42)
    df = _build(gl, pregame_out={(gid, 1): set()})
    assert _vac(df, gid)["vacuum_n"] >= 1


def test_traded_player_parity_history_vs_live():
    gl = _with_trade(_synthetic_gamelogs())
    gid, d = _team1_game(gl, 42)
    hist = _key(_build(gl, pregame_out={(gid, 1): set(), (gid, 2): set(), (gid, 3): set()}))
    tonight = gl[pd.to_datetime(gl["GAME_DATE"]) == d]
    past = gl[pd.to_datetime(gl["GAME_DATE"]) < d]
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
    # Live: injury report lists nobody; rosters say 101 now belongs to team 3.
    live = _build(past, extra_rows=extra, absent_override={}, current_team={101: 3})
    live = _key(live[live["_live"]])
    team1 = [k for k in live.index if k[1] == gid and live.loc[k, "TEAM_ID"] == 1]
    assert team1
    pd.testing.assert_frame_equal(hist.loc[team1, FEATS], live.loc[team1, FEATS], check_dtype=False)
