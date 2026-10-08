"""
Point-in-time guards for feature_store.

1. No look-ahead: features for games on/before date D must be identical
   whether or not games after D exist in the data.
2. Train/serve parity: a game passed as `extra_rows` (live scan path, no box
   score, absences from absent_override) must get the same features as the
   same game built from history.
"""

import numpy as np
import pandas as pd
import pytest

import feature_store as fs

FEATS = fs.MODEL_FEATURES


def _synthetic_gamelogs(n_days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [1, 2, 3, 4]
    rows = []
    start = pd.Timestamp("2024-10-22")
    gid = 0
    for d in range(n_days):
        date = start + pd.Timedelta(days=d)
        pairs = [(1, 2), (3, 4)] if d % 2 == 0 else [(1, 3), (2, 4)]
        for home, away in pairs:
            gid += 1
            for team in (home, away):
                for k in range(8):
                    pid = team * 100 + k
                    if rng.random() < 0.08:  # random absence
                        continue
                    minutes = max(0.0, rng.normal(34 - 3 * k, 4))
                    reb = int(rng.poisson(max(0.2, (10 - k) * minutes / 36)))
                    oreb = int(rng.binomial(reb, 0.25))
                    fga = int(rng.poisson(minutes / 3))
                    rows.append({
                        "PLAYER_ID": pid, "PLAYER_NAME": f"P{pid}", "TEAM_ID": team,
                        "GAME_ID": f"{gid:010d}", "GAME_DATE": date.isoformat(),
                        "MATCHUP": f"T{team} {'vs.' if team == home else '@'} T{away if team == home else home}",
                        "MIN": minutes, "REB": reb, "OREB": oreb, "DREB": reb - oreb,
                        "FGA": fga, "FGM": int(rng.binomial(fga, 0.47)), "FG3A": int(fga * 0.4),
                        "FTA": int(rng.poisson(2)), "TOV": int(rng.poisson(1.5)),
                        "PTS": int(rng.poisson(minutes / 2.5)),
                        "SEASON": "2024-25", "SEASON_TYPE": "Regular Season",
                    })
    return pd.DataFrame(rows)


def _tracking() -> pd.DataFrame:
    return pd.DataFrame({
        "PLAYER_ID": [100, 200], "SEASON": ["2023-24", "2023-24"], "REB": [9.0, 8.0],
        "REB_CHANCES": [15.0, 14.0], "REB_CHANCE_PCT_ADJ": [0.7, 0.6], "REB_CONTEST_PCT": [0.3, 0.2],
        "REB_CHANCE_DEFER": [1.0, 0.5], "AVG_REB_DIST": [6.0, 7.0],
    })


def _build(gl, **kw):
    return fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False, **kw)


def _key(df):
    return df.set_index(["PLAYER_ID", "GAME_ID"]).sort_index()


def test_no_lookahead():
    gl = _synthetic_gamelogs()
    cutoff = pd.Timestamp("2024-11-25")
    full = _key(_build(gl))
    trunc = _key(_build(gl[pd.to_datetime(gl["GAME_DATE"]) <= cutoff]))
    common = trunc.index
    assert len(common) > 500
    pd.testing.assert_frame_equal(full.loc[common, FEATS], trunc.loc[common, FEATS], check_dtype=False)


def test_scrambled_outcomes_dont_move_features():
    """Rows dated <= D may only use games strictly before their own date.
    Scrambling every box score on/after D must not change any of their features."""
    gl = _synthetic_gamelogs()
    cutoff = pd.Timestamp("2024-11-25")
    rng = np.random.default_rng(1)
    scr = gl.copy()
    late = pd.to_datetime(scr["GAME_DATE"]) >= cutoff
    for c in ["MIN", "REB", "OREB", "DREB", "FGA", "FGM", "FG3A", "FTA", "TOV", "PTS"]:
        scr.loc[late, c] = rng.permutation(scr.loc[late, c].values)
    a, b = _key(_build(gl)), _key(_build(scr))
    rows = a.index[a["GAME_DATE"] <= cutoff]
    assert (a.loc[rows, "GAME_DATE"] == cutoff).sum() > 10
    pd.testing.assert_frame_equal(a.loc[rows, FEATS], b.loc[rows, FEATS], check_dtype=False)


def test_live_rows_match_history():
    gl = _synthetic_gamelogs()
    last_date = pd.to_datetime(gl["GAME_DATE"]).max()
    hist = _key(_build(gl))

    tonight = gl[pd.to_datetime(gl["GAME_DATE"]) == last_date]
    past = gl[pd.to_datetime(gl["GAME_DATE"]) < last_date]

    # Absences tonight = players on each team who exist in history but have no row tonight.
    played = set(tonight["PLAYER_ID"])
    absent = {}
    for team in tonight["TEAM_ID"].unique():
        roster = set(gl.loc[gl["TEAM_ID"] == team, "PLAYER_ID"])
        absent[int(team)] = roster - played

    games = tonight[["GAME_ID", "TEAM_ID"]].drop_duplicates()
    opp = games.merge(games, on="GAME_ID")
    opp = opp[opp["TEAM_ID_x"] != opp["TEAM_ID_y"]].set_index(["GAME_ID", "TEAM_ID_x"])["TEAM_ID_y"]
    extra = pd.DataFrame({
        "PLAYER_ID": tonight["PLAYER_ID"].values,
        "PLAYER_NAME": tonight["PLAYER_NAME"].values,
        "TEAM_ID": tonight["TEAM_ID"].values,
        "OPP_TEAM_ID": [opp[(g, t)] for g, t in zip(tonight["GAME_ID"], tonight["TEAM_ID"])],
        "GAME_ID": tonight["GAME_ID"].values,
        "GAME_DATE": tonight["GAME_DATE"].values,
        "SEASON": tonight["SEASON"].values,
        "is_home": tonight["MATCHUP"].str.contains("vs.", regex=False).astype(int).values,
        "is_playoffs": 0,
    })
    live = _build(past, extra_rows=extra, absent_override=absent)
    live = _key(live[live["_live"]])
    assert len(live) == len(tonight)
    pd.testing.assert_frame_equal(hist.loc[live.index, FEATS], live[FEATS], check_dtype=False)


def test_prev_season_tracking_only():
    gl = _synthetic_gamelogs(n_days=10)
    df = _build(gl)
    # Tracking provided for 2023-24 only; rows are 2024-25 -> must use it as PRIOR season.
    assert df.loc[df["PLAYER_ID"] == 100, "prev_reb_pg"].eq(9.0).all()
    same = _tracking().assign(SEASON="2024-25")
    df2 = fs.build_feature_table(gamelogs=gl, tracking=same, apply_filters=False)
    assert df2["prev_reb_pg"].isna().all()
