import pandas as pd
import pytest
from sqlalchemy import create_engine

import lineups as lu


def _p(pid, team, pos="", lineup="Confirmed", roster="Active", ts="2026-03-10T21:53:00"):
    return {"personId": pid, "teamId": team, "playerName": f"P{pid}", "lineupStatus": lineup,
            "position": pos, "rosterStatus": roster, "timestamp": ts}


def _payload(home_players, away_players, gid="0022500936"):
    return {"games": [{"gameId": gid, "homeTeam": {"teamId": 1, "players": home_players},
                       "awayTeam": {"teamId": 2, "players": away_players}}]}


FULL = _payload([_p(i, 1, "G") for i in range(1, 6)] + [_p(6, 1), _p(7, 1, roster="Inactive")],
                [_p(i, 2, "F") for i in range(11, 16)] + [_p(17, 2, roster="Inactive")])


def test_parse_and_states_confirmed():
    df = lu.parse_lineups(FULL, "2026-03-10")
    assert len(df) == 13 and set(df["GAME_ID"]) == {"0022500936"}
    st = lu.lineup_states(df)["0022500936"]
    assert st.confirmed and st.starters == {1, 2, 3, 4, 5, 11, 12, 13, 14, 15}
    assert st.inactive == {1: {7}, 2: {17}}


def test_lineup_states_unconfirmed_cases():
    expected = _payload([_p(i, 1, "G", lineup="Expected") for i in range(1, 6)],
                        [_p(i, 2, "F") for i in range(11, 16)])
    four = _payload([_p(i, 1, "G") for i in range(1, 5)], [_p(i, 2, "F") for i in range(11, 16)])
    one_team = {"games": [{"gameId": "g", "homeTeam": {"teamId": 1, "players": [_p(i, 1, "G") for i in range(5)]},
                           "awayTeam": {"teamId": 2, "players": []}}]}
    for payload in (expected, four, one_team):
        states = lu.lineup_states(lu.parse_lineups(payload, "2026-03-10"))
        assert not any(s.confirmed for s in states.values())


def test_empty_season_file_parses_to_no_rows():
    assert lu.parse_lineups({"games": []}, "2024-11-05").empty


def test_pregame_inactive_respects_cutoff(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    pd.DataFrame([{"GAME_ID": "0022500936", "TIP_UTC": "2026-03-10T23:00:00Z", "GAME_DATE_EST": "2026-03-10"},
                  {"GAME_ID": "0022500937", "TIP_UTC": "2026-03-10T23:30:00Z", "GAME_DATE_EST": "2026-03-10"}]
                 ).to_sql("schedule", eng, index=False)
    early = lu.parse_lineups(FULL, "2026-03-10")                                   # 21:53Z: before T-30
    late = lu.parse_lineups(_payload([_p(8, 1, roster="Inactive", ts="2026-03-10T23:10:00Z")], [],
                                     gid="0022500937"), "2026-03-10")               # after T-30
    pd.concat([early, late]).to_sql(lu.TABLE, eng, index=False)
    out = lu.pregame_inactive(eng, lead_min=30)
    assert out[("0022500936", 1)] == {7} and out[("0022500936", 2)] == {17}
    assert ("0022500937", 1) not in out


def test_inactive_group_adds_listed_players_to_vacuum():
    import feature_store as fs
    from tests.test_feature_store_pit import _synthetic_gamelogs, _tracking

    gl = _synthetic_gamelogs()
    gid, team = gl["GAME_ID"].iloc[-1], int(gl["TEAM_ID"].iloc[-1])
    star = team * 100  # k=0: biggest minutes, always in rotation
    played = gl[~((gl["GAME_ID"] == gid) & (gl["PLAYER_ID"] == star))]
    base = fs.build_feature_table(gamelogs=played, tracking=_tracking(), apply_filters=False,
                                  pregame_out={(gid, team): set()})
    with_inactive = fs.build_feature_table(gamelogs=played, tracking=_tracking(), apply_filters=False,
                                           groups=["inactive"], pregame_out={(gid, team): set()},
                                           pregame_inactive={(gid, team): {star}})
    b = base[(base["GAME_ID"] == gid) & (base["TEAM_ID"] == team)]["vacuum_n"].iloc[0]
    w = with_inactive[(with_inactive["GAME_ID"] == gid) & (with_inactive["TEAM_ID"] == team)]["vacuum_n"].iloc[0]
    assert w == b + 1
