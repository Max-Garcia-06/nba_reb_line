"""Market -> (player, team, opponent, game) resolution for the live scan."""

import pandas as pd

import slate
from identity_bridge import PlayerIndex, norm_player_name
from kalshi_bridge import MarketLine


def test_norm_player_name_accents_and_suffixes():
    assert norm_player_name("Nikola Jokić") == "nikola jokic"
    assert norm_player_name("Wendell Carter Jr.") == "wendell carter"
    assert norm_player_name("De'Aaron Fox") == "deaaron fox"
    assert norm_player_name("Kristaps Porziņģis") == "kristaps porzingis"


def test_player_index_initial_fallback_only_when_unique():
    pi = PlayerIndex([(1, "Jalen Williams"), (2, "Jaylin Williams"), (3, "Nikola Jokic")])
    assert pi.resolve("Nikola Jokić") == 3
    assert pi.resolve("J. Williams") == 0          # ambiguous -> refuse
    assert PlayerIndex([(1, "Jalen Williams")]).resolve("J Williams") == 1


def _ml(name, event="KXNBAREB-26MAR10DENLAL"):
    return MarketLine(ticker=f"{event}-{name[:3].upper()}-8", player_name=name, player_id=0,
                      game_date="2026-03-10", line=7.5, yes_ask=0.5, yes_bid=0.48,
                      no_ask=0.52, no_bid=0.5, event_ticker=event)


def test_resolve_slate_maps_team_opponent_and_home(monkeypatch):
    monkeypatch.setattr(slate.de, "slate_schedule_index", lambda d: {
        "DENLAL": {"game_id": "0022500900", "home_team_id": 20, "away_team_id": 10,
                   "status": "Scheduled", "start_utc": None}})
    monkeypatch.setattr(slate, "recent_rosters", lambda d, teams: pd.DataFrame({
        "PLAYER_ID": [1, 2, 3], "PLAYER_NAME": ["Nikola Jokić", "LeBron James", "Someone Else"],
        "TEAM_ID": [10, 20, 30]}))
    lines = [_ml("Nikola Jokic"), _ml("LeBron James"), _ml("Someone Else"), _ml("X", event="BAD")]
    out, _, unresolved = slate.resolve_slate(lines, "2026-03-10")
    jok = out[lines[0].ticker]
    assert (jok.player_id, jok.team_id, jok.opp_team_id, jok.is_home) == (1, 10, 20, 0)
    leb = out[lines[1].ticker]
    assert (leb.player_id, leb.team_id, leb.opp_team_id, leb.is_home) == (2, 20, 10, 1)
    # Player on a team not in this game, and an unparseable event, are refused.
    assert set(unresolved) == {lines[2].ticker, lines[3].ticker}


def test_playoff_flag_excludes_play_in():
    assert slate._is_playoffs("0042500101")
    assert not slate._is_playoffs("0052500101")
    assert not slate._is_playoffs("0022500101")


def _gl():
    return pd.DataFrame({
        "PLAYER_ID": [1, 2, 3, 4],
        "PLAYER_NAME": ["Stays Put", "Moved Teams", "Free Agent", "Long Injury"],
        "TEAM_ID": [10, 20, 10, 10],
        "GAME_DATE": ["2026-04-10", "2026-04-12", "2026-04-12", "2025-12-01"],
    })


def test_recent_rosters_live_roster_overrides_last_played_team(monkeypatch):
    # Offseason: player 2 moved 20 -> 10, player 3 is unsigned, player 4 missed
    # all of last spring but is on the opening-night roster.
    monkeypatch.setattr(slate.de, "current_team_by_player", lambda d: {1: 10, 2: 10, 4: 10})
    r = slate.recent_rosters("2026-10-20", {10}, gl=_gl())
    assert sorted(r["PLAYER_ID"]) == [1, 2, 4]
    assert set(r["TEAM_ID"]) == {10}


def test_recent_rosters_falls_back_to_last_played_team(monkeypatch):
    # No live roster (historical date or nba_api down): last team played for,
    # within the 200-day window — the pre-existing point-in-time behaviour.
    monkeypatch.setattr(slate.de, "current_team_by_player", lambda d: {})
    r = slate.recent_rosters("2026-10-20", {10}, gl=_gl())
    assert sorted(r["PLAYER_ID"]) == [1, 3]


def test_current_team_by_player_only_for_live_dates(monkeypatch):
    import data_engine as de

    calls = []
    monkeypatch.setattr(de, "_fetch_current_rosters", lambda season: calls.append(season) or {7: 99})
    monkeypatch.setattr(de, "_today_et", lambda: "2026-10-20")
    assert de.current_team_by_player("2026-03-10") == {}      # backtests stay point-in-time
    assert de.current_team_by_player("2026-10-20") == {7: 99}
    assert calls == ["2026-27"]


def test_current_team_by_player_survives_api_failure(monkeypatch):
    import data_engine as de

    def boom(season):
        raise RuntimeError("stats.nba.com down")

    monkeypatch.setattr(de, "_fetch_current_rosters", boom)
    monkeypatch.setattr(de, "_today_et", lambda: "2026-10-20")
    assert de.current_team_by_player("2026-10-21") == {}
