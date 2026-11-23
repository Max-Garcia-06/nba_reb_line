"""Live game status comes from stats.nba.com ScoreboardV3 for today only.

The cdn.nba.com live scoreboard returns 403 from this host, so every scan fell back to
stored schedule status and every historical replay paid a failed request per date.
"""
import pytest

import data_engine as de


class _FakeScoreboard:
    calls: list[dict] = []
    games: list[dict] = []

    def __init__(self, **kwargs):
        type(self).calls.append(kwargs)

    def get_dict(self):
        return {"scoreboard": {"gameDate": "2026-11-23", "games": type(self).games}}


@pytest.fixture
def fake_sb(monkeypatch):
    _FakeScoreboard.calls = []
    _FakeScoreboard.games = []
    monkeypatch.setattr(de.scoreboardv3, "ScoreboardV3", _FakeScoreboard)
    monkeypatch.setattr(de, "_today_et", lambda: "2026-11-23")
    return _FakeScoreboard


def test_today_reads_scoreboardv3_for_that_date(fake_sb):
    fake_sb.games = [
        {"gameId": "0022600290", "gameStatus": 1, "gameStatusText": "7:30 pm ET"},
        {"gameId": "0022600291", "gameStatus": 2, "gameStatusText": "Q2 5:12"},
        {"gameId": "0022600292", "gameStatus": 3, "gameStatusText": "Final"},
    ]
    status = de._live_status_by_game_id("2026-11-23")
    assert status == {"0022600290": "Scheduled", "0022600291": "In Progress", "0022600292": "Final"}
    assert fake_sb.calls[0]["game_date"] == "2026-11-23"


def test_historical_date_makes_no_request(fake_sb):
    assert de._live_status_by_game_id("2026-03-10") == {}
    assert fake_sb.calls == []


@pytest.mark.parametrize("text", ["PPD", "Postponed"])
def test_postponed_game_is_not_scannable(fake_sb, text):
    fake_sb.games = [{"gameId": "0022600290", "gameStatus": 1, "gameStatusText": text}]
    status = de._live_status_by_game_id("2026-11-23")["0022600290"]
    assert status == "Postponed"
    assert not de.game_status_allows_scan(status)


def test_request_failure_falls_back_to_empty(monkeypatch):
    class Boom:
        def __init__(self, **kwargs):
            raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(de.scoreboardv3, "ScoreboardV3", Boom)
    monkeypatch.setattr(de, "_today_et", lambda: "2026-11-23")
    assert de._live_status_by_game_id("2026-11-23") == {}
