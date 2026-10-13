from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from injury_reports import (
    NOT_SUBMITTED,
    parse_words,
    pregame_out_by_game,
    report_url,
    team_tricode,
    to_first_last,
)

# Column x0 positions as in the 2025-26 report layout.
X = {"date": 24, "time": 121, "matchup": 201, "team": 265, "player": 426, "status": 587, "reason": 667}


def _w(col, top, text, dx=0):
    return {"x0": X[col] + dx, "top": top, "text": text}


def _header():
    return [
        {"x0": 289, "top": 46, "text": "Injury"}, {"x0": 356, "top": 46, "text": "Report:"},
        {"x0": 438, "top": 46, "text": "03/02/26"}, {"x0": 540, "top": 46, "text": "05:00"},
        {"x0": 602, "top": 46, "text": "PM"},
        _w("date", 108, "Game"), _w("date", 108, "Date", 30), _w("time", 108, "Game"), _w("time", 108, "Time", 28),
        _w("matchup", 108, "Matchup"), _w("team", 108, "Team"), _w("player", 108, "Player"),
        _w("player", 108, "Name", 31), _w("status", 108, "Current"), _w("status", 108, "Status", 37),
        _w("reason", 108, "Reason"),
    ]


PAGE1 = _header() + [
    _w("date", 129, "03/02/2026"), _w("time", 129, "07:00"), _w("time", 129, "(ET)", 25),
    _w("matchup", 129, "HOU@WAS"), _w("team", 129, "Houston"), _w("team", 129, "Rockets", 37),
    _w("player", 129, "Adams,"), _w("player", 129, "Steven", 32), _w("status", 129, "Out"),
    _w("reason", 129, "Injury/Illness"),
    _w("player", 173, "Smith"), _w("player", 173, "Jr.,", 26), _w("player", 173, "Jabari", 40),
    _w("status", 173, "Questionable"), _w("reason", 173, "Injury/Illness"),
    _w("reason", 190, "Sprain"),  # wrapped reason line: no player
    _w("team", 210, "Washington"), _w("team", 210, "Wizards", 50),
    _w("player", 210, "Sarr,"), _w("player", 210, "Alex", 25), _w("status", 210, "Out"),
    _w("reason", 210, "Not"), _w("reason", 210, "With", 20), _w("reason", 210, "Team", 40),  # must not move columns
    _w("time", 240, "10:00"), _w("time", 240, "(ET)", 25), _w("matchup", 240, "LAC@GSW"),
    _w("team", 240, "LA"), _w("team", 240, "Clippers", 15),
    _w("reason", 240, "NOT"), _w("reason", 240, "YET", 20), _w("reason", 240, "SUBMITTED", 37),
]
PAGE2 = [  # continuation page: no column header; the previous game carries over
    {"x0": 289, "top": 46, "text": "Injury"},
    _w("team", 83, "Golden"), _w("team", 83, "State", 35), _w("team", 83, "Warriors", 62),
    _w("player", 83, "Kuminga,"), _w("player", 83, "Jonathan", 45), _w("status", 83, "Out"),
]


def test_parse_rows_carry_forward_game_and_team():
    rows = parse_words([PAGE1, PAGE2])
    got = [(r["game_date"], r["matchup"], r["team"], r["player_name"], r["status"]) for r in rows]
    assert ("2026-03-02", "HOU@WAS", "HOU", "Steven Adams", "Out") in got
    assert ("2026-03-02", "HOU@WAS", "HOU", "Jabari Smith Jr.", "Questionable") in got
    assert ("2026-03-02", "HOU@WAS", "WAS", "Alex Sarr", "Out") in got
    assert ("2026-03-02", "LAC@GSW", "LAC", "", NOT_SUBMITTED) in got
    assert ("2026-03-02", "LAC@GSW", "GSW", "Jonathan Kuminga", "Out") in got
    assert len(rows) == 5


@pytest.mark.parametrize("raw,name", [("Adams, Steven", "Steven Adams"), ("Smith Jr., Jabari", "Jabari Smith Jr."),
                                      ("Nance Jr.,Larry", "Larry Nance Jr."), ("Nene", "Nene")])
def test_to_first_last(raw, name):
    assert to_first_last(raw) == name


@pytest.mark.parametrize("team,code", [("Houston Rockets", "HOU"), ("LA Clippers", "LAC"),
                                       ("Portland Trail Blazers", "POR"), ("Philadelphia 76ers", "PHI")])
def test_team_tricode(team, code):
    assert team_tricode(team) == code


def test_report_url_format():
    assert report_url(datetime(2026, 3, 2, 17, 0)) == \
        "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-03-02_05_00PM.pdf"
    assert report_url(datetime(2026, 3, 2, 11, 30)).endswith("Injury-Report_2026-03-02_11_30AM.pdf")


# ---------- pregame knowable absences ----------

def _reports(rows):
    return pd.DataFrame(rows, columns=["report_ts", "game_date", "matchup", "team", "player_name", "status"])


def test_pregame_out_uses_latest_submitted_report_before_cutoff():
    rep = _reports([
        ("2026-03-02T17:00", "2026-03-02", "HOU@WAS", "HOU", "Steven Adams", "Out"),
        ("2026-03-02T17:45", "2026-03-02", "HOU@WAS", "HOU", "Steven Adams", "Out"),
        ("2026-03-02T17:45", "2026-03-02", "HOU@WAS", "HOU", "Fred VanVleet", "Out"),
        ("2026-03-02T18:30", "2026-03-02", "HOU@WAS", "HOU", "Alperen Sengun", "Out"),  # after cutoff
        ("2026-03-02T17:45", "2026-03-02", "HOU@WAS", "WAS", "", NOT_SUBMITTED),
        ("2026-03-02T17:00", "2026-03-02", "HOU@WAS", "WAS", "Alex Sarr", "Out"),
    ])
    games = pd.DataFrame([{"GAME_ID": "g1", "TEAM_ID": 1, "tricode": "HOU", "matchup": "HOU@WAS",
                           "game_date": "2026-03-02", "cutoff": "2026-03-02T18:00"},
                          {"GAME_ID": "g1", "TEAM_ID": 2, "tricode": "WAS", "matchup": "HOU@WAS",
                           "game_date": "2026-03-02", "cutoff": "2026-03-02T18:00"}])
    roster = {1: {101: "Steven Adams", 102: "Fred VanVleet", 103: "Alperen Sengun"}, 2: {201: "Alex Sarr"}}
    out = pregame_out_by_game(rep, games, roster)
    assert out[("g1", 1)] == {101, 102}
    # WAS's latest report before the cutoff was NOT YET SUBMITTED: fall back to its last submitted one.
    assert out[("g1", 2)] == {201}


def test_game_without_report_coverage_is_absent_from_mapping():
    games = pd.DataFrame([{"GAME_ID": "g9", "TEAM_ID": 1, "tricode": "HOU", "matchup": "HOU@WAS",
                           "game_date": "2024-03-02", "cutoff": "2024-03-02T18:00"}])
    assert pregame_out_by_game(_reports([]), games, {1: {}}) == {}


def test_backfill_stores_latest_report_before_each_tip_minus_lead(tmp_path, monkeypatch):
    import injury_reports as ir
    from sqlalchemy import create_engine

    asked = []

    def fake_latest(cut, **kw):
        asked.append(cut)
        ts = cut.replace(minute=0)
        return ts, tmp_path / f"{ts:%H%M}.pdf"

    monkeypatch.setattr(ir, "latest_report_before", fake_latest)
    monkeypatch.setattr(ir, "parse_pdf", lambda p: [{"game_date": "2026-03-02", "game_time": "07:00",
                                                      "matchup": "HOU@WAS", "team": "HOU",
                                                      "player_name": "Steven Adams", "status": "Out"}])
    sched = pd.DataFrame({"GAME_DATE_EST": ["2026-03-02", "2026-03-02", "2026-03-02"],
                          "TIP_UTC": ["2026-03-03T00:00:00Z", "2026-03-03T00:00:00Z", "2026-03-03T03:00:00Z"]})
    eng = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    n = ir.backfill("2026-03-02", "2026-03-02", lead_min=60, engine=eng, schedule=sched, cache_dir=tmp_path)
    # tips 19:00 and 22:00 ET -> cutoffs 18:00 and 21:00 ET, one report each
    assert [c.strftime("%H:%M") for c in asked] == ["18:00", "21:00"]
    assert n == 2
    assert len(ir.load_reports(eng, "2026-03-02", "2026-03-02")) == 2


def test_pregame_out_for_gamelogs_reads_schedule_and_reports(tmp_path):
    import injury_reports as ir
    from sqlalchemy import create_engine

    eng = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    pd.DataFrame([{"GAME_ID": "0022500900", "GAME_DATE_EST": "2026-03-02", "TIP_UTC": "2026-03-03T00:00:00Z",
                   "AWAY_TRICODE": "HOU", "HOME_TRICODE": "WAS", "AWAY_TEAM_ID": 10, "HOME_TEAM_ID": 20}]
                 ).to_sql("schedule", eng, index=False)
    ir.store_report(eng, datetime(2026, 3, 2, 17, 45), [
        {"game_date": "2026-03-02", "game_time": "07:00", "matchup": "HOU@WAS", "team": "HOU",
         "player_name": "Steven Adams", "status": "Out"}])
    ir.store_report(eng, datetime(2026, 3, 2, 18, 30), [  # after tip - 60 min: ignored
        {"game_date": "2026-03-02", "game_time": "07:00", "matchup": "HOU@WAS", "team": "HOU",
         "player_name": "Fred VanVleet", "status": "Out"}])
    gl = pd.DataFrame({"PLAYER_ID": [1, 2, 3], "PLAYER_NAME": ["Steven Adams", "Fred VanVleet", "Alex Sarr"],
                       "TEAM_ID": [10, 10, 20], "GAME_ID": ["0022500800"] * 3,
                       "GAME_DATE": pd.to_datetime(["2026-03-01"] * 3), "SEASON": ["2025-26"] * 3})
    out = ir.pregame_out_for_gamelogs(gl, lead_min=60, engine=eng)
    assert out[("0022500900", 10)] == {1}
    assert out[("0022500900", 20)] == set()


def test_fetch_falls_back_to_hourly_filename(tmp_path):
    import injury_reports as ir

    class _Resp:
        def __init__(self, code, content=b""):
            self.status_code, self.content = code, content

        def raise_for_status(self):
            raise AssertionError("unexpected status")

    class _S:
        def __init__(self):
            self.urls = []

        def get(self, url, timeout=30):
            self.urls.append(url)
            return _Resp(200, b"%PDF-1.4") if url.endswith("_05PM.pdf") else _Resp(403)

    s = _S()
    p = ir.fetch_report(datetime(2025, 11, 15, 17, 0), tmp_path, s)
    assert p is not None and p.read_bytes().startswith(b"%PDF")
    assert s.urls[0].endswith("_05_00PM.pdf") and s.urls[1].endswith("_05PM.pdf")
    # quarter-hour times have no hourly equivalent
    s2 = _S()
    assert ir.fetch_report(datetime(2025, 11, 15, 17, 15), tmp_path, s2) is None
    assert len(s2.urls) == 1
