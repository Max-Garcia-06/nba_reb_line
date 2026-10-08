from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from data_engine import (
    _parse_game_datetime_utc,
    filter_market_lines_by_start_window,
    filter_market_lines_pregame,
)


@dataclass
class _Line:
    ticker: str
    event_ticker: str = ""


def test_parse_game_datetime_utc():
    dt = _parse_game_datetime_utc("2026-05-24T16:15:00Z")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.hour == 16 and dt.minute == 15


def test_filter_market_lines_by_start_window():
    now = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)
    schedule_index = {
        "BOSNYK": {
            "status": "Pre-Game",
            "start_utc": now + timedelta(hours=1),
        },
        "SASPHX": {
            "status": "Pre-Game",
            "start_utc": now + timedelta(hours=4),
        },
        "DALDEN": {
            "status": "Pre-Game",
            "start_utc": now - timedelta(hours=1),
        },
    }
    lines = [
        _Line("KXNBAREB-26MAY20BOSNYK-PLAYER-2", "KXNBAREB-26MAY20BOSNYK"),
        _Line("KXNBAREB-26MAY20SASPHX-PLAYER-2", "KXNBAREB-26MAY20SASPHX"),
        _Line("KXNBAREB-26MAY20DALDEN-PLAYER-2", "KXNBAREB-26MAY20DALDEN"),
        _Line("MOCK-TICKER"),
    ]
    kept, excluded = filter_market_lines_by_start_window(
        lines,
        "2026-05-24",
        within_hours=3.0,
        now=now,
        schedule_index=schedule_index,
    )
    assert len(kept) == 1
    assert "BOSNYK" in kept[0].event_ticker
    reasons = {e[3] for e in excluded}
    assert "too_far" in reasons
    assert "already_started" in reasons
    assert "unparseable_event" in reasons


def test_filter_market_lines_by_start_window_disabled():
    lines = [_Line("KXNBAREB-26MAY20BOSNYK-PLAYER-2", "KXNBAREB-26MAY20BOSNYK")]
    kept, excluded = filter_market_lines_by_start_window(
        lines,
        "2026-05-24",
        within_hours=0,
    )
    assert len(kept) == 1
    assert excluded == []


def test_filter_market_lines_pregame_with_schedule_index():
    schedule_index = {
        "DALDEN": {"status": "In Progress", "start_utc": None},
        "BOSNYK": {"status": "Pre-Game", "start_utc": None},
    }
    lines = [
        _Line("KXNBAREB-26MAY20DALDEN-PLAYER-2", "KXNBAREB-26MAY20DALDEN"),
        _Line("KXNBAREB-26MAY20BOSNYK-PLAYER-2", "KXNBAREB-26MAY20BOSNYK"),
        _Line("MOCK-TICKER"),
    ]
    kept, excluded = filter_market_lines_pregame(
        lines, "2026-05-20", schedule_index=schedule_index
    )
    assert len(kept) == 2
    assert len(excluded) == 1
    assert excluded[0][1] == "DALDEN"
