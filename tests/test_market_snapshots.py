from kalshi_bridge import MarketLine
from market_snapshots import MarketSnapshot, append_snapshots, load_snapshots, snapshot_path


def test_snapshot_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("market_snapshots.SNAPSHOT_DIR", tmp_path)
    ml = MarketLine(
        ticker="KXNBAREB-26MAY20BOSNYK-P-2",
        player_name="Test Player",
        player_id=1,
        game_date="2026-05-20",
        line=1.5,
        yes_ask=0.52,
        yes_bid=0.50,
        no_ask=0.52,
        no_bid=0.50,
        event_ticker="KXNBAREB-26MAY20BOSNYK",
    )
    append_snapshots("2026-05-20", [ml])
    snaps = load_snapshots("2026-05-20")
    assert len(snaps) == 1
    assert snaps[0].player_name == "Test Player"
    assert snapshot_path("2026-05-20").exists()


def test_snapshot_refuses_mock_client(monkeypatch):
    """Cron snapshots without .env would write placeholder books into the research tape."""
    import pytest
    import typer

    import kalshi_bridge
    import market_snapshots
    from pipeline import commands

    monkeypatch.setattr(kalshi_bridge, "get_client", lambda: kalshi_bridge.MockKalshiClient())
    wrote = []
    monkeypatch.setattr(market_snapshots, "append_snapshots", lambda gd, lines: wrote.append(lines) or 0)
    with pytest.raises(typer.Exit) as e:
        commands.snapshot(game_date="2026-10-20", series_ticker=None)
    assert e.value.exit_code == 1 and not wrote


def test_schedule_snapshots_refuses_mock_client(monkeypatch):
    import pytest
    import typer

    import kalshi_bridge
    import market_snapshots
    from pipeline import commands

    monkeypatch.setattr(kalshi_bridge, "get_client", lambda: kalshi_bridge.MockKalshiClient())
    wrote = []
    monkeypatch.setattr(market_snapshots, "append_snapshots", lambda gd, lines: wrote.append(lines) or 0)
    with pytest.raises(typer.Exit) as e:
        commands.schedule_snapshots(game_date="2026-10-20", interval=0, count=1, series_ticker=None)
    assert e.value.exit_code == 1 and not wrote
