"""Short per-scan iMessage summary: sent only for scans that priced markets."""
import subprocess
from types import SimpleNamespace

import notify


def _sig(name="Neemias Queta", side="yes", line=5.5, limit=0.47, edge=0.0527, n=1):
    return SimpleNamespace(player_name=name, recommended_side=side, kalshi_line=line,
                           limit_price=limit, edge=edge, recommended_contracts=n)


def _text(**over):
    kw = dict(now_et="18:50", mode="PAPER", games=["BOS@GSW", "LAL@DEN"], n_priced=46,
              signals=[], n_new=0, outcome="")
    kw.update(over)
    return notify.format_scan_text(**kw)


def test_priced_scan_without_edges_says_so():
    assert _text() == "NBA scan 18:50 ET · PAPER\nBOS@GSW, LAL@DEN · 46 mkts priced\nNo edges."


def test_signals_show_side_as_over_under_with_limit_and_edge():
    text = _text(signals=[_sig(), _sig(name="Jalen Duren", side="no", line=11.5, limit=0.42, edge=0.061, n=2)], n_new=1)
    lines = text.splitlines()
    assert lines[2] == "2 edges (1 new):"
    assert lines[3] == "Neemias Queta o5.5 @0.47 e+0.05 x1"
    assert lines[4] == "Jalen Duren u11.5 @0.42 e+0.06 x2"


def test_long_signal_lists_are_truncated():
    text = _text(signals=[_sig(name=f"P{i}") for i in range(8)], n_new=8)
    assert text.splitlines()[-1] == "+3 more"


def test_outcome_line_is_included():
    text = _text(mode="LIVE", signals=[_sig()], n_new=None, outcome="placed 1/1 ok")
    assert "1 edge:" in text and text.endswith("placed 1/1 ok")


def test_send_imessage_noops_without_recipient(monkeypatch):
    calls = []
    monkeypatch.setattr(notify, "IMESSAGE_RECIPIENT", "")
    monkeypatch.setattr(notify.subprocess, "run", lambda *a, **k: calls.append(a))
    notify.send_imessage("hi")
    assert calls == []


def test_send_imessage_passes_text_and_recipient_as_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(notify, "IMESSAGE_RECIPIENT", "+15555550123")
    monkeypatch.setattr(notify, "SCAN_TEXT_ENABLED", True)
    monkeypatch.setattr(notify.subprocess, "run", lambda argv, **k: calls.append(argv))
    notify.send_imessage('edge "quoted"')
    argv = calls[0]
    assert argv[0] == "osascript"
    assert argv[-2:] == ['edge "quoted"', "+15555550123"]  # argv, never spliced into the script


def test_send_imessage_failure_never_raises(monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("osascript", 20)

    monkeypatch.setattr(notify, "IMESSAGE_RECIPIENT", "+15555550123")
    monkeypatch.setattr(notify, "SCAN_TEXT_ENABLED", True)
    monkeypatch.setattr(notify.subprocess, "run", boom)
    notify.send_imessage("hi")


def test_scan_games_lists_each_matchup_once_in_order():
    from pipeline.commands import _scan_games

    mls = [SimpleNamespace(event_ticker="KXNBAREB-26NOV23BOSGSW", ticker=""),
           SimpleNamespace(event_ticker="", ticker="KXNBAREB-26NOV23LALDEN-LALAREAVES15-4"),
           SimpleNamespace(event_ticker="KXNBAREB-26NOV23BOSGSW", ticker="")]
    assert _scan_games(mls) == ["BOS@GSW", "LAL@DEN"]
