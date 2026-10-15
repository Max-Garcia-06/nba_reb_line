from lineups import GameLineup
from slate import SlateRow, lineup_gate


def _row(gid):
    return SlateRow(player_id=1, player_name="x", team_id=10, opp_team_id=20, game_id=gid, is_home=1)


def test_slate_skips_unconfirmed_games():
    resolved = {"T1": _row("g1"), "T2": _row("g2"), "T3": _row("g3")}
    states = {"g1": GameLineup(True, {1}), "g2": GameLineup(False, {1})}
    ok, skipped = lineup_gate(resolved, states, {"lineup"})
    assert ok == {"g1"}
    assert sorted(skipped) == ["T2: lineups not confirmed", "T3: lineups not confirmed"]


def test_lineup_gate_off_without_lineup_groups():
    resolved = {"T1": _row("g1")}
    assert lineup_gate(resolved, None, {"travel"}) == ({"g1"}, [])


def test_lineup_gate_fails_closed_when_json_unreadable():
    ok, skipped = lineup_gate({"T1": _row("g1")}, None, {"inactive"})
    assert ok == set() and skipped == ["T1: lineups not confirmed"]
