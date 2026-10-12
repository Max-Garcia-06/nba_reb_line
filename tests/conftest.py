import pytest


@pytest.fixture(autouse=True)
def _no_real_model_lag(monkeypatch):
    # The model-lag preflight reads models/ and data/nba_reb.db; keep tests off the
    # real files so their pass/fail can't drift with the live DB. Tests that need a
    # lag stub these themselves.
    monkeypatch.setattr("calibrate_preflight._latest_game_date", lambda: None)
