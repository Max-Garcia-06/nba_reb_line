import os

import pytest

# Must run before any test imports config (conftest loads first).
os.environ["NBA_REB_NO_DOTENV"] = "1"


@pytest.fixture(autouse=True)
def _no_real_model_lag(monkeypatch):
    # The model-lag preflight reads models/ and data/nba_reb.db; keep tests off the
    # real files so their pass/fail can't drift with the live DB. Tests that need a
    # lag stub these themselves.
    monkeypatch.setattr("calibrate_preflight._latest_game_date", lambda: None)
