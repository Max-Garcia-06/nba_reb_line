import os

import pytest

# Must run before any test imports config (conftest loads first).
os.environ["NBA_REB_NO_DOTENV"] = "1"


@pytest.fixture(autouse=True)
def _no_real_model_lag(monkeypatch, tmp_path):
    # The model-lag and blend-age preflights read models/ and data/nba_reb.db; keep
    # tests off the real files so their pass/fail can't drift with the live DB. By
    # default the model is current (lag 0) and no blend meta exists. Tests that need
    # a lag or blend meta stub these themselves.
    monkeypatch.setattr("calibrate_preflight._latest_game_date", lambda: "2026-06-13")
    monkeypatch.setattr("calibrate_preflight._model_data_end", lambda: "2026-06-13")
    monkeypatch.setattr("calibrate_preflight.BLEND_META_PATH", tmp_path / "_no_blend_meta.json")
    monkeypatch.setattr("calibrate_preflight.SEGMENT_BLEND_META_PATH", tmp_path / "_no_blend_seg.json")
