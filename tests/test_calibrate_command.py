from __future__ import annotations

import pytest
import typer


def test_calibrate_runs_without_journals(tmp_path, monkeypatch):
    # Regression: a bulk edit once left `paper=paper` in calibrate (NameError).
    monkeypatch.setattr("journal_reader.DATA_DIR", tmp_path)
    from pipeline.commands import calibrate

    with pytest.raises(typer.Exit) as e:
        calibrate(start=None, end=None, min_rows=None, family="isotonic")
    assert e.value.exit_code == 0
