"""Typer application entry: registers all pipeline commands."""

from __future__ import annotations

import logging
import uuid

import typer

from cli.output import console
from pipeline import commands as cmd
from settings_validate import require_valid_settings
from structured_logging import configure_structured_logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
configure_structured_logging(run_id=str(uuid.uuid4())[:8])

app = typer.Typer(add_completion=False, help="NBA Rebounds Edge Pipeline")


@app.callback()
def validate_settings_callback() -> None:
    """Refuse to run any command on invalid .env settings."""
    try:
        require_valid_settings()
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

app.command()(cmd.etl)
app.command("injury-backfill")(cmd.injury_backfill)
app.command()(cmd.train)
app.command()(cmd.evaluate)
app.command()(cmd.scan)
app.command()(cmd.mark)
app.command()(cmd.report)
app.command("report-range")(cmd.report_range)
app.command()(cmd.reconcile)
app.command()(cmd.calibrate)
app.command("refit-blend")(cmd.refit_blend)
app.command("model-vs-market")(cmd.model_vs_market)
app.command("segment-report")(cmd.segment_report)
app.command()(cmd.snapshot)
app.command("schedule-snapshots")(cmd.schedule_snapshots)
app.command()(cmd.backtest)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
