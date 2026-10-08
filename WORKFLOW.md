# NBA Rebounds Pipeline — Workflow

How to run the system. The *why* lives in `SYSTEM.md`.

```bash
.venv/bin/python run_pipeline.py <command> [options]
```

## 1. Pipeline

```text
nba_api ──etl──► SQLite: player_gamelogs, tracking_rebounds, schedule
                    │
                    ▼  feature_store (point-in-time as-of joins)
                 train (MODEL_FAMILY, default c7_ens_c4_c5) ──► models/rebound_model.pkl
                    │                                          + model_meta.pkl (trained_on)
                    │                                          + p_calibrator_oof.pkl
Kalshi ◄──scan──────┘  slate.py: markets → (player, team, opp, game) → PIT features
   │                   → PMF → P(REB > k) → calibrate → blend w/ market (per bucket)
   │                   → fee-adjusted edge vs maker/taker limit → Kelly → risk gates
   ├──► data/trades_YYYY-MM-DD.jsonl  (journal)   ──► reconcile ──► report / report-range
   ├──► data/snapshots/*.jsonl        (books)     ──► model-vs-market / refit-blend / backtest
   └──► CLV marks (15/30/60/90m)                  ──► segment-report
```

| Phase | Command(s) | Output |
|---|---|---|
| Data | `etl [--incremental]` | `data/nba_reb.db` |
| History | `python kalshi_history.py markets` / `candles` | `kalshi_markets`, `kalshi_candles` (2025-26) |
| Model | `train`, `evaluate` | model + OOF calibrator; walk-forward scores |
| Research | `python bakeoff.py run/summary`, `python market_eval.py` | model zoo comparison; model vs Kalshi |
| Markets | `snapshot`, `schedule-snapshots` | `data/snapshots/` |
| Trade | `scan [--live]`, `mark`, `reconcile` | journal, execution ledger |
| Review | `report`, `report-range`, `calibrate`, `segment-report` | P&L net of fees, calibrators, go/no-go |
| Blend | `refit-blend`, `fit-blend-segments`, `model-vs-market` | `models/blend_meta*.json` |

## 2. Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env      # Kalshi key id + PEM path; demo host until you mean it
```

Set `REQUIRE_KALSHI_CREDENTIALS=1` in production so a missing key fails instead of
silently using the mock client. `touch data/KILL_SWITCH` halts live orders; risk-limit
breaches create it automatically (`AUTO_KILL_ON_RISK_BREACH=true`).

## 3. Cold start / new season

```bash
# 1. Add the new season to SEASONS (config/.env), then pull everything
.venv/bin/python run_pipeline.py etl
# 2. Train + OOF calibrator (stamped with trained_on)
.venv/bin/python run_pipeline.py train
# 3. Seed blend weights from last season's full-slate history (PIT weekly models)
.venv/bin/python run_pipeline.py refit-blend --start 2025-11-18 --end 2026-06-13
```

Early season: players' rolling windows reach back into last season; `games_in_season`
and prior-season tracking tell the model how stale that is. Retrain weekly for the first
month (`train`), then every 2–4 weeks — there is still no cron for retraining.

## 4. Game day

| When (ET) | What |
|---|---|
| 12:00–22:00 every 2h | `snapshot` (cron) |
| 17:05–23:05 hourly | `scan` (dry-run cron; `scan-live` when trusted). Only games tipping within `SCAN_WITHIN_HOURS`=1.5 |
| before tip, optional | `data/absent_YYYY-MM-DD.json` `{"OUT": ["Name", ...]}` to add late scratches the injury feed missed |
| 04:30 (launchd) | `nightly`: `etl --incremental`, `reconcile` (3-day lookback), `report`; Sundays also `refit-blend` |

Scan skips — never prices off the market — any market whose player can't be resolved to
tonight's game or has fewer than `GAMES_FOR_LAMBDA_SANITY` games.

## 5. Feedback loop (weekly)

```bash
.venv/bin/python run_pipeline.py report-range --start <d> --end <d>   # net-of-fee P&L = truth
.venv/bin/python run_pipeline.py model-vs-market --start <d> --end <d> # full-slate, PIT
.venv/bin/python run_pipeline.py segment-report                        # CLV go/no-go
.venv/bin/python run_pipeline.py calibrate                             # once >=50 fills
```

Then, and only with evidence: `BLOCKED_SEGMENTS`, `RISKY_BAND_KELLY_MULT`, `MAX_YES_LINE`,
`REQUIRE_FILL_CALIB_FOR_LIVE=1` once a fill calibrator exists.

## 6. Install schedules

```bash
scripts/install_phase1_cron.sh   # cron (snapshots, dry-run scans, watchdog) + launchd nightly
```

Cron hours in `scripts/crontab.phase1.example` are system-local (PT): ET − 3.
