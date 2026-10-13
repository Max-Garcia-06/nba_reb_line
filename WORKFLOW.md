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
   │                   → PMF → P(REB > k) → calibrate → capped blend w/ market
   │                   → fee-adjusted edge vs maker/taker limit → Kelly → risk gates
   ├──► data/trades_YYYY-MM-DD.jsonl  (journal)   ──► reconcile ──► report / report-range
   ├──► data/paper/trades_*.jsonl     (dry-run)   ──► report-range --paper
   ├──► data/snapshots/*.jsonl        (books)     ──► model-vs-market / refit-blend / backtest
   └──► CLV marks (15/30/60m after each order + close before tip) ──► segment-report
```

| Phase | Command(s) | Output |
|---|---|---|
| Data | `etl [--incremental]` | `data/nba_reb.db` |
| History | `python kalshi_history.py markets` / `candles`, `injury-backfill --start --end` | `kalshi_markets`, `kalshi_candles` (2025-26), `injury_reports` |
| Model | `train`, `evaluate` | model + OOF calibrator; walk-forward scores |
| Research | `python bakeoff.py run/summary`, `python market_eval.py` | model zoo comparison; model vs Kalshi |
| Markets | `snapshot`, `schedule-snapshots` | `data/snapshots/` |
| Trade | `scan [--live]`, `mark`, `reconcile` | journal, execution ledger |
| Review | `report`, `report-range`, `calibrate`, `segment-report` | P&L net of fees, calibrators, go/no-go |
| Blend | `refit-blend` (capped blend, weekly), `model-vs-market` | `models/blend_meta.json` |

## 2. Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env      # Kalshi key id + PEM path; demo host until you mean it
```

`scan` and `snapshot`/`schedule-snapshots` exit 1 on the mock client (missing key),
dry-run included. Every command validates `.env` at startup and refuses to run on
out-of-range values.

`scan --live` additionally:

- refuses to start unless a daily loss cap (`DAILY_LOSS_LIMIT_*`) and a deployed cap
  (`MAX_DAILY_DEPLOYED_*`) are set;
- fails closed when the balance, resting orders, fills or markets can't be read;
- runs the risk desk (`check_pre_trade_risk`) on the proposed orders, with intraday
  P&L from Kalshi fills marked at the live mid, over `LOSS_LIMIT_LOOKBACK_DAYS`;
- sizes each player against what earlier scans already committed today (`MAX_BET_PCT`);
- sends every order with a deterministic `client_order_id`, `post_only` when resting
  inside the ask, and `expiration_time` one minute before tip;
- ends every run (edges or not, error or not) by cancelling today's resting orders
  that this scan did not re-confirm.

`touch data/KILL_SWITCH` halts live orders (checked before every order) and makes the
next scan cancel all of today's resting orders; risk-limit breaches create it
automatically (`AUTO_KILL_ON_RISK_BREACH=true`).

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
and prior-season tracking tell the model how stale that is. `nightly` retrains once a
week (`AUTO_RETRAIN`, ~1 min); live scans refuse once the model trails the newest DB
game by `MODEL_MAX_DATA_LAG_DAYS` (14), so a retrain that silently stops can't go unnoticed.

Offseason moves: on today/future slates `slate.recent_rosters` takes each player's team
from today's league rosters (nba_api PlayerIndex), not his last game — otherwise every
player who changed teams is skipped (or, if his old team is in the game, mispriced) until
he plays for the new one. Historical dates and API failures fall back to last-played team.

## 4. Game day

| When (ET) | What |
|---|---|
| 12:00–22:00 every 2h | `snapshot` (cron) |
| 17:05–23:05 hourly | `scan` (dry-run cron; `scan-live` when trusted). Only games tipping within `SCAN_WITHIN_HOURS`=1.5 |
| before tip, optional | `data/absent_YYYY-MM-DD.json` `{"OUT": ["Name", ...]}` to add late scratches the injury feed missed |
| 04:30 (launchd) | `nightly`: `etl --incremental`, `injury-backfill` (yesterday), `reconcile` (3-day lookback), `report`; weekly (>6 days since last) also `train` + `refit-blend`. Steps run independently |

Scan skips — never prices off the market — any market whose player can't be resolved to
tonight's game or has fewer than `GAMES_FOR_LAMBDA_SANITY` games.

## 5. Feedback loop (weekly)

```bash
.venv/bin/python run_pipeline.py report-range --start <d> --end <d>   # net-of-fee P&L = truth
.venv/bin/python run_pipeline.py report-range --paper --start <d> --end <d>  # dry-run trial
.venv/bin/python run_pipeline.py model-vs-market --start <d> --end <d> # full-slate, PIT, ΔLL CI
.venv/bin/python run_pipeline.py segment-report                        # CLV go/no-go
.venv/bin/python run_pipeline.py calibrate                             # once >=50 fills
```

Evaluation rules (2026-10, see SYSTEM.md §5):

- Books used for scoring, blend refits and backtests are pregame only (at least 60 min
  before tip); in-game snapshots are never read.
- `backtest` is walk-forward by default: each week trades on blend weights fit only on
  earlier full-slate scores (`--blend saved` reproduces the old in-sample behaviour).
  Resting orders fill only when trades or quotes reached the limit before tip
  (`--fill-model none` = everything fills, an upper bound). `--zero-vacuum` bounds the
  hindsight in teammate-absence features. ROI comes with a day-bootstrap 95% CI.
- `mark` labels every mark (15m/30m/60m after the order's own placement, `close` in the
  last 10 min before tip) and never marks after tip. Marks without a label are legacy
  and ignored by `segment-report`.
- A dry-run `scan` journals paper orders to `data/paper/` and schedules paper marks.
- `reconcile` re-checks an order until it is executed or canceled, and books the fees
  Kalshi actually charged. `scalar` settlements (player did not play) settle at their
  settlement value; ROI uses resolved cost only.

Then, and only with evidence: `BLOCKED_SEGMENTS`, `RISKY_BAND_KELLY_MULT`, `MAX_YES_LINE`,
`REQUIRE_FILL_CALIB_FOR_LIVE=1` once a fill calibrator exists.

## 6. Install schedules

```bash
scripts/install_phase1_cron.sh   # cron (snapshots, dry-run scans, watchdog) + launchd nightly
```

Cron hours in `scripts/crontab.phase1.example` are system-local (PT): ET − 3.
