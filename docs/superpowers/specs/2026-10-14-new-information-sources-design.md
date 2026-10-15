# New information sources — design

Date: 2026-10-14 · Branch: `feat/new-info-sources` · Status: approved in chat, awaiting spec review

## 1. Goal and success criteria

The model ties the market on 2025-26 full slates (ΔLL +0.0020, CI −0.0015..+0.0056) and the honest
Tier 4 holdout backtest shows no evidence of edge (ROI CI −26..+70%, CLV −0.0064). Re-tuning the
stack cannot fix that; the model needs information the market prices late or not at all.

This project adds five free sources as feature groups and lets a walk-forward ablation decide which
ship.

Success, per group and for the combined set:

1. Walk-forward `bakeoff.py` line log-loss (c7) improves vs the current feature set, with a
   date-clustered bootstrap 95% CI that excludes zero; **or** the 2025-26 full-slate model-vs-market
   ΔLL improves without line log-loss getting worse.
2. Every new feature is point-in-time: knowable at the decision time (§3). The existing scramble
   guard and live-equals-history guard are extended to cover it.
3. Live scanning works end to end on the new schedule (§8) in dry-run before the 2026-27 opener
   (2026-10-21).

Out of scope (pinned): sportsbook prop lines (The Odds API) and breaking-news feeds.

## 2. Sources verified (2026-10-14)

| # | Source | Endpoint | History |
|---|---|---|---|
| 2 | Confirmed starters + Inactive list | `https://stats.nba.com/js/data/leaders/00_daily_lineups_YYYYMMDD.json` | 2023-24, 2025-26 (2024-25 files empty). Per-player `timestamp` is UTC; confirmations land ~62–72 min before tip (checked 2026-03-10) |
| 2 | Box-score starters (all seasons) | nba_api `LeagueDashPlayerStats(starter_bench_nullable="Starters", date_from=date_to=D)` | all seasons |
| 3 | Per-game rebound tracking | nba_api `LeagueDashPtStats(pt_measure_type="Rebounding", per_mode="Totals", date_from=date_to=D)` | all seasons; ~170 calls/season |
| 4 | Opponent absences | existing `injury_reports` table + vacuum code | as Tier 4 |
| 5 | Kalshi game markets | `KXNBATOTAL` (14,787 settled), `KXNBASPREAD` (16,937) via `/historical/markets` + candles | Oct 2025 – Jun 2026 only |
| 6 | Travel / schedule load | existing `schedule` table + static arena table | all seasons |

## 3. Decision time

New setting `DECISION_LEAD_MIN` (default 30). Historical features use only information published at
or before `tip − DECISION_LEAD_MIN`:

- injury reports: latest report ≤ cutoff (re-run `injury-backfill --lead-min 30` for 2025-26; the
  current table was built at 60);
- lineups: a game's lineup counts as confirmed only if every starter's `timestamp` ≤ cutoff;
- game markets: last hourly candle mid ending ≤ cutoff.

Live features use what is available at scan time. Scans run between T-60 and T-5 (§8), so 30 is the
centre of the live window.

## 4. Feature groups

Groups are enabled by one setting, `FEATURE_GROUPS` (comma-separated, default empty = all **off**
until the ablation passes); each has a column list in `feature_store.py`, and `MODEL_FEATURES` is the
base list plus enabled groups. `FEATURE_VERSION` includes the enabled group names and the decision
lead so PIT caches never mix sets. Models record the feature list they were fit on; live preflight
refuses a model whose list differs from the enabled one.

### 4.1 `lineup` (source #2) — `lineups.py`

Tables: `starters` (GAME_DATE, PLAYER_ID, TEAM_ID; from `LeagueDashPlayerStats`),
`lineups` (game_id, team_id, player_id, lineup_status, roster_status, position, ts_utc; from the JSON).

Features:
- `starts_tonight` — 1 if listed starter. History: lineup JSON confirmed ≤ cutoff where available,
  else box-score starters. Live: lineup JSON (scan refuses the game if not confirmed, §8).
- `start_rate_roll` — share of the player's last 10 games started (state after game, as-of join).
- `role_change` = `starts_tonight − start_rate_roll`.
- `team_usual_starters_out` — count of the team's five most-frequent recent starters (last 10 team
  games) not starting tonight.

Vacuum (separate group `inactive`, so the ablation scores it on its own): players with `roster_status == "Inactive"` and timestamp ≤ cutoff (history) or listed
Inactive now (live) are added to the absent set used by `teammate_vacuum`, alongside report Outs.

Measurement (reported in SYSTEM.md, not a gate): on 2025-26, agreement of JSON-confirmed starters vs
box-score starters, and how much of the 36% hindsight vacuum gap the Inactive list closes.

### 4.2 `tracking` (source #3) — `tracking_daily.py`

Table: `tracking_daily` (GAME_DATE, PLAYER_ID, TEAM_ID, MIN, REB_CHANCES, OREB_CHANCES,
DREB_CHANCES, REB_CONTEST, REB_CHANCE_DEFER, AVG_REB_DIST), unique on (PLAYER_ID, GAME_DATE).

Features (state after game, rolling 10 + EWM like player form): `reb_chances_roll`,
`reb_chances_ewm`, `reb_chances36_roll`, `reb_conv_roll` (REB/REB_CHANCES), `reb_contest_pct_roll`,
`reb_defer_rate_roll`, `oreb_chance_share_roll`. Prior-season tracking features stay; the ablation
reports whether they still carry weight.

### 4.3 `opp_vacuum` (source #4) — `feature_store.py`

`opp_vacuum_reb`, `opp_vacuum_min`, `opp_vacuum_n`: the existing vacuum frame (same absence rules)
merged on (OPP_TEAM_ID, GAME_ID). Zero-filled like the own-team vacuum.

### 4.4 `game_market` (source #5) — `game_markets.py`

Tables: `kalshi_game_markets` (ticker, series, event_ticker, game_date, teams_code, line, side team
for spreads, result, open/close), `kalshi_game_candles` (ticker, end_ts, yes_bid, yes_ask), with a
done-table for resumable backfill (same pattern as `kalshi_history.backfill_candles`).

Per game at the cutoff: for the total ladder, mids P(total > L) per line, monotone-smoothed
(isotonic), interpolated to 0.5 → `mkt_total`. Same for spreads on the favourite side →
`mkt_spread_abs`. Requires ≥3 ladder lines with two-sided books; else NaN.

Features: `mkt_total`, `mkt_spread_abs`, `mkt_total_resid` (= `mkt_total` − `exp_total`, where
`exp_total` = (own `t_pts` + opp `t_pts_allowed`)/2 + (opp `t_pts` + own `t_pts_allowed`)/2, from two
new rolling team-state columns). NaN before Oct 2025 (XGB handles NaN; `_disp_X` untouched). Live: current
books for the event via the Kalshi client.

### 4.5 `travel` (source #6) — `travel.py`

Static `ARENAS` dict: tricode → (lat, lon, IANA tz, altitude_m). Venue = home team's arena.

Features: `travel_km` (great-circle, previous game venue → tonight's; 0 on season's first game),
`tz_shift` (hours, signed), `high_altitude_venue` (venue altitude > 1,000 m: DEN, UTA),
`games_last_7` (team games in the 7 days before tonight), `opp_rest_days`.

## 5. Data flow

```
nightly (yesterday): etl → injury-backfill → starters/lineups backfill → tracking-daily backfill
                     → game-markets backfill (markets + candles)
build_feature_table: gamelogs + new tables → groups (flagged) → as-of joins strictly before date
scan (live):         slate → lineups JSON + injury report + game-market books → extra_rows features
```

New CLI commands in `pipeline/commands.py`: `lineups-backfill`, `tracking-backfill`,
`game-markets-backfill` (each with `--start/--end`, idempotent natural-key upserts, and the ETL
uniqueness assertion).

## 6. Error handling

- Backfills: retry with backoff (reuse `data_engine._call` / `kalshi_history._get`); a failed date
  is logged and skipped, never written as empty; re-runs fill gaps.
- Live, enabled group with missing data:
  - `lineup`, `inactive`: game skipped until both teams confirmed (§8).
  - `tracking`, `travel`, `opp_vacuum`: computed from stored history; no live fetch.
  - `game_market`: NaN if fewer than 3 usable ladder lines (same as history) — not fail-closed,
    since history has the same NaN path.
- Preflight: if an enabled group's table has no rows for the last completed game date, warn
  (tracking/starters/game markets lag a day) and refuse live after 3 days stale.

## 7. Ablation and ship gate

`bakeoff.py --groups <set>` runs walk-forward c7 for: base, base + each group, all groups.
Report line LL, NLL, ECE and date-clustered bootstrap ΔLL CIs vs base. `game_market` is scored only
on test blocks from Dec 2025 on (enough training rows), and reported separately.

Then `model-vs-market` on 2025-26 full slates for base vs the winning set. Ship rule per §1. Winning
groups: flag on in `.env`, retrain, `refit-blend`, re-run the Apr 1 – Jun 13 2026 holdout backtest,
update SYSTEM.md §3–§5.

## 8. Live scan schedule

- New setting `SCAN_TIP_WINDOW_MIN = "5,60"`: scan only games tipping in (5, 60] minutes; replaces
  `SCAN_WITHIN_HOURS` for scheduled scans (kept for manual runs).
- With `lineup` or `inactive` enabled, a game is scanned only when both teams' lineups are confirmed in
  the JSON (an unreadable JSON confirms nothing).
- Dry-run scan emails only when a scan journals new paper orders (the 10-minute cadence would
  otherwise send ~70 a day).
- Cron: `scan` every 10 minutes 11:00–23:50 ET (weekend and playoff
  tips start at noon; a scan with no game in the window exits early) (replaces hourly `:05 14-20`). Snapshots unchanged.
  Order `expiration_time` (tip − 60 s) and CLV `close` mark are unchanged.
- Each scan re-reads the injury report and lineups JSON.

## 9. Testing

- Parsers: lineups JSON (starters, Inactive, UTC timestamps, unconfirmed games), tracking frame,
  game-market ladder → median (monotone smoothing, <3 lines → NaN), arena distance/tz.
- PIT: extend `tests/test_feature_store_pit.py` — scrambling outcomes on/after D (including the new
  tables' rows dated ≥ D) must not move any feature dated ≤ D; lineup/candle rows after the cutoff
  are ignored.
- Live-equals-history: a historical slate rebuilt through the live path with the stored lineup /
  market snapshot reproduces history features.
- Scan window and lineup gate: unit tests on the filter.
- Full suite green (270 tests today).

## 10. Rollout

1. Build sources + backfills (background), group flags off — production behaviour unchanged.
2. Ablation, measurements, decide groups.
3. Enable winners, retrain, refit blend, backtest, update SYSTEM.md.
4. Switch cron to the 10-minute scan schedule; dry-run + paper journal from the opener.
