# SYSTEM.md — How the NBA rebounds engine works, and why it changed on 2026-10-08

Read this before touching modelling, edge, sizing or reporting code. Operational
how-to lives in `WORKFLOW.md`; this file is the *why*.

---

## 1. What this system is

An automated pipeline that trades **Kalshi NBA player-rebound markets**
(`KXNBAREB`, e.g. "Victor Wembanyama: 8+ rebounds" = P(REB > 7.5)).

This repo was forked from `mlb_tb_line`'s first commit (May 2026) and then sat
still while the MLB engine went through 31 commits of hard-won fixes. On
2026-10-08 everything transferable was ported back, the NBA-specific leaks were
fixed, and the model was re-chosen by a walk-forward bake-off. MLB's
`SYSTEM.md` §2–§9 is required background: most of the mechanisms here exist
because of incidents documented there.

### Key modules

| Module | Role |
|---|---|
| `data_engine.py` | nba_api ETL (game logs, prior-season tracking, schedule) with natural-key dedupe; slate/start-window helpers |
| `feature_store.py` | point-in-time features (as-of joins on strictly earlier games) |
| `model_zoo.py`, `model.py` | candidate rebound PMF models; production wrapper with `trained_on` stamps |
| `bakeoff.py` | walk-forward PIT model comparison |
| `kalshi_history.py`, `market_eval.py` | 2025-26 Kalshi book backfill; model vs market on full slates |
| `slate.py`, `identity_bridge.py`, `injury_feed.py` | tonight's markets → player/game → PIT features |
| `probability_engine.py` | PMF → P(over)/P(under), coherent |
| `calibration.py`, `calibrate_preflight.py` | OOF + fill isotonic/Platt calibrators, staleness and model-mismatch gates |
| `market_blend.py` | capped logit shrinkage toward the market mid (one monotone curve) |
| `fees.py`, `execution_engine.py`, `edge_detector.py` | fee-aware edge vs maker/taker limit, Kelly sizing, gates |
| `risk_manager.py`, `journal_risk.py` | daily caps, kill switch |
| `trade_journal.py`, `reconcile_fills.py` | JSONL journal; fills are authoritative |
| `model_vs_market.py`, `backtest.py` | full-slate scoring and stack replay, PIT by default |
| `pipeline/commands.py` | CLI (`run_pipeline.py <cmd>`) |

---

## 2. What was wrong with the old NBA engine

1. **Look-ahead leakage.** Opponent shooting, pace and rebound-tracking stats
   were full-season aggregates joined on the *same* season, so November games
   saw April numbers. (The tracking docstring even said "prior season".)
2. **Train/serve skew.** `team_reb_vacuum` was always 0 in training but
   non-zero at inference; `vegas_spread_abs` was a constant 5.0.
3. **Wrong distribution.** XGB trained with an MAE objective predicts a
   *median*, which was then used as the NB *mean* with one global residual
   variance for every player. Bake-off: bias −0.17 rebounds/game, line
   log-loss 0.4274 vs 0.4199 now.
4. **Scan scored the wrong row.** It took each player's *last historical*
   feature row — last game's opponent, rest, home/away — forced
   `is_playoffs=1`, matched names with exact lowercase (every accented name
   missed), and for unmatched players invented a "model" probability from the
   market price, which can only ever manufacture edge.
5. Everything MLB later fixed: no fees, no market blend, taker-only, no fill
   reconciliation, no calibration, no CLV, no risk desk.

## 3. Data and features (point-in-time by construction)

Every feature is a *state after game g* (player state, team state) attached to
a target row with `merge_asof(..., allow_exact_matches=False)` — strictly
earlier dates. Tonight's scan rows go through the same function
(`build_feature_table(extra_rows=..., absent_override=...)`).

* Matchup features come from rolling team box scores (own/opponent missed FGs,
  pace, OREB%/DREB%, 3PA rate, expected |margin|) — no season aggregates.
* Rebound tracking is joined from the **previous** season only.
* Teammate vacuum: rotation teammates (≥15 trailing MPG, seen within 21 days,
  still on the team) who don't play. History: no box-score row. Live: injury
  feed OUT list + `data/absent_<date>.json`. Late scratches after the scan are
  the remaining gap.

Guards (`tests/test_feature_store_pit.py`): scrambling every outcome on/after
date D must not move any feature dated ≤ D; live-path rows must equal history
rows. Both were verified to fail on deliberately injected leaks (same-day
as-of match, next-game shift). On the real 2026-03-10 slate the live path
reproduced history PMFs exactly (max diff 6e-16).

ETL dedupes on natural keys and replaces per season — the MLB incident in its
§9 (silent 4x duplicated training rows) can't happen here, and the write path
asserts uniqueness.

## 4. Model choice: walk-forward bake-off

`bakeoff.py`: every candidate is retrained on all rows before each test block
(monthly in 2025-26, bi-monthly before) and scored on 62,763 player-games
(2023-24..2025-26). Primary metric: log-loss of P(REB > k) at half-integer
lines within 6 of the player's average (the lines Kalshi lists).

| Model | line LL | NLL | ECE | bias |
|---|---|---|---|---|
| **c7: pool of c4 + c5 (production)** | **0.4199** | **2.2400** | 0.0020 | +0.02 |
| c5: minutes × per-minute rate, NB | 0.4206 | 2.2418 | 0.0041 | +0.03 |
| c4: XGB mean + heteroscedastic NB | 0.4207 | 2.2433 | 0.0025 | +0.02 |
| c3: XGB mean + global NB | 0.4217 | 2.2494 | 0.0025 | +0.01 |
| c6: LightGBM multiclass PMF | 0.4230 | 2.2629 | 0.0041 | −0.03 |
| c2: XGB Poisson | 0.4265 | 2.2767 | 0.0254 | +0.02 |
| c1: legacy (MAE + global-variance NB) | 0.4274 | 2.2913 | 0.0183 | −0.17 |
| c0: naive EWM + NB | 0.4294 | 2.2691 | 0.0038 | +0.03 |

c7 is best in every season; date-clustered bootstrap 95% CIs exclude zero vs
c5 (−0.0007), c4 (−0.0008) and legacy (−0.0074). c4 vs c5 is a tie — they
fail differently (dispersion model vs minutes model), which is why pooling
helps. Poisson is badly under-dispersed (ECE 0.025).

## 5. Model vs market (2025-26 Kalshi history)

MLB's central finding was that the market beat the model and that unshrunk
"edges" were mostly model error. The first honest test here: score PIT
predictions against Kalshi's pregame mid (last hourly candle ≥60 min before
tip, two-sided, spread ≤ 0.25) on every settled 2025-26 market.

**Raw model vs book** (`market_eval.py`, 17,034 markets, Nov 2025 – Jun 2026,
bake-off monthly PIT predictions, uncalibrated):

| Model | log-loss | Δ vs market (0.5186) | fitted w |
|---|---|---|---|
| c7 ensemble | 0.5192 | +0.0006 | 0.47 |
| c4 NB-hetero | 0.5196 | +0.0010 | 0.46 |
| c5 minutes × rate | 0.5199 | +0.0013 | 0.44 |
| c1 legacy | 0.5254 | +0.0068 | 0.28 |
| c0 naive | 0.5253 | +0.0067 | 0.27 |

The new model roughly **ties** the market; the old one lost clearly. By
disagreement bucket the model is at or slightly ahead of the book below 0.15
and badly behind at ≥0.15 (ΔLL +0.057, n=539) — the same winner's-curse
signature MLB found. Per-bucket blend weights exist for exactly this.

**Blend weights, 2026-10-08 (superseded).** Per-bucket weights (Nov–Jun: global 0.40;
<0.05 0.73, 0.05–0.10 0.46, 0.10–0.15 0.44, ≥0.15 0.14) were non-monotone in p_model
and a weekly 30-day refit dropped any bucket under 100 rows back to the global weight
(≥0.15 had ~80 rows a month, so it reverted to ~0.40 most Sundays).

**Calibration and blend, re-tested 2026-10-13** (all 16,137 2025-26 full-slate PIT
scores, `models/` untouched by the test):

* *OOF isotonic calibrator* — forward log-loss vs raw probabilities, month by month:
  worse in 6 of 8 months, overall 0.53014 vs 0.52942. `USE_OOF_CALIBRATION` is now off;
  PIT scoring, refits and backtests follow the same setting.
* *Blend*, weekly walk-forward Jan–Jun 2026 (expanding window, raw input, n=14,365):

| Blend | all markets | ≥0.15 disagreement (n=486) |
|---|---|---|
| market only | 0.52776 | 0.63405 |
| plain logit blend | 0.52609 | 0.65146 |
| 4 buckets (old production) | 0.52612 | 0.64008 |
| **capped, cap 0.3, w fit** (new) | **0.52584** | 0.63702 |
| capped, cap 0.2, w fit | 0.52600 | 0.63527 |

  Every blend beats the market overall by ~0.0019, but only a cap keeps the model
  from losing badly where it disagrees most — which is where trades come from. A cap
  fit jointly with w drifts to "no cap" (overall log-loss barely weights the tail),
  so `BLEND_CAP` is a fixed risk setting and only w is refit weekly, over a season-long
  window. Both cap values were picked on this same season: in-sample, and the model
  still loses to the market at ≥0.15 under any blend.

**Out-of-sample stack backtest, superseded.** The 2026-10-08 figure (82 trades,
ROI +24.6%, CI +7% to +41%) used blend weights swapped in by hand, a spread-based fill
heuristic that filled at least 25% of every order whether or not anything traded, and
books with no tip cutoff. Do not quote it.

**Holdout rerun, 2026-10-12** (Apr 1 – Jun 13 2026; walk-forward weekly blend weights
fit only on earlier full-slate scores since 2025-11-01; pregame books only; fees,
maker limits, Kelly, live gates incl. bid > 0 and spread ≤ 0.08):

| Variant | Signals | Filled | ROI | Day-bootstrap 95% CI | CLV / contract |
|---|---|---|---|---|---|
| Fills from trades/quotes before tip (default) | 178 | 100 (56%) | +16.5% | −6.6% to +37.9% | −0.0026 |
| Everything fills (upper bound) | 178 | 178 | +17.1% | +1.1% to +33.6% | +0.0054 |
| Teammate-absence features zeroed | 27 | 9 | −23.4% | −71% to +15% | −0.0062 |
| **Tier 3 stack** (raw probs, capped blend cap 0.3; fills from trades) | 106 | 68 (64%) | +19.1% | −11.8% to +51.2% | −0.0003 |

The Tier 3 row (2026-10-13) trades less and its fills are calibrated (win rate 51.5% vs
mean model p 0.503, against 54% vs 0.565 before) with CLV ≈ 0 instead of negative;
the CI still spans zero.

Full-slate model vs market (Nov 18 2025 – Jun 13 2026, 16,137 markets, PIT):
ΔLL +0.0021 (model slightly worse), paired day-bootstrap CI −0.0014 to +0.0056 —
**not distinguishable from zero**. The ≥0.15 disagreement bucket is still clearly
worse (ΔLL +0.065, n=557).

Reading:

* With realistic fills the edge is **not statistically established**: the CI spans
  zero, and CLV on filled orders is negative. Resting orders fill when the price moves
  against them (adverse selection), and the fills that do happen win 54% against a
  model mean of 0.565.
* Most signals depend on the teammate-absence features. History knows every late
  scratch; the live scan only knows ESPN's OUT list. Zeroing those features is a
  pessimistic bound (it creates the opposite skew), but the drop from 178 to 27
  signals means the edge is concentrated exactly where the backtest has hindsight.
* Before small live: run the dry-run paper journal (`report-range --paper`) and judge
  it on net P&L and close CLV; rebuild historical absences from time-stamped injury
  reports (Tier 4) before trusting any backtest that uses them.

## 6. Trading stack (ported from MLB, see MLB SYSTEM.md §3)

Probability plumbing per market: PMF → raw P(over) → calibrate in P(over) space
(Platt if enabled → fill-segmented isotonic of the current model → OOF isotonic if
enabled → raw; support-clamped, ±0.12 cap) → capped blend toward the side's market
mid (logit offset ≤ `BLEND_CAP`)
→ maker-aware limit → `edge = p_blend − limit − fee` → gates (min p, tail
threshold, spread, realistic ask, blocked segments, VPIN) → fractional Kelly
on net-of-fee odds → portfolio caps → risk desk.

Ported **mechanisms**, not MLB **values**. Defaults that came from MLB fill
evidence are neutral here until NBA evidence exists: `BLOCKED_SEGMENTS=""`
(MLB: `1.5:no`), `MAX_YES_LINE=99` (MLB: 2.0), `RISKY_BAND_KELLY_MULT=1.0`
(MLB: 0.4). Rule from MLB §8 still applies: never raise a fitted blend weight
or its floor by hand to "get volume back".

## 7. Feedback loops

Same as MLB §5, with two changes:

* `model-vs-market`, `refit-blend` and `backtest` use **point-in-time weekly
  models by default** (cached in `models/pit_cache/`). MLB scores with the
  saved model, which has seen the days it scores — look-ahead that inflates
  the fitted `w` the weekly refit then trades on.
* Books come from live snapshots when present, else the Kalshi historical
  candles, so the whole 2025-26 season is replayable.

`backtest.py` settles each side at its own contract price and nets fees. (MLB's
`_pnl` treats a NO signal's NO-price limit as a YES price — inverted P&L/CLV
on every NO trade — and ignores fees.)

## 8. Ops

Cron hours are system-local (PT); the `TZ=America/New_York` prefix only sets
the job's environment. Nightly runs under launchd (fires missed runs on wake;
cron doesn't). No power/wake-alarm scripts: macOS keeps a single repeat wake,
owned by `mlb_tb_line`. The stuck-scan watchdog only reaps this repo's
processes.

## 9. Known gaps

* Retraining runs Sundays inside `nightly` (MLB still has none — its fit takes ~11h).
  It invalidates the fill calibrator, so set `AUTO_RETRAIN=0` before turning on
  `REQUIRE_FILL_CALIB_FOR_LIVE`, then retrain by hand and recalibrate together.
* Live rosters (PlayerIndex) are a snapshot of now: same-day trades show up only
  once stats.nba.com updates them; until then the player is skipped or on his old team.
* Injury feed is unofficial (ESPN JSON) and fails open (vacuum = 0, logged).
* No hyper-parameter tuning was ported (`tune`); model_zoo params are hand-set.
* Fill-based calibrator and segment blocks need NBA fills that don't exist yet.
