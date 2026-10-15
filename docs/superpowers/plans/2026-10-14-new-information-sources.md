# New Information Sources Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add five free information sources (confirmed lineups + Inactive list, per-game rebound tracking, opponent absences, Kalshi game totals/spreads, travel/schedule load) as flag-gated feature groups, and let a walk-forward ablation decide which ship.

**Architecture:** Each source gets one module that owns fetch → store (SQLite, resumable per-date backfill) → a pure feature function. `feature_store.build_feature_table(groups=...)` attaches each enabled group with the same strictly-before as-of discipline as today; `MODEL_FEATURES` = base list + enabled groups (`FEATURE_GROUPS` env). Models remember the feature list they were fit on. Live scans move to a 10-minute cadence, trade games tipping in (5, 60] minutes, and only once both lineups are confirmed.

**Tech Stack:** Python 3, pandas, numpy, scikit-learn (IsotonicRegression), nba_api, requests, SQLAlchemy/SQLite, XGBoost/LightGBM, typer CLI, pytest. Venv: `.venv/bin/python`.

**Spec:** `docs/superpowers/specs/2026-10-14-new-information-sources-design.md`

## Global Constraints

- Every new historical feature uses only information published at or before `tip − DECISION_LEAD_MIN` (default **30** minutes); state-after-game features join with `merge_asof(..., allow_exact_matches=False)` (strictly earlier dates).
- All groups default **off**: `FEATURE_GROUPS` env, comma-separated, default `""`. Group names: `travel`, `opp_vacuum`, `tracking`, `lineup`, `inactive`, `game_market`.
- With no groups enabled, every existing feature value and the existing 270 tests must be unchanged (except `FEATURE_VERSION` string and the vacuum cutoff moving from 60 to `DECISION_LEAD_MIN`).
- Live scan window `SCAN_TIP_WINDOW_MIN = "5,60"`; with `lineup` or `inactive` enabled a game is scanned only when both teams' lineups are `Confirmed`.
- Kalshi public endpoints are rate-limited at 5 req/s (`kalshi_history._limiter`); nba_api calls go through `data_engine._call` (0.65 s spacing, retries).
- Backfills never mark a date done on a fetch failure, and (except lineups) never on an empty result for a date that had games.
- New tables are written through `data_engine._assert_unique` on their natural key.
- Commits: conventional (`feat:`/`fix:`/`test:`/`docs:`), message body normal prose, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Work on branch `feat/new-info-sources`; the user asks before push/merge.
- Run tests with `.venv/bin/python -m pytest -q`; lint changed files with `.venv/bin/python -m pyflakes <files>`.

## Review Focus

1. **Opponent with no markets tonight (live):** opponent vacuum must still be computed when only one team has Kalshi markets — pinned by `test_opp_vacuum_live_parity_one_sided_slate` (Task 4).
2. **Lineup JSON lists a team with fewer than 5 starters or an `Expected` status:** game is unconfirmed and skipped, never priced with `starts_tonight = 0` — pinned by `test_lineup_states_unconfirmed_cases` (Task 7) and `test_slate_skips_unconfirmed_games` (Task 10).
3. **Game-market ladder with one-sided books or median outside the ladder:** returns NaN, never an extrapolated total — pinned by `test_ladder_median_edge_cases` (Task 8).
4. **Backfill hits a future or today date / stats.nba.com returns an empty frame:** not marked done, re-fetched next run — pinned by `test_run_backfill_skips_today_and_empty` (Task 2).
5. **Old pickled model (no `features` attribute) after groups are enabled:** keeps predicting on the base list, and live preflight refuses because meta features differ — pinned by `test_legacy_model_uses_base_features` (Task 1) and `test_preflight_flags_feature_mismatch` (Task 1).

---

### Task 1: Feature-group plumbing, model feature lists, ablation tooling

**Files:**
- Modify: `config.py` (after `PREGAME_ABSENCES`, ~line 156)
- Modify: `feature_store.py:40-61` (constants), `feature_store.py:263-375` (`build_feature_table` signature/return)
- Modify: `model_zoo.py` (every `MODEL_FEATURES` use; `Model` base class)
- Modify: `calibrate_preflight.py` (`check_calibrate_preflight`, new `_model_features`)
- Modify: `bakeoff.py` (`FEAT_CACHE`, `run`, new `--tag`, new `ablation`, `date_bootstrap`, `line_ll_rows`)
- Modify: `tests/conftest.py` (stub `_model_features`)
- Test: `tests/test_feature_groups.py`

**Interfaces:**
- Produces: `config.FEATURE_GROUPS: list[str]`, `config.DECISION_LEAD_MIN: int`; `feature_store.BASE_FEATURES`, `feature_store.FEATURE_GROUPS: dict[str, list[str]]`, `feature_store.ENABLED_GROUPS: list[str]`, `feature_store.model_features(groups) -> list[str]`, `feature_store.feature_version(groups, lead_min) -> str`, `feature_store.MODEL_FEATURES`, `feature_store.FEATURE_VERSION`; `build_feature_table(..., groups: list[str] | None = None)`; `model_zoo.Model.feats` property; every candidate `fit(df, features=None)`; `bakeoff.date_bootstrap(diff, dates, n_boot=2000, seed=0, alpha=0.05) -> (point, lo, hi)`, `bakeoff.line_ll_rows(pmf, y, lf) -> np.ndarray`, `bakeoff.ablation(model, base_tag="base", since=None) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_feature_groups.py
import numpy as np
import pandas as pd
import pytest

import feature_store as fs
from tests.test_feature_store_pit import _synthetic_gamelogs, _tracking


def test_base_features_unchanged_and_groups_append_sorted():
    assert fs.model_features([]) == fs.BASE_FEATURES
    got = fs.model_features(["travel", "opp_vacuum"])
    assert got == fs.BASE_FEATURES + fs.FEATURE_GROUPS["opp_vacuum"] + fs.FEATURE_GROUPS["travel"]


def test_unknown_group_raises():
    with pytest.raises(ValueError, match="Unknown FEATURE_GROUPS"):
        fs.model_features(["nope"])


def test_feature_version_names_groups_and_lead():
    assert fs.feature_version([], 30) == "v3-lead30-base"
    assert fs.feature_version(["travel", "lineup"], 30) == "v3-lead30-lineup+travel"


def test_build_feature_table_returns_requested_group_columns():
    gl = _synthetic_gamelogs(n_days=20)
    df = fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False, groups=["opp_vacuum"])
    for c in fs.FEATURE_GROUPS["opp_vacuum"]:
        assert c in df.columns
    assert "travel_km" not in df.columns


def _train_frame():
    gl = _synthetic_gamelogs(n_days=60)
    df = fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False)
    return df[df["REB"].notna() & df["min_roll"].notna()].reset_index(drop=True)


def test_model_remembers_feature_list():
    from model_zoo import XGBNBHetero

    df = _train_frame()
    m = XGBNBHetero().fit(df, features=fs.BASE_FEATURES)
    assert m.feats == fs.BASE_FEATURES
    p = m.pmf(df.assign(extra_col=1.0))  # extra columns are ignored
    assert p.shape == (len(df), 21)


def test_legacy_model_uses_base_features():
    from model_zoo import XGBNBHetero

    df = _train_frame()
    m = XGBNBHetero().fit(df, features=fs.BASE_FEATURES)
    del m.__dict__["features"]  # pickles from before this change have no attribute
    assert m.feats == fs.BASE_FEATURES
    assert m.pmf(df).shape[0] == len(df)


def test_preflight_flags_feature_mismatch(monkeypatch):
    import calibrate_preflight as cp

    monkeypatch.setattr(cp, "_model_features", lambda: fs.BASE_FEATURES[:-1])
    res = cp.check_calibrate_preflight(live=True)
    assert any("feature set" in e for e in res.errors)


def test_date_bootstrap_constant_diff_has_degenerate_ci():
    from bakeoff import date_bootstrap

    dates = np.repeat(pd.date_range("2026-01-01", periods=10).values, 5)
    point, lo, hi = date_bootstrap(np.full(50, -0.01), dates, n_boot=200)
    assert point == pytest.approx(-0.01) and lo == pytest.approx(-0.01) and hi == pytest.approx(-0.01)


def test_line_ll_rows_matches_score():
    from bakeoff import line_frame, line_ll_rows, score

    rng = np.random.default_rng(0)
    pmf = rng.dirichlet(np.ones(21), size=30)
    y = rng.integers(0, 15, size=30).astype(float)
    feat = pd.DataFrame({"reb_roll": rng.uniform(2, 12, 30), "reb_ewm": np.nan})
    lf = line_frame(feat)
    assert line_ll_rows(pmf, y, lf).mean() == pytest.approx(score(pmf, y, lf)["ll_line"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_feature_groups.py -q`
Expected: FAIL — `AttributeError: module 'feature_store' has no attribute 'model_features'` (and import errors for `date_bootstrap`, `line_ll_rows`).

- [ ] **Step 3: Add config settings**

In `config.py`, directly after the `PREGAME_ABSENCES = ...` line:

```python
# Optional feature groups (feature_store.FEATURE_GROUPS), comma-separated, e.g.
# "travel,opp_vacuum". All off until the walk-forward ablation (bakeoff.py ablation)
# shows a group helps; the model records the list it was trained on and live
# preflight refuses a mismatch.
FEATURE_GROUPS = [g.strip() for g in os.getenv("FEATURE_GROUPS", "").split(",") if g.strip()]
# Point-in-time decision time for historical features: only information published at
# or before tip − DECISION_LEAD_MIN (injury reports, lineups, game-market candles).
# Live scans run between T-60 and T-5, so 30 is the middle of the live window.
DECISION_LEAD_MIN = int(os.getenv("DECISION_LEAD_MIN", "30"))
```

- [ ] **Step 4: Feature registry in `feature_store.py`**

Replace the `from config import ...` line, the `FEATURE_VERSION = ...` line and the `MODEL_FEATURES = [...]` block with:

```python
from config import DECISION_LEAD_MIN, FEATURE_GROUPS as _CONFIG_GROUPS, MIN_GAMES, PREGAME_ABSENCES, ROLLING_WINDOW
```

```python
BASE_FEATURES = [
    # player form
    "reb_roll", "reb_ewm", "reb_std_roll", "oreb_roll", "dreb_roll",
    "reb36_roll", "reb36_ewm", "min_roll", "min_ewm", "min_std_roll",
    "reb_share_roll", "reb_season_avg", "games_in_season", "games_played",
    "rest_days", "is_b2b", "is_home", "is_playoffs",
    # prior season
    "prev_reb_pg", "prev_reb_chances", "prev_reb_chance_pct_adj",
    "prev_reb_contest_pct", "prev_reb_chance_defer", "prev_avg_reb_dist",
    # matchup
    "exp_opp_missed", "exp_own_missed", "exp_pace", "opp_oreb_pct", "opp_dreb_pct",
    "opp_3pa_rate", "exp_margin_abs",
    # teammate vacuum
    "vacuum_reb", "vacuum_min", "vacuum_n",
]

# Optional groups (FEATURE_GROUPS env). See docs/superpowers/specs/2026-10-14-new-information-sources-design.md.
FEATURE_GROUPS: dict[str, list[str]] = {
    "travel": ["travel_km", "tz_shift", "high_altitude_venue", "games_last_7", "opp_rest_days"],
    "opp_vacuum": ["opp_vacuum_reb", "opp_vacuum_min", "opp_vacuum_n"],
    "tracking": ["reb_chances_roll", "reb_chances_ewm", "reb_chances36_roll", "reb_conv_roll",
                 "reb_contest_pct_roll", "reb_defer_rate_roll", "oreb_chance_share_roll"],
    "lineup": ["starts_tonight", "start_rate_roll", "role_change", "team_usual_starters_out"],
    "game_market": ["mkt_total", "mkt_spread_abs", "mkt_total_resid"],
    # Modifier, no columns: adds the pre-tip Inactive list (lineups JSON) to vacuum absences.
    "inactive": [],
}


def model_features(groups) -> list[str]:
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown FEATURE_GROUPS: {sorted(unknown)}")
    return BASE_FEATURES + [c for g in sorted(set(groups)) for c in FEATURE_GROUPS[g]]


def feature_version(groups, lead_min: int) -> str:
    """PIT model caches and feature caches are keyed by this; bump the prefix when semantics change."""
    return f"v3-lead{lead_min}-" + ("+".join(sorted(set(groups))) or "base")


ENABLED_GROUPS = sorted(set(_CONFIG_GROUPS))
MODEL_FEATURES = model_features(ENABLED_GROUPS)
FEATURE_VERSION = feature_version(ENABLED_GROUPS, DECISION_LEAD_MIN)
```

In `build_feature_table`, add the parameter `groups: Optional[list[str]] = None,` (last), document it in the docstring (`groups : optional feature groups to attach (default: ENABLED_GROUPS)`), add as the first body line:

```python
    groups = ENABLED_GROUPS if groups is None else sorted(set(groups))
    feature_cols = model_features(groups)
```

and change the final return to:

```python
    return df[ID_COLS + ["_live", TARGET, "MIN", "last_game_date"] + feature_cols].reset_index(drop=True)
```

To make `test_build_feature_table_returns_requested_group_columns` pass before Task 4 lands, add a temporary stub right before `if apply_filters:` — **Task 3/4/5/6/8 replace it group by group**:

```python
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
```

(Keep this loop permanently: it guarantees an enabled group whose source table is missing yields NaN columns instead of a KeyError.)

- [ ] **Step 5: Models remember their feature list (`model_zoo.py`)**

Change the import to `from feature_store import BASE_FEATURES, MODEL_FEATURES`. In `_xgb_fit`, add a `feats: list[str]` parameter (after `full`) and replace the four `MODEL_FEATURES` uses with `feats`:

```python
def _xgb_fit(objective: str, tr: pd.DataFrame, ca: pd.DataFrame, full: pd.DataFrame, feats: list[str],
             target: str = "REB", margin: str | None = None, **over) -> tuple[xgb.XGBRegressor, xgb.XGBRegressor]:
    ...
    m_ho.fit(tr[feats], tr[target], eval_set=[(ca[feats], ca[target])], verbose=False, **kw_tr, **kw_ca)
    ...
    m_full.fit(full[feats], full[target], verbose=False,
               **({"base_margin": full[margin].values} if margin else {}))
```

Base class:

```python
class Model:
    name = "base"

    def _use(self, features) -> list[str]:
        self.features = list(MODEL_FEATURES if features is None else features)
        return self.features

    @property
    def feats(self) -> list[str]:
        # Pickles from before feature groups have no attribute; they were fit on the base list.
        return self.__dict__.get("features", BASE_FEATURES)

    def fit(self, df: pd.DataFrame, features=None) -> "Model":
        raise NotImplementedError

    def pmf(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
```

Every candidate's `fit(self, df)` becomes `fit(self, df, features=None)` with `f = self._use(features)` as its first line, uses `f` for training columns, and every `pmf` uses `self.feats`. Concretely:
- `NaiveNB.fit`: add `self._use(features)` (it uses no feature matrix).
- `LegacyMAENB`, `XGBPoisson`, `XGBNB`: `df[MODEL_FEATURES]`/`ca[MODEL_FEATURES]` → `df[f]`/`ca[f]` in fit, `df[self.feats]` in pmf; `_xgb_fit(..., df, f)`.
- `XGBNBHetero.fit`: `ho, self.m = _xgb_fit("count:poisson", tr, ca, df, f)`; `ho.predict(ca[f])`; `pmf`: `self.m.predict(df[self.feats])`.
- `MinutesRate.fit`: `_xgb_fit("reg:squarederror", tr, ca, df, f, target="MIN")`, `ho_m.predict(ca[f])`, `_xgb_fit("count:poisson", pos_tr, pos_ca, pos, f, margin="log_min")`, `ho_r.predict(pos_ca[f], ...)`; `minutes_grid`/`pmf` use `df[self.feats]`.
- `LGBMulticlass`: `tr[f]`, `ca[f]`, `df[f]` in fit; `df[self.feats]` in pmf.
- `Ensemble45.fit`: `f = self._use(features); self.parts = [XGBNBHetero().fit(df, f), MinutesRate().fit(df, f)]`.

Verify none are left: `grep -n "MODEL_FEATURES" model_zoo.py` shows only the import and `_use`.

- [ ] **Step 6: Preflight refuses a feature-set mismatch (`calibrate_preflight.py`)**

Add next to `_model_data_end`:

```python
def _model_features() -> list[str] | None:
    import pickle

    from model import META_PATH

    try:
        with open(META_PATH, "rb") as f:
            return pickle.load(f).get("features")
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return None
```

In `check_calibrate_preflight`, right after the model-lag block:

```python
    from feature_store import ENABLED_GROUPS, MODEL_FEATURES

    trained_feats = _model_features()
    if trained_feats is not None and list(trained_feats) != list(MODEL_FEATURES):
        _flag(f"Model was trained on a different feature set ({len(trained_feats)} features) than "
              f"FEATURE_GROUPS={ENABLED_GROUPS} ({len(MODEL_FEATURES)} features); run train")
```

In `tests/conftest.py` add to the autouse fixture:

```python
    monkeypatch.setattr("calibrate_preflight._model_features", lambda: None)
```

- [ ] **Step 7: Ablation tooling (`bakeoff.py`)**

Replace `FEAT_CACHE = ...` with a function, and key outputs by tag:

```python
def _feat_cache() -> Path:
    from feature_store import FEATURE_VERSION
    return Path(DATA_DIR) / f"features_{FEATURE_VERSION}.pkl"


def load_features(refresh: bool = False) -> pd.DataFrame:
    path = _feat_cache()
    if path.exists() and not refresh:
        return pd.read_pickle(path)
    from feature_store import build_feature_table
    df = build_feature_table()
    df = df[~df["_live"] & df["REB"].notna()].reset_index(drop=True)
    df.to_pickle(path)
    return df
```

`run(models, since, tag="")`: output path `OUT_DIR / (f"{name}__{tag}.npz" if tag else f"{name}.npz")`; the fit line becomes `m = CANDIDATES[name]().fit(train, features=MODEL_FEATURES)` (import `MODEL_FEATURES` from `feature_store` inside `run`). Log `FEATURE_VERSION` at start.

Refactor `_load_pmfs` through a helper and add the ablation pieces:

```python
def _load_npz(f: Path) -> pd.DataFrame:
    z = np.load(f)
    idx = pd.MultiIndex.from_arrays([z["keys"][:, 0].astype(int), z["keys"][:, 1]], names=["PLAYER_ID", "GAME_ID"])
    return pd.DataFrame(z["pmf"], index=idx)


def _load_pmfs() -> dict[str, pd.DataFrame]:
    return {f.stem: _load_npz(f) for f in sorted(OUT_DIR.glob("*.npz"))}


def line_ll_rows(pmf: np.ndarray, y: np.ndarray, lf: pd.DataFrame) -> np.ndarray:
    """Per-(row, line) log-loss of P(REB > line), same clipping as score()."""
    p = np.clip(prob_over(pmf[lf["row"].values], lf["line"].values), 1e-6, 1 - 1e-6)
    o = (y[lf["row"].values] > lf["line"].values).astype(float)
    return -(o * np.log(p) + (1 - o) * np.log(1 - p))


def date_bootstrap(diff: np.ndarray, dates: np.ndarray, n_boot: int = 2000, seed: int = 0,
                   alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of ``diff`` with a CI that resamples whole game dates (players on one slate share news)."""
    s = pd.DataFrame({"d": diff, "date": dates}).groupby("date")["d"].agg(["sum", "count"])
    sums, cnts = s["sum"].to_numpy(), s["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n_boot, len(s)))
    boots = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return (float(sums.sum() / cnts.sum()), float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)))


def ablation(model: str, base_tag: str = "base", since: str | None = None) -> pd.DataFrame:
    """ΔLL (tag − base; negative = tag better) per tagged run of ``model``, on shared rows."""
    feat = load_features().set_index(["PLAYER_ID", "GAME_ID"])
    runs = {f.stem.split("__", 1)[1]: _load_npz(f) for f in sorted(OUT_DIR.glob(f"{model}__*.npz"))}
    base = runs.pop(base_tag)
    rows = []
    for tag, d in runs.items():
        common = base.index.intersection(d.index).intersection(feat.index)
        f = feat.loc[common]
        if since:
            keep = (f["GAME_DATE"] >= pd.Timestamp(since)).to_numpy()
            common, f = common[keep], f[keep]
        y, lf = f["REB"].to_numpy(), line_frame(f)
        ll_b = line_ll_rows(base.loc[common].to_numpy(), y, lf)
        ll_t = line_ll_rows(d.loc[common].to_numpy(), y, lf)
        point, lo, hi = date_bootstrap(ll_t - ll_b, f["GAME_DATE"].to_numpy()[lf["row"].values])
        rows.append({"tag": tag, "rows": len(common), "ll_base": ll_b.mean(), "ll_tag": ll_t.mean(),
                     "d_ll": point, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).sort_values("d_ll")
```

Also make `score()` use `line_ll_rows` for `ll` (keeps one definition). CLI: `choices=["run", "summary", "features", "ablation"]`, add `--tag` (default `""`), `--model` (default `"c7_ens_c4_c5"`), `--base-tag` (default `"base"`), `--since-score` (default `None`); `ablation` prints `ablation(a.model, a.base_tag, a.since_score).round(5).to_string(index=False)`. `run` passes `a.tag`.

- [ ] **Step 8: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_feature_groups.py -q`
Expected: PASS (9 tests).

- [ ] **Step 9: Full suite + lint**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m pyflakes config.py feature_store.py model_zoo.py calibrate_preflight.py bakeoff.py`
Expected: all pass, no pyflakes output. (`pregame_out_for_gamelogs` still uses `lead_min=60` here; Task 9 moves it.)

- [ ] **Step 10: Commit**

```bash
git add config.py feature_store.py model_zoo.py calibrate_preflight.py bakeoff.py tests/conftest.py tests/test_feature_groups.py
git commit -m "feat: flag-gated feature groups, per-model feature lists, ablation tooling"
```

---

### Task 2: Resumable per-date backfill store

**Files:**
- Create: `daily_store.py`
- Modify: `data_engine.py:58-59` (`_get_engine` SQLite busy timeout)
- Test: `tests/test_daily_store.py`

**Interfaces:**
- Consumes: `data_engine._assert_unique(table, key, engine)`, `data_engine._today_et() -> str`.
- Produces: `daily_store.DONE_TABLE = "daily_backfill_done"`, `done_dates(engine, source) -> set[str]`, `completed_game_dates(engine, start, end) -> list[tuple[str, str, str]]` (date, SEASON, season_type), `store_day(engine, *, table, source, date, df, key, date_col="GAME_DATE") -> int`, `run_backfill(engine, *, source, table, key, start, end, fetch, refetch=False, allow_empty=False, date_col="GAME_DATE") -> int` where `fetch(date: str, season: str, season_type: str) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_daily_store.py
import pandas as pd
import pytest
from sqlalchemy import create_engine

import daily_store as ds


@pytest.fixture
def eng(tmp_path, monkeypatch):
    e = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    pd.DataFrame({
        "GAME_ID": ["0012500001", "0022500001", "0022500002", "0042500001", "0022500003"],
        "SEASON": ["2025-26"] * 5,
        "GAME_DATE_EST": ["2025-10-10", "2025-10-21", "2025-10-21", "2026-04-20", "2026-10-14"],
    }).to_sql("schedule", e, index=False)
    monkeypatch.setattr("data_engine._today_et", lambda: "2026-10-14")
    return e


def _fetch_ok(date, season, stype):
    return pd.DataFrame({"PLAYER_ID": [1, 2], "GAME_DATE": [date, date], "STYPE": [stype, stype]})


def test_completed_game_dates_skip_preseason_and_today(eng):
    assert ds.completed_game_dates(eng, "2025-10-01", "2026-12-31") == [
        ("2025-10-21", "2025-26", "Regular Season"), ("2026-04-20", "2025-26", "Playoffs")]


def test_run_backfill_marks_done_and_is_idempotent(eng):
    calls = []

    def fetch(d, s, t):
        calls.append(d)
        return _fetch_ok(d, s, t)

    n = ds.run_backfill(eng, source="x", table="xt", key=["PLAYER_ID", "GAME_DATE"],
                        start="2025-10-01", end="2026-12-31", fetch=fetch)
    assert n == 4 and calls == ["2025-10-21", "2026-04-20"]
    assert ds.run_backfill(eng, source="x", table="xt", key=["PLAYER_ID", "GAME_DATE"],
                           start="2025-10-01", end="2026-12-31", fetch=fetch) == 0
    assert len(calls) == 2
    assert ds.done_dates(eng, "x") == {"2025-10-21", "2026-04-20"}


def test_run_backfill_skips_today_and_empty(eng):
    def fetch(d, s, t):
        if d == "2025-10-21":
            raise RuntimeError("stats.nba.com down")
        return pd.DataFrame(columns=["PLAYER_ID", "GAME_DATE"])

    n = ds.run_backfill(eng, source="y", table="yt", key=["PLAYER_ID", "GAME_DATE"],
                        start="2025-10-01", end="2026-12-31", fetch=fetch)
    assert n == 0 and ds.done_dates(eng, "y") == set()  # failure and empty both stay undone
    ds.run_backfill(eng, source="y", table="yt", key=["PLAYER_ID", "GAME_DATE"],
                    start="2026-04-01", end="2026-12-31", fetch=fetch, allow_empty=True)
    assert ds.done_dates(eng, "y") == {"2026-04-20"}  # allowed empty is done; today never fetched


def test_store_day_replaces_rows_for_date(eng):
    key = ["PLAYER_ID", "GAME_DATE"]
    ds.store_day(eng, table="zt", source="z", date="2025-10-21", df=_fetch_ok("2025-10-21", "", ""), key=key)
    ds.store_day(eng, table="zt", source="z", date="2025-10-21",
                 df=pd.DataFrame({"PLAYER_ID": [9], "GAME_DATE": ["2025-10-21"], "STYPE": ["r"]}), key=key)
    assert pd.read_sql("SELECT PLAYER_ID FROM zt", eng)["PLAYER_ID"].tolist() == [9]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_daily_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'daily_store'`.

- [ ] **Step 3: Implement `daily_store.py`**

```python
"""
daily_store.py
--------------
Per-date tables filled by resumable backfills (tracking_daily, starters, lineups).

A date is marked done in DONE_TABLE only after its rows are written, in the same
transaction. Failed fetches, and empty results unless ``allow_empty``, stay undone so
the next run (nightly: a 3-day lookback) retries them. Today and future dates are
never fetched: stats for unfinished games would be stored as complete.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable

import pandas as pd
from sqlalchemy import inspect, text

import data_engine as de

log = logging.getLogger(__name__)

DONE_TABLE = "daily_backfill_done"
_SEASON_TYPES = {"002": "Regular Season", "004": "Playoffs", "005": "PlayIn"}


def done_dates(engine, source: str) -> set[str]:
    with engine.connect() as conn:
        if not inspect(conn).has_table(DONE_TABLE):
            return set()
        rows = conn.execute(text(f"SELECT date FROM {DONE_TABLE} WHERE source = :s"), {"s": source}).fetchall()
    return {r[0] for r in rows}


def completed_game_dates(engine, start: str, end: str) -> list[tuple[str, str, str]]:
    """[(date, SEASON, season_type)] with regular-season/play-in/playoff games in [start, end], before today ET."""
    sch = pd.read_sql(text("SELECT GAME_ID, SEASON, GAME_DATE_EST FROM schedule "
                           "WHERE GAME_DATE_EST BETWEEN :a AND :b AND GAME_DATE_EST < :t"),
                      engine, params={"a": start, "b": end, "t": de._today_et()})
    sch["season_type"] = sch["GAME_ID"].astype(str).str[:3].map(_SEASON_TYPES)
    sch = sch.dropna(subset=["season_type"])
    return sorted(set(zip(sch["GAME_DATE_EST"], sch["SEASON"], sch["season_type"])))


def store_day(engine, *, table: str, source: str, date: str, df: pd.DataFrame, key: list[str],
              date_col: str = "GAME_DATE") -> int:
    """Replace ``table`` rows for ``date`` with ``df`` and mark (source, date) done, atomically."""
    with engine.begin() as conn:
        if inspect(conn).has_table(table):
            conn.execute(text(f"DELETE FROM {table} WHERE {date_col} = :d"), {"d": date})
        if len(df):
            df.to_sql(table, conn, if_exists="append", index=False)
        if inspect(conn).has_table(DONE_TABLE):
            conn.execute(text(f"DELETE FROM {DONE_TABLE} WHERE source = :s AND date = :d"), {"s": source, "d": date})
        pd.DataFrame({"source": [source], "date": [date]}).to_sql(DONE_TABLE, conn, if_exists="append", index=False)
    if len(df):
        de._assert_unique(table, key, engine)
    return len(df)


def run_backfill(engine, *, source: str, table: str, key: list[str], start: str, end: str,
                 fetch: Callable[[str, str, str], pd.DataFrame], refetch: bool = False,
                 allow_empty: bool = False, date_col: str = "GAME_DATE") -> int:
    done = set() if refetch else done_dates(engine, source)
    parts: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for date, season, stype in completed_game_dates(engine, start, end):
        parts[date].append((season, stype))
    todo = [d for d in sorted(parts) if d not in done]
    log.info(f"[{source}] {len(todo)} date(s) to fetch ({len(parts) - len(todo)} already done)")
    n = 0
    for i, date in enumerate(todo, 1):
        try:
            df = pd.concat([fetch(date, s, t) for s, t in parts[date]], ignore_index=True)
        except Exception as e:  # network / parse: retry next run
            log.warning(f"[{source}] {date} failed: {e}")
            continue
        df = df.drop_duplicates(subset=key, keep="last") if len(df) else df
        if df.empty and not allow_empty:
            log.warning(f"[{source}] {date} returned no rows; leaving it undone")
            continue
        n += store_day(engine, table=table, source=source, date=date, df=df, key=key, date_col=date_col)
        if i % 25 == 0:
            log.info(f"[{source}] {i}/{len(todo)} dates")
    return n
```

- [ ] **Step 4: SQLite busy timeout**

Backfills (nba_api) and the Kalshi candle backfill run in parallel processes on the same file. In `data_engine._get_engine` and `kalshi_history._engine`:

```python
    return create_engine(f"sqlite:///{DB_PATH}", connect_args={"timeout": 60})
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_daily_store.py -q && .venv/bin/python -m pytest -q`
Expected: PASS; full suite green.

- [ ] **Step 6: Commit**

```bash
git add daily_store.py data_engine.py kalshi_history.py tests/test_daily_store.py
git commit -m "feat: resumable per-date backfill store"
```

---

### Task 3: `travel` group — travel distance, time zones, altitude, schedule load

**Files:**
- Create: `travel.py`
- Modify: `feature_store.py` (`build_feature_table`: `schedule` param + travel block)
- Test: `tests/test_travel.py`

**Interfaces:**
- Consumes: `schedule` table columns `GAME_ID, SEASON, GAME_DATE_EST, TIP_UTC, HOME_TEAM_ID, AWAY_TEAM_ID, HOME_TRICODE, AWAY_TRICODE`.
- Produces: `travel.ARENAS: dict[str, tuple[float, float, str, int]]` (lat, lon, IANA tz, altitude m), `travel.travel_features(schedule) -> DataFrame[TEAM_ID, GAME_ID, travel_km, tz_shift, high_altitude_venue, games_last_7, sched_rest_days]`; `build_feature_table(..., schedule: Optional[pd.DataFrame] = None)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_travel.py
import numpy as np
import pandas as pd
import pytest

import feature_store as fs
import travel
from tests.test_feature_store_pit import _synthetic_gamelogs, _tracking

BOS, LAL, DEN = 1610612738, 1610612747, 1610612743


def _sched(rows):
    return pd.DataFrame(rows, columns=["GAME_ID", "SEASON", "GAME_DATE_EST", "TIP_UTC", "HOME_TEAM_ID",
                                       "AWAY_TEAM_ID", "HOME_TRICODE", "AWAY_TRICODE"])


SCHED = _sched([
    ("0012500001", "2025-26", "2025-10-15", "2025-10-15T23:00:00Z", BOS, LAL, "BOS", "LAL"),  # preseason: ignored
    ("0022500001", "2025-26", "2025-11-01", "2025-11-01T23:00:00Z", BOS, LAL, "BOS", "LAL"),
    ("0022500002", "2025-26", "2025-11-03", "2025-11-04T02:00:00Z", DEN, LAL, "DEN", "LAL"),
    ("0022500003", "2025-26", "2025-11-04", "2025-11-05T03:00:00Z", LAL, DEN, "LAL", "DEN"),
])


def test_travel_features_values():
    tf = travel.travel_features(SCHED).set_index(["TEAM_ID", "GAME_ID"])
    lal = tf.xs(LAL)
    assert lal.loc["0022500001", "travel_km"] == 0.0           # first game of season
    assert lal.loc["0022500002", "travel_km"] == pytest.approx(2840, rel=0.03)  # Boston -> Denver
    assert lal.loc["0022500002", "tz_shift"] == -2.0           # ET -> MT
    assert lal.loc["0022500002", "high_altitude_venue"] == 1
    assert lal.loc["0022500003", "high_altitude_venue"] == 0
    assert lal.loc["0022500003", "games_last_7"] == 2
    assert lal.loc["0022500003", "sched_rest_days"] == 1
    assert lal.loc["0022500001", "sched_rest_days"] == 10      # season opener
    assert "0012500001" not in tf.index.get_level_values("GAME_ID")


def test_unknown_venue_gives_nan_distance():
    s = SCHED.copy()
    s.loc[s["GAME_ID"] == "0022500002", "HOME_TRICODE"] = "XXX"
    tf = travel.travel_features(s).set_index(["TEAM_ID", "GAME_ID"])
    assert np.isnan(tf.loc[(LAL, "0022500002"), "travel_km"])


def test_travel_group_columns_in_feature_table():
    gl = _synthetic_gamelogs(n_days=10)
    df = fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False,
                                groups=["travel"], schedule=SCHED)
    for c in fs.FEATURE_GROUPS["travel"]:
        assert c in df.columns  # synthetic team ids aren't in SCHED: NaN, no KeyError
    assert df["travel_km"].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_travel.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'travel'`.

- [ ] **Step 3: Implement `travel.py`**

```python
"""
travel.py
---------
Schedule-load features from the stored schedule (all known before tip): great-circle
distance from the team's previous game venue, time-zone shift, high-altitude venue,
games in the previous 7 days, and days since the previous game.

Venue = home team's arena. Neutral-site games (NBA Cup final, international games)
are attributed to the listed home team; there are a handful per season.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# tricode -> (lat, lon, IANA tz, altitude m)
ARENAS: dict[str, tuple[float, float, str, int]] = {
    "ATL": (33.757, -84.396, "America/New_York", 320),
    "BOS": (42.366, -71.062, "America/New_York", 5),
    "BKN": (40.683, -73.976, "America/New_York", 10),
    "CHA": (35.225, -80.839, "America/New_York", 230),
    "CHI": (41.881, -87.674, "America/Chicago", 180),
    "CLE": (41.496, -81.688, "America/New_York", 200),
    "DAL": (32.790, -96.810, "America/Chicago", 140),
    "DEN": (39.749, -105.008, "America/Denver", 1600),
    "DET": (42.341, -83.055, "America/Detroit", 190),
    "GSW": (37.768, -122.388, "America/Los_Angeles", 5),
    "HOU": (29.751, -95.362, "America/Chicago", 15),
    "IND": (39.764, -86.155, "America/Indiana/Indianapolis", 220),
    "LAC": (33.945, -118.343, "America/Los_Angeles", 30),
    "LAL": (34.043, -118.267, "America/Los_Angeles", 80),
    "MEM": (35.138, -90.051, "America/Chicago", 80),
    "MIA": (25.781, -80.188, "America/New_York", 2),
    "MIL": (43.045, -87.917, "America/Chicago", 190),
    "MIN": (44.980, -93.276, "America/Chicago", 260),
    "NOP": (29.949, -90.082, "America/Chicago", 1),
    "NYK": (40.751, -73.993, "America/New_York", 10),
    "OKC": (35.463, -97.515, "America/Chicago", 360),
    "ORL": (28.539, -81.384, "America/New_York", 30),
    "PHI": (39.901, -75.172, "America/New_York", 5),
    "PHX": (33.446, -112.071, "America/Phoenix", 330),
    "POR": (45.532, -122.667, "America/Los_Angeles", 15),
    "SAC": (38.580, -121.500, "America/Los_Angeles", 8),
    "SAS": (29.427, -98.438, "America/Chicago", 200),
    "TOR": (43.643, -79.379, "America/Toronto", 80),
    "UTA": (40.768, -111.901, "America/Denver", 1300),
    "WAS": (38.898, -77.021, "America/New_York", 20),
}
HIGH_ALTITUDE_M = 1000
GAME_PREFIXES = ("002", "004", "005")
MAX_REST = 10


def _km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _utc_offset_h(venue: str, date: pd.Timestamp) -> float:
    a = ARENAS.get(venue)
    if a is None or pd.isna(date):
        return np.nan
    noon = datetime(date.year, date.month, date.day, 12, tzinfo=ZoneInfo(a[2]))
    return noon.utcoffset().total_seconds() / 3600.0


def travel_features(schedule: pd.DataFrame) -> pd.DataFrame:
    s = schedule[schedule["GAME_ID"].astype(str).str[:3].isin(GAME_PREFIXES)].copy()
    s["GAME_ID"] = s["GAME_ID"].astype(str)
    s["GAME_DATE"] = pd.to_datetime(s["GAME_DATE_EST"]).dt.normalize()
    s["venue"] = s["HOME_TRICODE"].astype(str)
    tg = pd.concat([s.assign(TEAM_ID=s["HOME_TEAM_ID"]), s.assign(TEAM_ID=s["AWAY_TEAM_ID"])], ignore_index=True)
    tg["TEAM_ID"] = tg["TEAM_ID"].astype(int)
    tg = tg.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    g = tg.groupby("TEAM_ID")
    prev_venue, prev_date = g["venue"].shift(1), g["GAME_DATE"].shift(1)
    first = g["SEASON"].shift(1).ne(tg["SEASON"])  # first game of the team's season

    coords = lambda v: (ARENAS.get(v, (np.nan, np.nan))[0], ARENAS.get(v, (np.nan, np.nan))[1])
    cur = np.array([coords(v) for v in tg["venue"]])
    prv = np.array([coords(v) if isinstance(v, str) else (np.nan, np.nan) for v in prev_venue])
    km = _km(prv[:, 0], prv[:, 1], cur[:, 0], cur[:, 1])
    cur_off = np.array([_utc_offset_h(v, d) for v, d in zip(tg["venue"], tg["GAME_DATE"])])
    prv_off = np.array([_utc_offset_h(v, d) if isinstance(v, str) else np.nan
                        for v, d in zip(prev_venue, prev_date)])

    out = tg[["TEAM_ID", "GAME_ID"]].copy()
    out["travel_km"] = np.where(first, 0.0, km)
    out["tz_shift"] = np.where(first, 0.0, cur_off - prv_off)
    alt = tg["venue"].map(lambda v: ARENAS[v][3] if v in ARENAS else np.nan)
    out["high_altitude_venue"] = np.where(alt.isna(), np.nan, (alt > HIGH_ALTITUDE_M).astype(float))
    rest = (tg["GAME_DATE"] - prev_date).dt.days
    out["sched_rest_days"] = np.where(first, MAX_REST, rest.clip(upper=MAX_REST))

    last7 = np.zeros(len(tg))
    for _, idx in tg.groupby("TEAM_ID").groups.items():
        d = tg.loc[idx, "GAME_DATE"].values
        lo = np.searchsorted(d, d - np.timedelta64(7, "D"), side="left")
        hi = np.searchsorted(d, d, side="left")
        last7[np.asarray(idx)] = hi - lo
    out["games_last_7"] = last7
    return out
```

- [ ] **Step 4: Attach the group in `feature_store.build_feature_table`**

Add parameter `schedule: Optional[pd.DataFrame] = None,` (document: "schedule for the travel group; default loaded from SQLite when gamelogs come from SQLite"). Insert after the matchup block (before the teammate vacuum):

```python
    if "travel" in groups:
        import travel

        sched = de.load_schedule() if schedule is None and gamelogs is None else schedule
        if sched is not None and len(sched):
            tf = travel.travel_features(sched)
            df = df.merge(tf.drop(columns=["sched_rest_days"]), on=["TEAM_ID", "GAME_ID"], how="left")
            opp = tf[["TEAM_ID", "GAME_ID", "sched_rest_days"]].rename(
                columns={"TEAM_ID": "OPP_TEAM_ID", "sched_rest_days": "opp_rest_days"})
            df = df.merge(opp, on=["OPP_TEAM_ID", "GAME_ID"], how="left")
```

(The NaN-fill loop from Task 1 covers a missing schedule.) Ensure `df["TEAM_ID"]`/`df["OPP_TEAM_ID"]` are `int` before merging: add `df["OPP_TEAM_ID"] = df["OPP_TEAM_ID"].astype(int)` right after the targets/extra concat — check `test_live_rows_match_history` still passes.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest tests/test_travel.py tests/test_feature_store_pit.py -q && .venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add travel.py feature_store.py tests/test_travel.py
git commit -m "feat: travel and schedule-load feature group"
```

---

### Task 4: `opp_vacuum` group — opponent absences

**Files:**
- Modify: `feature_store.py` (teammate-vacuum block)
- Test: `tests/test_opp_vacuum.py`

**Interfaces:**
- Consumes: `teammate_vacuum(targets, ...)` (unchanged signature); targets need `TEAM_ID, GAME_ID, GAME_DATE, SEASON, _live`.
- Produces: columns `opp_vacuum_reb`, `opp_vacuum_min`, `opp_vacuum_n` (0-filled).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_opp_vacuum.py
import pandas as pd

import feature_store as fs
from tests.test_feature_store_pit import _key, _synthetic_gamelogs, _tracking

G = ["opp_vacuum"]


def _build(gl, **kw):
    return fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False, groups=G, **kw)


def test_opp_vacuum_mirrors_opponent_vacuum():
    df = _build(_synthetic_gamelogs())
    tv = df.drop_duplicates(["TEAM_ID", "GAME_ID"]).set_index(["TEAM_ID", "GAME_ID"])["vacuum_reb"]
    for r in df.sample(50, random_state=0).itertuples():
        assert r.opp_vacuum_reb == tv.get((r.OPP_TEAM_ID, r.GAME_ID), 0.0)


def test_opp_vacuum_live_parity_one_sided_slate():
    """Live slate with markets for ONE team only: opponent vacuum must still match history."""
    gl = _synthetic_gamelogs()
    last = pd.to_datetime(gl["GAME_DATE"]).max()
    hist = _key(_build(gl))
    tonight, past = gl[pd.to_datetime(gl["GAME_DATE"]) == last], gl[pd.to_datetime(gl["GAME_DATE"]) < last]
    played = set(tonight["PLAYER_ID"])
    absent = {int(t): set(gl.loc[gl["TEAM_ID"] == t, "PLAYER_ID"]) - played for t in tonight["TEAM_ID"].unique()}
    games = tonight[["GAME_ID", "TEAM_ID"]].drop_duplicates()
    opp = games.merge(games, on="GAME_ID")
    opp = opp[opp["TEAM_ID_x"] != opp["TEAM_ID_y"]].set_index(["GAME_ID", "TEAM_ID_x"])["TEAM_ID_y"]
    one_side = tonight[tonight["TEAM_ID"] == tonight["TEAM_ID"].iloc[0]]
    extra = pd.DataFrame({
        "PLAYER_ID": one_side["PLAYER_ID"].values, "PLAYER_NAME": one_side["PLAYER_NAME"].values,
        "TEAM_ID": one_side["TEAM_ID"].values,
        "OPP_TEAM_ID": [opp[(g, t)] for g, t in zip(one_side["GAME_ID"], one_side["TEAM_ID"])],
        "GAME_ID": one_side["GAME_ID"].values, "GAME_DATE": one_side["GAME_DATE"].values,
        "SEASON": one_side["SEASON"].values,
        "is_home": one_side["MATCHUP"].str.contains("vs.", regex=False).astype(int).values, "is_playoffs": 0,
    })
    live = _build(past, extra_rows=extra, absent_override=absent)
    live = _key(live[live["_live"]])
    cols = fs.FEATURE_GROUPS["opp_vacuum"]
    assert live["opp_vacuum_n"].sum() > 0 or hist.loc[live.index, "opp_vacuum_n"].sum() == 0
    pd.testing.assert_frame_equal(hist.loc[live.index, cols], live[cols], check_dtype=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_opp_vacuum.py -q`
Expected: FAIL — `opp_vacuum_reb` values are NaN (Task 1 stub) so the equality asserts fail.

- [ ] **Step 3: Implement**

Replace the vacuum call and merge in `build_feature_table`:

```python
    # Vacuum for every team-game in the slate, including opponents that have no target rows
    # (live: an opponent without Kalshi markets) so opp_vacuum is defined for them too.
    vac_targets = df[["TEAM_ID", "GAME_ID", "GAME_DATE", "SEASON", "_live"]]
    if "opp_vacuum" in groups:
        vac_targets = pd.concat([vac_targets, df[["OPP_TEAM_ID", "GAME_ID", "GAME_DATE", "SEASON", "_live"]]
                                 .rename(columns={"OPP_TEAM_ID": "TEAM_ID"})], ignore_index=True)
    vac = teammate_vacuum(vac_targets, pstate, pool, played, absent_override, pregame_out=pregame_out,
                          appearances=gl[["PLAYER_ID", "GAME_DATE", "TEAM_ID"]], current_team=current_team)
    df = df.merge(vac, on=["TEAM_ID", "GAME_ID"], how="left")
    for c in ("vacuum_reb", "vacuum_min", "vacuum_n"):
        df[c] = df[c].fillna(0.0)
    if "opp_vacuum" in groups:
        ov = vac.rename(columns={"TEAM_ID": "OPP_TEAM_ID", "vacuum_reb": "opp_vacuum_reb",
                                 "vacuum_min": "opp_vacuum_min", "vacuum_n": "opp_vacuum_n"})
        df = df.merge(ov, on=["OPP_TEAM_ID", "GAME_ID"], how="left")
        for c in FEATURE_GROUPS["opp_vacuum"]:
            df[c] = df[c].fillna(0.0)
```

`teammate_vacuum` reads `targets["_live"]` and drops duplicates on (TEAM_ID, GAME_ID) itself, so the concatenated frame needs no extra handling.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_opp_vacuum.py tests/test_feature_store_pit.py tests/test_pregame_vacuum.py -q && .venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add feature_store.py tests/test_opp_vacuum.py
git commit -m "feat: opponent-absence feature group"
```

---

### Task 5: `tracking` group — per-game rebound tracking

**Files:**
- Create: `nba_daily.py`
- Modify: `feature_store.py` (new `tracking_state`, `tracking_daily` param + block)
- Modify: `pipeline/commands.py` (new `tracking_backfill`), `cli/main.py` (register)
- Test: `tests/test_tracking_daily.py`

**Interfaces:**
- Consumes: `daily_store.run_backfill`, `data_engine._call`, `data_engine._get_engine`.
- Produces: `nba_daily.TRACKING_TABLE = "tracking_daily"`, `nba_daily.TRACKING_KEY = ["PLAYER_ID", "GAME_DATE"]`, `nba_daily.fetch_tracking_day(date, season, season_type) -> DataFrame`, `nba_daily.backfill_tracking(start, end, engine=None, refetch=False) -> int`, `nba_daily.load_table(table, engine=None) -> DataFrame` (empty on missing table), `feature_store.tracking_state(td, window=ROLLING_WINDOW) -> DataFrame`, `build_feature_table(..., tracking_daily: Optional[pd.DataFrame] = None)`, CLI `tracking-backfill --start --end [--refetch]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tracking_daily.py
import numpy as np
import pandas as pd
import pytest

import feature_store as fs
from tests.test_feature_store_pit import _key, _synthetic_gamelogs, _tracking

G = ["tracking"]


def _td_from(gl, seed=0):
    rng = np.random.default_rng(seed)
    t = gl[["PLAYER_ID", "TEAM_ID", "GAME_DATE", "MIN", "REB"]].copy()
    t["GAME_DATE"] = pd.to_datetime(t["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    t["REB_CHANCES"] = t["REB"] + rng.integers(0, 6, len(t))
    t["OREB_CHANCES"] = (t["REB_CHANCES"] * 0.3).round()
    t["DREB_CHANCES"] = t["REB_CHANCES"] - t["OREB_CHANCES"]
    t["REB_CONTEST"] = (t["REB"] * 0.4).round()
    t["REB_CHANCE_DEFER"] = rng.integers(0, 3, len(t))
    t["AVG_REB_DIST"] = rng.uniform(4, 10, len(t))
    return t


def _build(gl, td, **kw):
    return fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False, groups=G,
                                  tracking_daily=td, **kw)


def test_tracking_state_ratios():
    td = pd.DataFrame({"PLAYER_ID": [1] * 3, "GAME_DATE": ["2025-11-01", "2025-11-03", "2025-11-05"],
                       "MIN": [30, 30, 30], "REB": [5, 7, 9], "REB_CHANCES": [10, 10, 10],
                       "OREB_CHANCES": [3, 3, 3], "DREB_CHANCES": [7, 7, 7], "REB_CONTEST": [2, 2, 2],
                       "REB_CHANCE_DEFER": [1, 1, 1], "AVG_REB_DIST": [6, 6, 6], "TEAM_ID": 9})
    st = fs.tracking_state(td).iloc[-1]
    assert st["reb_chances_roll"] == pytest.approx(10)
    assert st["reb_conv_roll"] == pytest.approx(21 / 30)
    assert st["reb_contest_pct_roll"] == pytest.approx(6 / 21)
    assert st["reb_defer_rate_roll"] == pytest.approx(0.1)
    assert st["oreb_chance_share_roll"] == pytest.approx(0.3)
    assert st["reb_chances36_roll"] == pytest.approx(12.0)


def test_tracking_features_no_lookahead():
    gl = _synthetic_gamelogs()
    td = _td_from(gl)
    cut = pd.Timestamp("2024-11-25")
    scr = td.copy()
    late = pd.to_datetime(scr["GAME_DATE"]) >= cut
    for c in ["REB_CHANCES", "REB_CONTEST", "OREB_CHANCES", "REB_CHANCE_DEFER"]:
        scr.loc[late, c] = np.random.default_rng(3).permutation(scr.loc[late, c].values)
    a, b = _key(_build(gl, td)), _key(_build(gl, scr))
    rows = a.index[a["GAME_DATE"] <= cut]
    cols = fs.FEATURE_GROUPS["tracking"]
    assert a.loc[rows, cols].notna().any().all()
    pd.testing.assert_frame_equal(a.loc[rows, cols], b.loc[rows, cols], check_dtype=False)


def test_missing_tracking_table_gives_nan_columns():
    df = _build(_synthetic_gamelogs(n_days=10), pd.DataFrame())
    assert df[fs.FEATURE_GROUPS["tracking"]].isna().all().all()


def test_fetch_tracking_day_shapes(monkeypatch):
    import nba_daily

    raw = pd.DataFrame({"PLAYER_ID": [1], "TEAM_ID": [2], "MIN": [30.0], "REB": [8], "OREB_CHANCES": [3],
                        "DREB_CHANCES": [9], "REB_CHANCES": [12], "REB_CONTEST": [3],
                        "REB_CHANCE_DEFER": [1], "AVG_REB_DIST": [6.1], "PLAYER_NAME": ["x"]})
    seen = {}

    def fake_call(fn, **kw):
        seen.update(kw)
        return [raw]

    monkeypatch.setattr("data_engine._call", fake_call)
    out = nba_daily.fetch_tracking_day("2026-03-10", "2025-26", "Regular Season")
    assert seen["date_from_nullable"] == "03/10/2026" and seen["per_mode_simple"] == "Totals"
    assert out["GAME_DATE"].tolist() == ["2026-03-10"] and out["SEASON"].tolist() == ["2025-26"]
    assert "PLAYER_NAME" not in out.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tracking_daily.py -q`
Expected: FAIL — `AttributeError: module 'feature_store' has no attribute 'tracking_state'` / `No module named 'nba_daily'`.

- [ ] **Step 3: Implement `nba_daily.py` (tracking part)**

```python
"""
nba_daily.py
------------
Per-date nba_api pulls stored by daily_store's resumable backfill:

tracking_daily  LeagueDashPtStats Rebounding, one game date per call (Totals):
                rebound chances, contested rebounds, deferred chances, distance.
starters        LeagueDashPlayerStats StarterBench=Starters, one game date per call.

A player plays at most one game per date, so (PLAYER_ID, GAME_DATE) is the key.
"""

from __future__ import annotations

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashptstats
from sqlalchemy import inspect

import daily_store
import data_engine as de

TRACKING_TABLE = "tracking_daily"
TRACKING_KEY = ["PLAYER_ID", "GAME_DATE"]
TRACKING_COLS = ["PLAYER_ID", "TEAM_ID", "MIN", "REB", "OREB_CHANCES", "DREB_CHANCES", "REB_CHANCES",
                 "REB_CONTEST", "REB_CHANCE_DEFER", "AVG_REB_DIST"]


def _mdy(date: str) -> str:
    return f"{date[5:7]}/{date[8:10]}/{date[:4]}"


def fetch_tracking_day(date: str, season: str, season_type: str) -> pd.DataFrame:
    df = de._call(leaguedashptstats.LeagueDashPtStats, season=season, season_type_all_star=season_type,
                  pt_measure_type="Rebounding", per_mode_simple="Totals", player_or_team="Player",
                  date_from_nullable=_mdy(date), date_to_nullable=_mdy(date))[0]
    out = df[TRACKING_COLS].copy()
    out["GAME_DATE"], out["SEASON"] = date, season
    return out


def backfill_tracking(start: str, end: str, engine=None, refetch: bool = False) -> int:
    return daily_store.run_backfill(engine or de._get_engine(), source="tracking", table=TRACKING_TABLE,
                                    key=TRACKING_KEY, start=start, end=end, fetch=fetch_tracking_day,
                                    refetch=refetch)


def load_table(table: str, engine=None) -> pd.DataFrame:
    engine = engine or de._get_engine()
    if not inspect(engine).has_table(table):
        return pd.DataFrame()
    return pd.read_sql(f"SELECT * FROM {table}", engine)
```

`data_engine._call(fn, *args, **kwargs)` takes keyword arguments; the test's `fake_call(fn, **kw)` matches that call shape.

- [ ] **Step 4: `tracking_state` + block in `feature_store.py`**

Add after `player_state`:

```python
TRACKING_SUM_COLS = ["MIN", "REB", "REB_CHANCES", "OREB_CHANCES", "REB_CONTEST", "REB_CHANCE_DEFER"]


def tracking_state(td: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Per-game rebound-tracking state AFTER each game (inclusive), like player_state."""
    t = td.copy()
    t["GAME_DATE"] = pd.to_datetime(t["GAME_DATE"]).dt.normalize()
    t["PLAYER_ID"] = t["PLAYER_ID"].astype(int)
    for c in TRACKING_SUM_COLS:
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t = t.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    g = t.groupby("PLAYER_ID")

    def rsum(c):
        return g[c].transform(lambda x: x.rolling(window, min_periods=3).sum())

    st = t[["PLAYER_ID", "GAME_DATE"]].copy()
    chances = rsum("REB_CHANCES").replace(0, np.nan)
    st["reb_chances_roll"] = g["REB_CHANCES"].transform(lambda x: x.rolling(window, min_periods=3).mean())
    st["reb_chances_ewm"] = g["REB_CHANCES"].transform(lambda x: x.ewm(halflife=EWM_HALFLIFE, min_periods=3).mean())
    st["reb_chances36_roll"] = 36 * rsum("REB_CHANCES") / rsum("MIN").replace(0, np.nan)
    st["reb_conv_roll"] = rsum("REB") / chances
    st["reb_contest_pct_roll"] = rsum("REB_CONTEST") / rsum("REB").replace(0, np.nan)
    st["reb_defer_rate_roll"] = rsum("REB_CHANCE_DEFER") / chances
    st["oreb_chance_share_roll"] = rsum("OREB_CHANCES") / chances
    return st
```

Add parameter `tracking_daily: Optional[pd.DataFrame] = None,` and, after the player-form block:

```python
    if "tracking" in groups:
        if tracking_daily is None and gamelogs is None:
            from nba_daily import TRACKING_TABLE, load_table

            tracking_daily = load_table(TRACKING_TABLE)
        if tracking_daily is not None and len(tracking_daily):
            df = _asof(df, tracking_state(tracking_daily), by="PLAYER_ID")
```

- [ ] **Step 5: CLI**

In `pipeline/commands.py` after `injury_backfill`:

```python
def tracking_backfill(
    start: str = typer.Option(..., "--start", help="First game date YYYY-MM-DD."),
    end: str = typer.Option(..., "--end", help="Last game date YYYY-MM-DD."),
    refetch: bool = typer.Option(False, "--refetch", help="Re-download dates already stored."),
):
    """Per-game rebound tracking (one nba_api call per game date) into tracking_daily."""
    import nba_daily

    _header(f"Tracking backfill — {start} → {end}")
    _success(f"Stored {nba_daily.backfill_tracking(start, end, refetch=refetch):,} rows in {nba_daily.TRACKING_TABLE}.")
```

In `cli/main.py` after the `injury-backfill` line: `app.command("tracking-backfill")(cmd.tracking_backfill)`.

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest tests/test_tracking_daily.py -q && .venv/bin/python -m pytest -q && .venv/bin/python -m pyflakes nba_daily.py feature_store.py pipeline/commands.py`
Expected: PASS; no lint output.

- [ ] **Step 7: Smoke-test one real date**

Run: `.venv/bin/python run_pipeline.py tracking-backfill --start 2026-03-10 --end 2026-03-10 && sqlite3 data/nba_reb.db "select count(*), count(distinct TEAM_ID) from tracking_daily where GAME_DATE='2026-03-10'"`
Expected: about 200 rows, 22 teams (11 games).

- [ ] **Step 8: Commit**

```bash
git add nba_daily.py feature_store.py pipeline/commands.py cli/main.py tests/test_tracking_daily.py
git commit -m "feat: per-game rebound tracking feature group"
```

---

### Task 6: Starters history + `lineup` group features

**Files:**
- Modify: `nba_daily.py` (starters fetch/backfill)
- Modify: `feature_store.py` (new `lineup_features`, params `starters`, `live_starters`, block)
- Modify: `pipeline/commands.py`, `cli/main.py` (`starters-backfill`)
- Test: `tests/test_lineup_features.py`

**Interfaces:**
- Consumes: `daily_store.run_backfill`, `nba_daily.load_table`.
- Produces: `nba_daily.STARTERS_TABLE = "starters"`, `nba_daily.STARTERS_KEY = ["PLAYER_ID", "GAME_DATE"]`, `nba_daily.fetch_starters_day(date, season, season_type)`, `nba_daily.backfill_starters(start, end, engine=None, refetch=False)`; `feature_store.lineup_features(df, gl, starters, live_starters, window=ROLLING_WINDOW) -> DataFrame` (index aligned to `df`, columns = `FEATURE_GROUPS["lineup"]`); `build_feature_table(..., starters=None, live_starters: Optional[dict[str, set[int]]] = None)`; CLI `starters-backfill`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lineup_features.py
import numpy as np
import pandas as pd

import feature_store as fs
from tests.test_feature_store_pit import _key, _synthetic_gamelogs, _tracking

G = ["lineup"]


def _starters_from(gl):
    """Five highest-minute players per team-game start (deterministic stand-in for box-score starters)."""
    g = gl.sort_values("MIN", ascending=False).groupby(["TEAM_ID", "GAME_ID"]).head(5)
    out = g[["PLAYER_ID", "TEAM_ID", "GAME_DATE"]].copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    return out


def _build(gl, st, **kw):
    return fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False, groups=G,
                                  starters=st, **kw)


def test_lineup_values_and_role_change():
    gl = _synthetic_gamelogs()
    st = _starters_from(gl)
    df = _build(gl, st)
    s = set(zip(st["PLAYER_ID"], pd.to_datetime(st["GAME_DATE"])))
    exp = [float((p, d) in s) for p, d in zip(df["PLAYER_ID"], df["GAME_DATE"])]
    assert df["starts_tonight"].tolist() == exp
    m = df["start_rate_roll"].notna()
    assert np.allclose(df.loc[m, "role_change"], df.loc[m, "starts_tonight"] - df.loc[m, "start_rate_roll"])
    assert df["team_usual_starters_out"].dropna().between(0, 5).all()


def test_lineup_no_lookahead_and_live_parity():
    gl = _synthetic_gamelogs()
    st = _starters_from(gl)
    last = pd.to_datetime(gl["GAME_DATE"]).max()
    hist = _key(_build(gl, st))
    tonight, past = gl[pd.to_datetime(gl["GAME_DATE"]) == last], gl[pd.to_datetime(gl["GAME_DATE"]) < last]
    st_past = st[pd.to_datetime(st["GAME_DATE"]) < last]
    st_tonight = st[pd.to_datetime(st["GAME_DATE"]) == last]
    live_starters = {}
    for gid, tid in tonight[["GAME_ID", "TEAM_ID"]].drop_duplicates().itertuples(index=False):
        live_starters.setdefault(gid, set()).update(st_tonight.loc[st_tonight["TEAM_ID"] == tid, "PLAYER_ID"])
    played = set(tonight["PLAYER_ID"])
    absent = {int(t): set(gl.loc[gl["TEAM_ID"] == t, "PLAYER_ID"]) - played for t in tonight["TEAM_ID"].unique()}
    games = tonight[["GAME_ID", "TEAM_ID"]].drop_duplicates()
    opp = games.merge(games, on="GAME_ID")
    opp = opp[opp["TEAM_ID_x"] != opp["TEAM_ID_y"]].set_index(["GAME_ID", "TEAM_ID_x"])["TEAM_ID_y"]
    extra = pd.DataFrame({
        "PLAYER_ID": tonight["PLAYER_ID"].values, "PLAYER_NAME": tonight["PLAYER_NAME"].values,
        "TEAM_ID": tonight["TEAM_ID"].values,
        "OPP_TEAM_ID": [opp[(g, t)] for g, t in zip(tonight["GAME_ID"], tonight["TEAM_ID"])],
        "GAME_ID": tonight["GAME_ID"].values, "GAME_DATE": tonight["GAME_DATE"].values,
        "SEASON": tonight["SEASON"].values,
        "is_home": tonight["MATCHUP"].str.contains("vs.", regex=False).astype(int).values, "is_playoffs": 0,
    })
    live = _build(past, st_past, extra_rows=extra, absent_override=absent, live_starters=live_starters)
    live = _key(live[live["_live"]])
    cols = fs.FEATURE_GROUPS["lineup"]
    pd.testing.assert_frame_equal(hist.loc[live.index, cols], live[cols], check_dtype=False)


def test_unconfirmed_live_game_has_nan_lineup_features():
    gl = _synthetic_gamelogs(n_days=20)
    last = pd.to_datetime(gl["GAME_DATE"]).max()
    tonight, past = gl[pd.to_datetime(gl["GAME_DATE"]) == last], gl[pd.to_datetime(gl["GAME_DATE"]) < last]
    extra = tonight[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE", "SEASON"]].assign(
        OPP_TEAM_ID=tonight["TEAM_ID"].map({1: 2, 2: 1, 3: 4, 4: 3}), is_home=0, is_playoffs=0)
    live = _build(past, _starters_from(past), extra_rows=extra, live_starters={})
    live = live[live["_live"]]
    assert live["starts_tonight"].isna().all() and live["team_usual_starters_out"].isna().all()
```

The synthetic schedule alternates pairings ((1,2),(3,4) on even days, (1,3),(2,4) on odd); the `OPP_TEAM_ID` map in the last test is only used for the lineup columns, which don't depend on the opponent.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_lineup_features.py -q`
Expected: FAIL — `TypeError: build_feature_table() got an unexpected keyword argument 'starters'`.

- [ ] **Step 3: Starters fetch in `nba_daily.py`**

```python
STARTERS_TABLE = "starters"
STARTERS_KEY = ["PLAYER_ID", "GAME_DATE"]


def fetch_starters_day(date: str, season: str, season_type: str) -> pd.DataFrame:
    df = de._call(leaguedashplayerstats.LeagueDashPlayerStats, season=season, season_type_all_star=season_type,
                  starter_bench_nullable="Starters", per_mode_detailed="Totals",
                  date_from_nullable=_mdy(date), date_to_nullable=_mdy(date))[0]
    out = df[["PLAYER_ID", "TEAM_ID"]].copy()
    out["GAME_DATE"], out["SEASON"] = date, season
    return out


def backfill_starters(start: str, end: str, engine=None, refetch: bool = False) -> int:
    return daily_store.run_backfill(engine or de._get_engine(), source="starters", table=STARTERS_TABLE,
                                    key=STARTERS_KEY, start=start, end=end, fetch=fetch_starters_day,
                                    refetch=refetch)
```

- [ ] **Step 4: `lineup_features` in `feature_store.py`**

```python
USUAL_STARTER_GAMES = 10


def lineup_features(df: pd.DataFrame, gl: pd.DataFrame, starters: pd.DataFrame,
                    live_starters: Optional[dict[str, set[int]]], window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    starts_tonight           1/0 if tonight's starter (history: box-score starters; live: confirmed
                             lineup), NaN when unknown (date not backfilled / lineup unconfirmed)
    start_rate_roll          share of the player's last ``window`` games started (strictly before)
    role_change              starts_tonight − start_rate_roll
    team_usual_starters_out  of the team's 5 most frequent starters over its previous 10 games,
                             how many are not starting tonight
    """
    st = starters.copy()
    st["GAME_DATE"] = pd.to_datetime(st["GAME_DATE"]).dt.normalize()
    st["PLAYER_ID"], st["TEAM_ID"] = st["PLAYER_ID"].astype(int), st["TEAM_ID"].astype(int)
    covered = set(st["GAME_DATE"])
    started = set(zip(st["PLAYER_ID"], st["GAME_DATE"]))

    hist = gl[["PLAYER_ID", "GAME_DATE"]].sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    hist["_started"] = [float((p, d) in started) if d in covered else np.nan
                        for p, d in zip(hist["PLAYER_ID"], hist["GAME_DATE"])]
    hist["start_rate_roll"] = hist.groupby("PLAYER_ID")["_started"].transform(
        lambda x: x.rolling(window, min_periods=3).mean())
    out = _asof(df[["PLAYER_ID", "GAME_DATE"]], hist[["PLAYER_ID", "GAME_DATE", "start_rate_roll"]], by="PLAYER_ID")
    out = out[["start_rate_roll"]].copy()

    live_starters = live_starters or {}
    by_team = {t: g.groupby("GAME_DATE")["PLAYER_ID"].agg(set).sort_index() for t, g in st.groupby("TEAM_ID")}

    def tonight(pid_team, gid, date, live):
        if live:
            return live_starters.get(str(gid))
        s = by_team.get(pid_team)
        return s.get(date) if s is not None and date in covered else None

    starts, usual_out = [], {}
    for r in df[["PLAYER_ID", "TEAM_ID", "GAME_ID", "GAME_DATE", "_live"]].itertuples(index=False):
        tn = tonight(int(r.TEAM_ID), r.GAME_ID, r.GAME_DATE, r._live)
        starts.append(np.nan if tn is None else float(int(r.PLAYER_ID) in tn))
        key = (int(r.TEAM_ID), str(r.GAME_ID))
        if key not in usual_out:
            s = by_team.get(int(r.TEAM_ID))
            prev = s[s.index < r.GAME_DATE].iloc[-USUAL_STARTER_GAMES:] if s is not None else []
            counts = pd.Series([p for ps in prev for p in ps], dtype="int64").value_counts(sort=False)
            usual = set(counts.sort_values(ascending=False, kind="stable").index[:5]) if len(counts) else set()
            usual_out[key] = np.nan if tn is None or not usual else float(len(usual - set(tn)))
    out["starts_tonight"] = starts
    out["team_usual_starters_out"] = [usual_out[(int(t), str(g))] for t, g in zip(df["TEAM_ID"], df["GAME_ID"])]
    out["role_change"] = out["starts_tonight"] - out["start_rate_roll"]
    return out[FEATURE_GROUPS["lineup"]].set_index(df.index)
```

In `build_feature_table` add params `starters: Optional[pd.DataFrame] = None, live_starters: Optional[dict[str, set[int]]] = None,` and, after the tracking block:

```python
    if "lineup" in groups:
        if starters is None and gamelogs is None:
            from nba_daily import STARTERS_TABLE, load_table

            starters = load_table(STARTERS_TABLE)
        if starters is not None and len(starters):
            lf = lineup_features(df, gl, starters, live_starters)
            for c in lf.columns:
                df[c] = lf[c].values
```

(`_asof` returns rows in the input order, so `.values` alignment is safe; `df` must have a fresh RangeIndex here — it does after the merges above.)

- [ ] **Step 5: CLI**

`pipeline/commands.py`:

```python
def starters_backfill(
    start: str = typer.Option(..., "--start", help="First game date YYYY-MM-DD."),
    end: str = typer.Option(..., "--end", help="Last game date YYYY-MM-DD."),
    refetch: bool = typer.Option(False, "--refetch", help="Re-download dates already stored."),
):
    """Box-score starters (one nba_api call per game date) into the starters table."""
    import nba_daily

    _header(f"Starters backfill — {start} → {end}")
    _success(f"Stored {nba_daily.backfill_starters(start, end, refetch=refetch):,} rows in {nba_daily.STARTERS_TABLE}.")
```

`cli/main.py`: `app.command("starters-backfill")(cmd.starters_backfill)`.

- [ ] **Step 6: Run tests + smoke**

Run: `.venv/bin/python -m pytest tests/test_lineup_features.py -q && .venv/bin/python -m pytest -q`
Then: `.venv/bin/python run_pipeline.py starters-backfill --start 2026-03-10 --end 2026-03-10 && sqlite3 data/nba_reb.db "select count(*) from starters where GAME_DATE='2026-03-10'"`
Expected: PASS; `110` (11 games × 10 starters).

- [ ] **Step 7: Commit**

```bash
git add nba_daily.py feature_store.py pipeline/commands.py cli/main.py tests/test_lineup_features.py
git commit -m "feat: starters backfill and lineup feature group"
```

---

### Task 7: Lineups JSON (confirmed starters, Inactive list) + `inactive` group

**Files:**
- Create: `lineups.py`
- Modify: `feature_store.py` (`pregame_inactive` param, union into `pregame_out`)
- Modify: `pipeline/commands.py`, `cli/main.py` (`lineups-backfill`)
- Test: `tests/test_lineups.py`

**Interfaces:**
- Consumes: `daily_store.run_backfill`, `config.DECISION_LEAD_MIN`, schedule `TIP_UTC`.
- Produces: `lineups.URL`, `lineups.TABLE = "lineups"`, `lineups.KEY = ["GAME_ID", "PLAYER_ID"]`, `lineups.parse_lineups(payload: dict, game_date: str) -> DataFrame` (columns `GAME_DATE, GAME_ID, TEAM_ID, PLAYER_ID, PLAYER_NAME, lineup_status, roster_status, position, ts_utc`), `lineups.fetch_day(game_date, session=None) -> DataFrame`, `lineups.backfill(start, end, engine=None, refetch=False) -> int`, `@dataclass GameLineup(confirmed: bool, starters: set[int], inactive: dict[int, set[int]])`, `lineups.lineup_states(df) -> dict[str, GameLineup]`, `lineups.todays_lineups(game_date) -> dict[str, GameLineup] | None`, `lineups.pregame_inactive(engine=None, lead_min=None) -> dict[tuple[str, int], set[int]]`; `build_feature_table(..., pregame_inactive=None)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lineups.py
import pandas as pd
import pytest
from sqlalchemy import create_engine

import lineups as lu


def _p(pid, team, pos="", lineup="Confirmed", roster="Active", ts="2026-03-10T21:53:00"):
    return {"personId": pid, "teamId": team, "playerName": f"P{pid}", "lineupStatus": lineup,
            "position": pos, "rosterStatus": roster, "timestamp": ts}


def _payload(home_players, away_players, gid="0022500936"):
    return {"games": [{"gameId": gid, "homeTeam": {"teamId": 1, "players": home_players},
                       "awayTeam": {"teamId": 2, "players": away_players}}]}


FULL = _payload([_p(i, 1, "G") for i in range(1, 6)] + [_p(6, 1), _p(7, 1, roster="Inactive")],
                [_p(i, 2, "F") for i in range(11, 16)] + [_p(17, 2, roster="Inactive")])


def test_parse_and_states_confirmed():
    df = lu.parse_lineups(FULL, "2026-03-10")
    assert len(df) == 13 and set(df["GAME_ID"]) == {"0022500936"}
    st = lu.lineup_states(df)["0022500936"]
    assert st.confirmed and st.starters == {1, 2, 3, 4, 5, 11, 12, 13, 14, 15}
    assert st.inactive == {1: {7}, 2: {17}}


def test_lineup_states_unconfirmed_cases():
    expected = _payload([_p(i, 1, "G", lineup="Expected") for i in range(1, 6)],
                        [_p(i, 2, "F") for i in range(11, 16)])
    four = _payload([_p(i, 1, "G") for i in range(1, 5)], [_p(i, 2, "F") for i in range(11, 16)])
    one_team = {"games": [{"gameId": "g", "homeTeam": {"teamId": 1, "players": [_p(i, 1, "G") for i in range(5)]},
                           "awayTeam": {"teamId": 2, "players": []}}]}
    for payload in (expected, four, one_team):
        states = lu.lineup_states(lu.parse_lineups(payload, "2026-03-10"))
        assert not any(s.confirmed for s in states.values())


def test_empty_season_file_parses_to_no_rows():
    assert lu.parse_lineups({"games": []}, "2024-11-05").empty


def test_pregame_inactive_respects_cutoff(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    pd.DataFrame([{"GAME_ID": "0022500936", "TIP_UTC": "2026-03-10T23:00:00Z", "GAME_DATE_EST": "2026-03-10"},
                  {"GAME_ID": "0022500937", "TIP_UTC": "2026-03-10T23:30:00Z", "GAME_DATE_EST": "2026-03-10"}]
                 ).to_sql("schedule", eng, index=False)
    early = lu.parse_lineups(FULL, "2026-03-10")                                   # 21:53Z: before T-30
    late = lu.parse_lineups(_payload([_p(8, 1, roster="Inactive", ts="2026-03-10T23:10:00Z")], [],
                                     gid="0022500937"), "2026-03-10")               # after T-30
    pd.concat([early, late]).to_sql(lu.TABLE, eng, index=False)
    out = lu.pregame_inactive(eng, lead_min=30)
    assert out[("0022500936", 1)] == {7} and out[("0022500936", 2)] == {17}
    assert ("0022500937", 1) not in out


def test_inactive_group_adds_listed_players_to_vacuum():
    import feature_store as fs
    from tests.test_feature_store_pit import _synthetic_gamelogs, _tracking

    gl = _synthetic_gamelogs()
    gid, team = gl["GAME_ID"].iloc[-1], int(gl["TEAM_ID"].iloc[-1])
    star = team * 100  # k=0: biggest minutes, always in rotation
    played = gl[~((gl["GAME_ID"] == gid) & (gl["PLAYER_ID"] == star))]
    base = fs.build_feature_table(gamelogs=played, tracking=_tracking(), apply_filters=False,
                                  pregame_out={(gid, team): set()})
    with_inactive = fs.build_feature_table(gamelogs=played, tracking=_tracking(), apply_filters=False,
                                           groups=["inactive"], pregame_out={(gid, team): set()},
                                           pregame_inactive={(gid, team): {star}})
    b = base[(base["GAME_ID"] == gid) & (base["TEAM_ID"] == team)]["vacuum_n"].iloc[0]
    w = with_inactive[(with_inactive["GAME_ID"] == gid) & (with_inactive["TEAM_ID"] == team)]["vacuum_n"].iloc[0]
    assert w == b + 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_lineups.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'lineups'`.

- [ ] **Step 3: Implement `lineups.py`**

```python
"""
lineups.py
----------
NBA daily lineups JSON (stats.nba.com/js/data/leaders/00_daily_lineups_YYYYMMDD.json):
starters with lineupStatus Expected/Confirmed and every rostered player's rosterStatus
(Active/Inactive), each with a UTC timestamp of its last update. Confirmations land
~60-70 min before tip (checked on 2026-03-10).

History: stored per date (2023-24 and 2025-26 have files; 2024-25 files are empty).
Live: re-read on every scan. The Inactive list catches players ruled out after the
last injury report (Questionable/Doubtful players who then sit).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
import requests
from sqlalchemy import inspect, text

import daily_store
import data_engine as de
from config import DECISION_LEAD_MIN

log = logging.getLogger(__name__)

URL = "https://stats.nba.com/js/data/leaders/00_daily_lineups_{ymd}.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/"}
TABLE = "lineups"
KEY = ["GAME_ID", "PLAYER_ID"]
COLS = ["GAME_DATE", "GAME_ID", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "lineup_status", "roster_status",
        "position", "ts_utc"]


@dataclass
class GameLineup:
    confirmed: bool
    starters: set[int] = field(default_factory=set)
    inactive: dict[int, set[int]] = field(default_factory=dict)


def parse_lineups(payload: dict, game_date: str) -> pd.DataFrame:
    rows = []
    for g in payload.get("games", []) or []:
        for side in ("homeTeam", "awayTeam"):
            t = g.get(side) or {}
            for p in t.get("players", []) or []:
                rows.append({
                    "GAME_DATE": game_date, "GAME_ID": str(g["gameId"]),
                    "TEAM_ID": int(p.get("teamId") or t.get("teamId")), "PLAYER_ID": int(p["personId"]),
                    "PLAYER_NAME": p.get("playerName", ""), "lineup_status": p.get("lineupStatus") or "",
                    "roster_status": p.get("rosterStatus") or "", "position": p.get("position") or "",
                    "ts_utc": p.get("timestamp") or "",
                })
    return pd.DataFrame(rows, columns=COLS)


def fetch_day(game_date: str, session=None) -> pd.DataFrame:
    r = (session or requests).get(URL.format(ymd=game_date.replace("-", "")), headers=HEADERS, timeout=30)
    r.raise_for_status()
    return parse_lineups(r.json(), game_date)


def backfill(start: str, end: str, engine=None, refetch: bool = False) -> int:
    return daily_store.run_backfill(engine or de._get_engine(), source="lineups", table=TABLE, key=KEY,
                                    start=start, end=end, fetch=lambda d, s, t: fetch_day(d),
                                    refetch=refetch, allow_empty=True)


def lineup_states(df: pd.DataFrame) -> dict[str, GameLineup]:
    """Per GAME_ID: confirmed = two teams, five Confirmed starters each."""
    out: dict[str, GameLineup] = {}
    for gid, g in df.groupby("GAME_ID"):
        st = g[g["position"].astype(str) != ""]
        teams = set(g["TEAM_ID"].astype(int))
        confirmed = (len(teams) == 2 and all((st["TEAM_ID"] == t).sum() == 5 for t in teams)
                     and (st["lineup_status"] == "Confirmed").all())
        inactive = {int(t): set(g.loc[(g["TEAM_ID"] == t) & (g["roster_status"] == "Inactive"), "PLAYER_ID"]
                                .astype(int)) for t in teams}
        out[str(gid)] = GameLineup(bool(confirmed), set(st["PLAYER_ID"].astype(int)), inactive)
    return out


def todays_lineups(game_date: str) -> dict[str, GameLineup] | None:
    """Live states for ``game_date``; None when the JSON can't be read (callers skip lineup games)."""
    try:
        return lineup_states(fetch_day(game_date))
    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning("Lineups unavailable for %s (%s)", game_date, e)
        return None


def pregame_inactive(engine=None, lead_min: int | None = None) -> dict[tuple[str, int], set[int]]:
    """
    {(GAME_ID, TEAM_ID): Inactive player ids} as known at tip − lead_min. A team-game is
    included only if it has a lineup row stamped at or before the cutoff (else the list
    wasn't out yet and callers fall back as for an uncovered game).
    """
    engine = engine or de._get_engine()
    lead = DECISION_LEAD_MIN if lead_min is None else lead_min
    if not inspect(engine).has_table(TABLE):
        return {}
    lu = pd.read_sql(text(f"SELECT GAME_ID, TEAM_ID, PLAYER_ID, roster_status, ts_utc FROM {TABLE}"), engine)
    sch = pd.read_sql(text("SELECT GAME_ID, TIP_UTC FROM schedule"), engine)
    lu = lu.merge(sch, on="GAME_ID")
    tip = pd.to_datetime(lu["TIP_UTC"], utc=True, errors="coerce")
    ts = pd.to_datetime(lu["ts_utc"], utc=True, errors="coerce")
    lu = lu[ts.notna() & tip.notna() & (ts <= tip - pd.Timedelta(minutes=lead))]
    out: dict[tuple[str, int], set[int]] = {}
    for (gid, tid), g in lu.groupby(["GAME_ID", "TEAM_ID"]):
        out[(str(gid), int(tid))] = set(g.loc[g["roster_status"] == "Inactive", "PLAYER_ID"].astype(int))
    return out
```

(`pd.to_datetime(..., utc=True)` treats the JSON's naive timestamps as UTC, which Task-plan research confirmed.)

- [ ] **Step 4: Union into vacuum absences (`feature_store.py`)**

Add parameter `pregame_inactive: Optional[dict[tuple[str, int], set[int]]] = None,`. Right after `pregame_out` is resolved (the `if pregame_out is None and gamelogs is None and PREGAME_ABSENCES:` block):

```python
    if "inactive" in groups:
        if pregame_inactive is None and gamelogs is None:
            from lineups import pregame_inactive as _load_inactive

            pregame_inactive = _load_inactive()
        merged = {k: set(v) for k, v in (pregame_out or {}).items()}
        for k, v in (pregame_inactive or {}).items():
            merged.setdefault(k, set()).update(v)
        pregame_out = merged
```

- [ ] **Step 5: CLI**

```python
def lineups_backfill(
    start: str = typer.Option(..., "--start", help="First game date YYYY-MM-DD."),
    end: str = typer.Option(..., "--end", help="Last game date YYYY-MM-DD."),
    refetch: bool = typer.Option(False, "--refetch", help="Re-download dates already stored."),
):
    """NBA daily lineups JSON (confirmed starters, Inactive list) into the lineups table."""
    import lineups

    _header(f"Lineups backfill — {start} → {end}")
    _success(f"Stored {lineups.backfill(start, end, refetch=refetch):,} rows in {lineups.TABLE}.")
```

`cli/main.py`: `app.command("lineups-backfill")(cmd.lineups_backfill)`.

- [ ] **Step 6: Run tests + smoke**

Run: `.venv/bin/python -m pytest tests/test_lineups.py -q && .venv/bin/python -m pytest -q`
Then: `.venv/bin/python run_pipeline.py lineups-backfill --start 2026-03-10 --end 2026-03-10 && sqlite3 data/nba_reb.db "select roster_status, count(*) from lineups where GAME_DATE='2026-03-10' group by 1"`
Expected: PASS; Active ≈ 289, Inactive ≈ 100.

- [ ] **Step 7: Commit**

```bash
git add lineups.py feature_store.py pipeline/commands.py cli/main.py tests/test_lineups.py
git commit -m "feat: NBA lineups JSON backfill, live lineups, and Inactive-list absences"
```

---

### Task 8: `game_market` group — Kalshi totals and spreads

**Files:**
- Create: `game_markets.py`
- Modify: `kalshi_history.py` (`fetch_candles(..., path=None)`, `_candle_window(..., lookback_hours=LOOKBACK_HOURS)`, `backfill_candles(..., markets=None, table=..., done_table=..., lookback_hours=..., path_for=None)`)
- Modify: `feature_store.py` (`team_state` adds `t_pts`, `t_pts_allowed`; `_exp_total`; `game_market` param + block)
- Modify: `pipeline/commands.py`, `cli/main.py` (`game-markets-backfill`)
- Test: `tests/test_game_markets.py`

**Interfaces:**
- Consumes: `kalshi_history._get`, `kalshi_history.parse_event_ticker`, `kalshi_history.backfill_candles`, `data_engine.parse_kalshi_event_matchup`, `data_engine.matchup_slug`.
- Produces: `game_markets.SERIES = ("KXNBATOTAL", "KXNBASPREAD")`, `MARKETS_TABLE = "kalshi_game_markets"`, `CANDLES_TABLE = "kalshi_game_candles"`, `CANDLES_DONE = "kalshi_game_candles_done"`, `two_sided_mid(bid, ask) -> float`, `ladder_median(lines, p_over, min_lines=3) -> float`, `event_features(rows: DataFrame[series, line, team, mid], home: str) -> dict` (`mkt_total`, `mkt_home_margin`), `fetch_markets() -> DataFrame`, `backfill(workers=4) -> None`, `history_features(engine=None, lead_min=None) -> DataFrame[GAME_ID, mkt_total, mkt_home_margin]`, `live_features(game_date, schedule_index) -> DataFrame[GAME_ID, mkt_total, mkt_home_margin]`; `build_feature_table(..., game_market: Optional[pd.DataFrame] = None)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_game_markets.py
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

import game_markets as gm


def test_two_sided_mid():
    assert gm.two_sided_mid(0.40, 0.44) == pytest.approx(0.42)
    assert np.isnan(gm.two_sided_mid(0.0, 0.44)) and np.isnan(gm.two_sided_mid(0.4, 1.0))
    assert np.isnan(gm.two_sided_mid(0.5, 0.5))


def test_ladder_median_interpolates():
    assert gm.ladder_median([220.5, 224.5, 228.5], [0.7, 0.5, 0.3]) == pytest.approx(224.5)
    assert gm.ladder_median([220.5, 224.5, 228.5], [0.8, 0.6, 0.4]) == pytest.approx(226.5)


def test_ladder_median_edge_cases():
    assert np.isnan(gm.ladder_median([220.5, 224.5], [0.7, 0.3]))                 # < 3 lines
    assert np.isnan(gm.ladder_median([220.5, 224.5, 228.5], [0.9, 0.8, 0.6]))     # median above ladder
    assert np.isnan(gm.ladder_median([220.5, 224.5, 228.5], [0.4, 0.3, 0.2]))     # median below ladder
    # non-monotone noise is smoothed, not trusted
    assert gm.ladder_median([220.5, 222.5, 224.5, 226.5], [0.7, 0.45, 0.55, 0.3]) == pytest.approx(223.5, abs=1.0)


def test_event_features_combines_spread_sides():
    rows = pd.DataFrame([
        ("KXNBATOTAL", 220.5, "", 0.7), ("KXNBATOTAL", 224.5, "", 0.5), ("KXNBATOTAL", 228.5, "", 0.3),
        # home SAS: P(SAS by > L); away NYK: P(NYK by > L) = P(margin < -L)
        ("KXNBASPREAD", 2.5, "SAS", 0.65), ("KXNBASPREAD", 6.5, "SAS", 0.45), ("KXNBASPREAD", 10.5, "SAS", 0.25),
        ("KXNBASPREAD", 1.5, "NYK", 0.20),
    ], columns=["series", "line", "team", "mid"])
    f = gm.event_features(rows, home="SAS")
    assert f["mkt_total"] == pytest.approx(224.5)
    assert 4.5 < f["mkt_home_margin"] < 6.5


def test_history_features_uses_last_fresh_candle_before_cutoff(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    tip = pd.Timestamp("2026-03-11T00:00:00Z")
    pd.DataFrame([{"GAME_ID": "0022500940", "GAME_DATE_EST": "2026-03-10", "TIP_UTC": "2026-03-11T00:00:00Z",
                   "AWAY_TRICODE": "NYK", "HOME_TRICODE": "SAS"}]).to_sql("schedule", eng, index=False)
    mk = pd.DataFrame({"ticker": [f"KXNBATOTAL-26MAR10NYKSAS-{x}" for x in (220, 224, 228)],
                       "series": "KXNBATOTAL", "game_date": "2026-03-10", "teams_code": "NYKSAS",
                       "line": [220.5, 224.5, 228.5], "team": ""})
    mk.to_sql(gm.MARKETS_TABLE, eng, index=False)
    cut = int((tip - pd.Timedelta(minutes=30)).timestamp())
    hour = cut - cut % 3600
    rows = []
    for t, (p_ok, p_late) in zip(mk["ticker"], [(0.7, 0.9), (0.5, 0.9), (0.3, 0.9)]):
        rows += [{"ticker": t, "end_ts": hour, "yes_bid": p_ok - 0.01, "yes_ask": p_ok + 0.01},
                 {"ticker": t, "end_ts": hour + 3600, "yes_bid": p_late - 0.01, "yes_ask": p_late + 0.01}]
    pd.DataFrame(rows).to_sql(gm.CANDLES_TABLE, eng, index=False)
    out = gm.history_features(eng, lead_min=30).set_index("GAME_ID")
    assert out.loc["0022500940", "mkt_total"] == pytest.approx(224.5)
    assert np.isnan(out.loc["0022500940", "mkt_home_margin"])


def test_game_market_group_columns():
    import feature_store as fs
    from tests.test_feature_store_pit import _synthetic_gamelogs, _tracking

    gl = _synthetic_gamelogs(n_days=20)
    gid = gl["GAME_ID"].iloc[-1]
    gmf = pd.DataFrame({"GAME_ID": [gid], "mkt_total": [230.0], "mkt_home_margin": [-7.0]})
    df = fs.build_feature_table(gamelogs=gl, tracking=_tracking(), apply_filters=False,
                                groups=["game_market"], game_market=gmf)
    g = df[df["GAME_ID"] == gid]
    assert (g["mkt_total"] == 230.0).all() and (g["mkt_spread_abs"] == 7.0).all()
    assert g["mkt_total_resid"].notna().all()
    assert df.loc[df["GAME_ID"] != gid, "mkt_total"].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_game_markets.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'game_markets'`.

- [ ] **Step 3: Generalise `kalshi_history` candle backfill**

```python
def _candle_window(m: dict, tip_utc: str | None, lookback_hours: int = LOOKBACK_HOURS) -> tuple[int, int]:
    ...  # body unchanged except: start = max(_ts(m["open_time"]), end - lookback_hours * 3600)


def fetch_candles(ticker: str, start_ts: int, end_ts: int, path: str | None = None) -> list[dict]:
    if end_ts <= start_ts:
        return []
    data = _get(path or f"/historical/markets/{ticker}/candlesticks",
                {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 60})
    ...  # rest unchanged


def backfill_candles(workers: int = 6, flush_every: int = 500, *, markets: pd.DataFrame | None = None,
                     table: str = "kalshi_candles", done_table: str = "kalshi_candles_done",
                     lookback_hours: int = LOOKBACK_HOURS, path_for=None) -> None:
```

Inside `backfill_candles`: `markets = load_markets() if markets is None else markets`; replace the literal table names with `table`/`done_table`; `job` becomes:

```python
    def job(m):
        tip = tips.get((m["game_date"], m["teams_code"]))
        s, e = _candle_window(m, tip, lookback_hours)
        return m["ticker"], fetch_candles(m["ticker"], s, e, path=path_for(m) if path_for else None)
```

Existing behaviour (no arguments) is unchanged.

- [ ] **Step 4: Implement `game_markets.py`**

```python
"""
game_markets.py
---------------
Kalshi NBA game totals (KXNBATOTAL: "Over L points scored") and spreads (KXNBASPREAD:
"<team> wins by over L points") → per-game market-implied median total and home margin.

History: market lists from /historical/markets (settled before Kalshi's historical
cutoff) plus /markets?status=settled (after it); hourly candles per ticker; the price
at decision time is the last candle closing at or before tip − DECISION_LEAD_MIN and
no more than STALE_SEC older. Live: current books from the public /markets endpoint.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sqlalchemy import inspect, text

import kalshi_history as kh
from config import DECISION_LEAD_MIN

log = logging.getLogger(__name__)

SERIES = ("KXNBATOTAL", "KXNBASPREAD")
MARKETS_TABLE = "kalshi_game_markets"
CANDLES_TABLE = "kalshi_game_candles"
CANDLES_DONE = "kalshi_game_candles_done"
MIN_LADDER = 3
STALE_SEC = 2 * 3600
_TEAM_RE = re.compile(r"^([A-Z]+)\d+$")


def two_sided_mid(bid, ask) -> float:
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return np.nan
    return (bid + ask) / 2 if bid > 0 and ask < 1 and ask > bid else np.nan


def ladder_median(lines, p_over, min_lines: int = MIN_LADDER) -> float:
    """Line where a monotone (isotonic, decreasing) fit of P(X > line) crosses 0.5; NaN outside the ladder."""
    d = pd.DataFrame({"l": np.asarray(lines, float), "p": np.asarray(p_over, float)}).dropna()
    d = d.groupby("l", as_index=False)["p"].mean().sort_values("l")
    if len(d) < min_lines:
        return np.nan
    l = d["l"].to_numpy()
    p = IsotonicRegression(increasing=False, y_min=0.0, y_max=1.0).fit_transform(l, d["p"].to_numpy())
    if p[0] < 0.5 or p[-1] > 0.5:
        return np.nan
    i = int(np.nonzero(p <= 0.5)[0][0])
    if i == 0:
        return float(l[0])
    l0, l1, p0, p1 = l[i - 1], l[i], p[i - 1], p[i]
    return float(l0 + (p0 - 0.5) * (l1 - l0) / (p0 - p1)) if p0 != p1 else float((l0 + l1) / 2)


def _team(ticker: str) -> str:
    m = _TEAM_RE.match(str(ticker).split("-")[-1])
    return m.group(1) if m else ""


def event_features(rows: pd.DataFrame, home: str) -> dict:
    tot = rows[rows["series"] == "KXNBATOTAL"]
    sp = rows[rows["series"] == "KXNBASPREAD"]
    is_home = (sp["team"] == home).to_numpy()
    x = np.where(is_home, sp["line"], -sp["line"])      # home margin M
    p = np.where(is_home, sp["mid"], 1 - sp["mid"])     # P(M > x)
    return {"mkt_total": ladder_median(tot["line"], tot["mid"]), "mkt_home_margin": ladder_median(x, p)}


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def _rows(ms: list[dict], series: str) -> list[dict]:
    out = []
    for m in ms:
        game_date, teams = kh.parse_event_ticker(m.get("event_ticker", ""))
        out.append({"ticker": m["ticker"], "series": series, "event_ticker": m.get("event_ticker"),
                    "game_date": game_date, "teams_code": teams, "line": kh._f(m.get("floor_strike")),
                    "team": _team(m["ticker"]) if series == "KXNBASPREAD" else "",
                    "result": m.get("result"), "open_time": m.get("open_time"), "close_time": m.get("close_time"),
                    "occurrence_time": m.get("occurrence_datetime"), "status": m.get("status")})
    return out


def _paged(path: str, params: dict) -> list[dict]:
    out, cursor = [], None
    while True:
        p = dict(params, limit=1000, **({"cursor": cursor} if cursor else {}))
        page = kh._get(path, p)
        ms = page.get("markets", []) or []
        out.extend(ms)
        cursor = page.get("cursor")
        if not cursor or not ms:
            return out


def fetch_markets() -> pd.DataFrame:
    rows = []
    for s in SERIES:
        rows += _rows(_paged("/historical/markets", {"series_ticker": s}), s)
        rows += _rows(_paged("/markets", {"series_ticker": s, "status": "settled"}), s)
    df = pd.DataFrame(rows).drop_duplicates("ticker", keep="last")
    return df[df["game_date"].notna()]


def backfill(workers: int = 4) -> None:
    """Refresh the market list, then fetch candles for every settled ticker not yet done (resumable)."""
    mk = fetch_markets()
    eng = kh._engine()
    with eng.begin() as conn:
        if inspect(conn).has_table(MARKETS_TABLE):
            conn.execute(text(f"DELETE FROM {MARKETS_TABLE}"))
        mk.to_sql(MARKETS_TABLE, conn, if_exists="append", index=False, chunksize=1000)
    log.info(f"{len(mk):,} game markets stored")
    cutoff = kh._get("/historical/cutoff", {}).get("market_settled_ts", "")

    def path_for(m):
        hist = cutoff and str(m.get("close_time") or "") < cutoff
        return (f"/historical/markets/{m['ticker']}/candlesticks" if hist
                else f"/series/{m['series']}/markets/{m['ticker']}/candlesticks")

    kh.backfill_candles(workers=workers, markets=mk, table=CANDLES_TABLE, done_table=CANDLES_DONE,
                        lookback_hours=4, path_for=path_for)


def history_features(engine=None, lead_min: int | None = None) -> pd.DataFrame:
    engine = engine or kh._engine()
    lead = DECISION_LEAD_MIN if lead_min is None else lead_min
    cols = ["GAME_ID", "mkt_total", "mkt_home_margin"]
    if not (inspect(engine).has_table(MARKETS_TABLE) and inspect(engine).has_table(CANDLES_TABLE)):
        return pd.DataFrame(columns=cols)
    mk = pd.read_sql(text(f"SELECT ticker, series, game_date, teams_code, line, team FROM {MARKETS_TABLE}"), engine)
    sch = pd.read_sql(text("SELECT GAME_ID, GAME_DATE_EST, TIP_UTC, AWAY_TRICODE, HOME_TRICODE FROM schedule"), engine)
    sch["teams_code"] = sch["AWAY_TRICODE"] + sch["HOME_TRICODE"]
    mk = mk.merge(sch.rename(columns={"GAME_DATE_EST": "game_date"}), on=["game_date", "teams_code"])
    mk["cut_ts"] = (pd.to_datetime(mk["TIP_UTC"], utc=True).astype("int64") // 10**9) - lead * 60
    cd = pd.read_sql(text(f"SELECT ticker, end_ts, yes_bid, yes_ask FROM {CANDLES_TABLE}"), engine)
    cd = cd.merge(mk[["ticker", "cut_ts"]], on="ticker")
    cd = cd[(cd["end_ts"] <= cd["cut_ts"]) & (cd["end_ts"] > cd["cut_ts"] - STALE_SEC)]
    last = cd.sort_values("end_ts").groupby("ticker").tail(1)
    last = last.assign(mid=[two_sided_mid(b, a) for b, a in zip(last["yes_bid"], last["yes_ask"])])
    mk = mk.merge(last[["ticker", "mid"]], on="ticker", how="left")
    out = [{"GAME_ID": str(gid), **event_features(g, home=g["HOME_TRICODE"].iloc[0])}
           for gid, g in mk.groupby("GAME_ID")]
    return pd.DataFrame(out, columns=cols)


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

def live_features(game_date: str, schedule_index: dict[str, dict]) -> pd.DataFrame:
    """Current books for ``game_date`` events; schedule_index = data_engine.slate_schedule_index(game_date)."""
    import data_engine as de

    cols = ["GAME_ID", "mkt_total", "mkt_home_margin"]
    rows = []
    for s in SERIES:
        try:
            ms = _paged("/markets", {"series_ticker": s, "status": "open"})
        except Exception as e:  # treated like a missing ladder: NaN features, as in history
            log.warning("Game markets %s unavailable (%s)", s, e)
            continue
        for r in _rows(ms, s):
            m = next(x for x in ms if x["ticker"] == r["ticker"])
            r["mid"] = two_sided_mid(m.get("yes_bid_dollars"), m.get("yes_ask_dollars"))
            rows.append(r)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df[df["game_date"] == game_date]
    out = []
    for et, g in df.groupby("event_ticker"):
        mu = de.parse_kalshi_event_matchup(str(et))
        row = schedule_index.get(de.matchup_slug(*mu)) if mu else None
        if row:
            out.append({"GAME_ID": row["game_id"], **event_features(g, home=mu[1])})
    if not out:
        return pd.DataFrame(columns=cols)
    # a game has one totals event and one spread event: combine their features
    res = pd.DataFrame(out).groupby("GAME_ID", as_index=False).agg(
        {"mkt_total": "max", "mkt_home_margin": "max"})
    return res[cols]
```

Note `live_features` groups by `event_ticker`, so totals and spreads arrive as separate rows per game (each with the other field NaN); the final `groupby(...).max()` merges them (NaN-skipping). Replace the `next(...)` lookup with a dict built once (`by_ticker = {m["ticker"]: m for m in ms}`) to avoid O(n²).

- [ ] **Step 5: Feature-table block (`feature_store.py`)**

In `team_state` add to `cols`: `"PTS": "t_pts", "o_PTS": "t_pts_allowed",`. In the matchup block add:

```python
    df["_exp_total"] = ((own["t_pts"].values + opp["t_pts_allowed"].values) / 2
                        + (opp["t_pts"].values + own["t_pts_allowed"].values) / 2)
```

Add parameter `game_market: Optional[pd.DataFrame] = None,` and before the vacuum block:

```python
    if "game_market" in groups:
        if game_market is None and gamelogs is None:
            from game_markets import history_features

            game_market = history_features()
        if game_market is not None and len(game_market):
            gmf = game_market[["GAME_ID", "mkt_total", "mkt_home_margin"]].copy()
            gmf["GAME_ID"] = gmf["GAME_ID"].astype(str)
            df = df.merge(gmf.drop_duplicates("GAME_ID", keep="last"), on="GAME_ID", how="left")
            df["mkt_spread_abs"] = df["mkt_home_margin"].abs()
            df["mkt_total_resid"] = df["mkt_total"] - df["_exp_total"]
```

- [ ] **Step 6: CLI**

```python
def game_markets_backfill(
    workers: int = typer.Option(4, "--workers", help="Parallel candle fetchers (shared 5 req/s limit)."),
):
    """Kalshi game totals/spreads: refresh the market list and fetch missing hourly candles."""
    import game_markets

    _header("Game-market backfill (KXNBATOTAL, KXNBASPREAD)")
    game_markets.backfill(workers=workers)
    _success("Game-market candles up to date.")
```

`cli/main.py`: `app.command("game-markets-backfill")(cmd.game_markets_backfill)`.

- [ ] **Step 7: Run tests**

Run: `.venv/bin/python -m pytest tests/test_game_markets.py -q && .venv/bin/python -m pytest -q && .venv/bin/python -m pyflakes game_markets.py kalshi_history.py feature_store.py`
Expected: PASS; no lint output.

- [ ] **Step 8: Commit**

```bash
git add game_markets.py kalshi_history.py feature_store.py pipeline/commands.py cli/main.py tests/test_game_markets.py
git commit -m "feat: Kalshi game totals/spreads feature group"
```

---

### Task 9: One decision time for injury reports

**Files:**
- Modify: `injury_reports.py:327` (`pregame_out_for_gamelogs` default lead)
- Modify: `pipeline/commands.py:77` (`injury-backfill --lead-min` default)
- Modify: `config.py` comment on `PREGAME_ABSENCES` ("tip − 60 min" → "tip − DECISION_LEAD_MIN")
- Test: `tests/test_injury_reports.py` (append)

**Interfaces:**
- Consumes: `config.DECISION_LEAD_MIN`.
- Produces: `pregame_out_for_gamelogs(gl, *, lead_min: int | None = None, engine=None)` — `None` means `DECISION_LEAD_MIN`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_injury_reports.py`)

```python
def test_pregame_out_default_lead_is_decision_lead(tmp_path, monkeypatch):
    import injury_reports as ir
    from sqlalchemy import create_engine

    monkeypatch.setattr(ir, "DECISION_LEAD_MIN", 30)
    eng = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    pd.DataFrame([{"GAME_ID": "0022500900", "GAME_DATE_EST": "2026-03-02", "TIP_UTC": "2026-03-03T00:00:00Z",
                   "AWAY_TRICODE": "HOU", "HOME_TRICODE": "WAS", "AWAY_TEAM_ID": 10, "HOME_TEAM_ID": 20}]
                 ).to_sql("schedule", eng, index=False)
    ir.store_report(eng, datetime(2026, 3, 2, 18, 15), [  # T-45: inside a 30-min lead, outside 60
        {"game_date": "2026-03-02", "game_time": "07:00", "matchup": "HOU@WAS", "team": "HOU",
         "player_name": "Steven Adams", "status": "Out"}])
    gl = pd.DataFrame({"PLAYER_ID": [1], "PLAYER_NAME": ["Steven Adams"], "TEAM_ID": [10],
                       "GAME_ID": ["0022500800"], "GAME_DATE": pd.to_datetime(["2026-03-01"]), "SEASON": ["2025-26"]})
    assert ir.pregame_out_for_gamelogs(gl, engine=eng)[("0022500900", 10)] == {1}
    assert ir.pregame_out_for_gamelogs(gl, engine=eng, lead_min=60).get(("0022500900", 10), set()) == set()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_injury_reports.py::test_pregame_out_default_lead_is_decision_lead -q`
Expected: FAIL — `AttributeError: <module 'injury_reports'> has no attribute 'DECISION_LEAD_MIN'`.

- [ ] **Step 3: Implement**

In `injury_reports.py`: `from config import DECISION_LEAD_MIN` (top-level); signature `lead_min: int | None = None`; first body line `lead_min = DECISION_LEAD_MIN if lead_min is None else lead_min`; docstring "at least ``lead_min`` (default DECISION_LEAD_MIN)". In `pipeline/commands.py` `injury_backfill`: `lead_min: int = typer.Option(None, "--lead-min", help="... (default DECISION_LEAD_MIN)")` and `lead_min = DECISION_LEAD_MIN if lead_min is None else lead_min` (import from config).

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add injury_reports.py pipeline/commands.py config.py tests/test_injury_reports.py
git commit -m "feat: injury-report cutoff follows DECISION_LEAD_MIN"
```

---

### Task 10: Live scan — tip window, lineup gate, live sources, data-staleness preflight

**Files:**
- Modify: `config.py` (`SCAN_TIP_WINDOW_MIN`)
- Modify: `settings_validate.py` (validate window)
- Modify: `data_engine.py` (`filter_market_lines_by_start_window(..., min_lead_minutes=0.0)`)
- Modify: `pipeline/commands.py` (`scan` window block; dry-run email only when new paper orders)
- Modify: `slate.py` (`build_slate_predictions`: lineups, inactive, game markets)
- Modify: `calibrate_preflight.py` (`group_data_lag_days`, gate)
- Test: `tests/test_scan_time_window.py` (append), `tests/test_slate_new_sources.py`, `tests/test_feature_groups.py` (append)

**Interfaces:**
- Consumes: `lineups.todays_lineups`, `lineups.GameLineup`, `game_markets.history_features`, `game_markets.live_features`, `feature_store.ENABLED_GROUPS`, `calibrate_preflight.latest_completed_game_date`.
- Produces: `config.SCAN_TIP_WINDOW_MIN: tuple[float, float] | None`; excluded reason `"too_close"`; `slate.lineup_gate(resolved, states, groups) -> tuple[set[str], list[str]]` (game ids allowed, skip messages); `calibrate_preflight.GROUP_TABLES`, `group_data_lag_days(engine, today) -> dict[str, int | None]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scan_time_window.py`:

```python
def test_start_window_excludes_games_too_close_to_tip():
    now = datetime(2026, 5, 24, 12, 0, 0, tzinfo=timezone.utc)
    idx = {"BOSNYK": {"status": "Pre-Game", "start_utc": now + timedelta(minutes=3)},
           "SASPHX": {"status": "Pre-Game", "start_utc": now + timedelta(minutes=30)}}
    lines = [_Line("A", "KXNBAREB-26MAY24BOSNYK"), _Line("B", "KXNBAREB-26MAY24SASPHX")]
    kept, excluded = filter_market_lines_by_start_window(
        lines, "2026-05-24", within_hours=1.0, min_lead_minutes=5, now=now, schedule_index=idx)
    assert [k.ticker for k in kept] == ["B"]
    assert excluded[0][3] == "too_close"
```

Create `tests/test_slate_new_sources.py`:

```python
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
```

Append to `tests/test_feature_groups.py`:

```python
def test_group_data_lag_days(tmp_path):
    import calibrate_preflight as cp
    from sqlalchemy import create_engine

    eng = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    pd.DataFrame({"GAME_ID": ["0022500001", "0022500002"], "GAME_DATE_EST": ["2026-03-01", "2026-03-05"]}
                 ).to_sql("schedule", eng, index=False)
    pd.DataFrame({"GAME_DATE": ["2026-03-01"]}).to_sql("tracking_daily", eng, index=False)
    lags = cp.group_data_lag_days(eng, today="2026-03-06", groups=["tracking", "lineup", "travel"])
    assert lags == {"tracking": 4, "lineup": None}
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scan_time_window.py tests/test_slate_new_sources.py tests/test_feature_groups.py -q`
Expected: FAIL — unexpected keyword `min_lead_minutes`; `ImportError: cannot import name 'lineup_gate'`; no `group_data_lag_days`.

- [ ] **Step 3: Tip window**

`config.py` after `SCAN_WITHIN_HOURS`:

```python
# Scheduled scans: only games tipping in (lo, hi] minutes, as "lo,hi". Lineups are confirmed
# ~60-70 min before tip, so the window sits after confirmation; orders expire at tip − 60 s.
# Empty string falls back to SCAN_WITHIN_HOURS (manual full-slate scans).
_tw = os.getenv("SCAN_TIP_WINDOW_MIN", "5,60").strip()
SCAN_TIP_WINDOW_MIN = tuple(float(x) for x in _tw.split(",")) if _tw else None
```

`settings_validate.py`: add a check that, when set, it has two values with `0 <= lo < hi` (follow the existing pattern in that file for `SCAN_WITHIN_HOURS`).

`data_engine.filter_market_lines_by_start_window`: add keyword `min_lead_minutes: float = 0.0`; after the `delta_sec < 0` branch:

```python
        if delta_sec < float(min_lead_minutes) * 60.0:
            if et not in seen_events:
                seen_events.add(et)
                excluded.append((et, key, start_iso, "too_close"))
            continue
```

`pipeline/commands.py` `scan`: import `SCAN_TIP_WINDOW_MIN` next to `SCAN_WITHIN_HOURS` and replace the `if SCAN_WITHIN_HOURS > 0:` block with:

```python
        window = SCAN_TIP_WINDOW_MIN or ((0.0, SCAN_WITHIN_HOURS * 60.0) if SCAN_WITHIN_HOURS > 0 else None)
        if window:
            from data_engine import filter_market_lines_by_start_window

            lo, hi = window
            market_lines, excluded_window = filter_market_lines_by_start_window(
                market_lines, game_date, within_hours=hi / 60.0, min_lead_minutes=lo
            )
            if excluded_window:
                console.print(
                    f"[dim]Entry window: excluded {len(excluded_window)} game(s) not tipping in "
                    f"{lo:g}–{hi:g} min; {len(market_lines)} market(s) remain.[/dim]\n"
                )
            if not market_lines:
                _warn(f"No markets tipping in {lo:g}–{hi:g} min (SCAN_TIP_WINDOW_MIN).")
                raise typer.Exit(0)
```

Dry-run email (10-minute cadence would send ~70 a day): change the dry-run branch's `_email_scan_summary(...)` to run only `if n_paper:`.

- [ ] **Step 4: Slate wiring (`slate.py`)**

```python
LINEUP_GROUPS = {"lineup", "inactive"}


def lineup_gate(resolved: dict[str, SlateRow], states, groups) -> tuple[set[str], list[str]]:
    """Game ids that may be priced, and skip messages. With lineup/inactive features on, a game
    is priced only once both lineups are confirmed (the features don't exist before that);
    an unreadable lineups JSON confirms nothing."""
    games = {r.game_id for r in resolved.values()}
    if not (set(groups) & LINEUP_GROUPS):
        return games, []
    ok = {g for g in games if states and g in states and states[g].confirmed}
    return ok, sorted(f"{t}: lineups not confirmed" for t, r in resolved.items() if r.game_id not in ok)
```

In `build_slate_predictions`, after `resolve_slate`:

```python
    from feature_store import ENABLED_GROUPS

    groups = set(ENABLED_GROUPS)
    states = None
    if groups & LINEUP_GROUPS:
        import lineups

        states = lineups.todays_lineups(game_date)
    ok_games, gate_skips = lineup_gate(resolved, states, groups)
    skipped += gate_skips
    resolved = {t: r for t, r in resolved.items() if r.game_id in ok_games}
    if not resolved:
        return [], [], skipped
```

After `absent, report_ok = absences_by_team(...)` (and the `require_absences` check):

```python
    if "inactive" in groups and states:
        for gid in ok_games:
            for team, pids in states[gid].inactive.items():
                absent.setdefault(int(team), set()).update(pids)
    live_starters = ({gid: states[gid].starters for gid in ok_games} if "lineup" in groups and states else None)
    game_market = None
    if "game_market" in groups:
        import game_markets

        game_market = pd.concat([game_markets.history_features(),
                                 game_markets.live_features(game_date, de.slate_schedule_index(game_date))],
                                ignore_index=True)
```

and pass `live_starters=live_starters, game_market=game_market` to `build_feature_table`. Move `de.slate_schedule_index(game_date)` into a local `idx` used by both calls.

- [ ] **Step 5: Data-staleness preflight (`calibrate_preflight.py`)**

```python
# Enabled feature groups read nightly-backfilled tables; trading on a stale one prices
# tonight with features that stopped updating.
GROUP_TABLES = {"tracking": ("tracking_daily", "GAME_DATE"), "lineup": ("starters", "GAME_DATE"),
                "inactive": ("lineups", "GAME_DATE"), "game_market": ("kalshi_game_markets", "game_date")}
GROUP_MAX_LAG_DAYS = 3


def group_data_lag_days(engine, today: str, groups) -> dict[str, int | None]:
    from sqlalchemy import inspect, text

    expected = latest_completed_game_date(engine, today=today)
    out: dict[str, int | None] = {}
    for g in groups:
        if g not in GROUP_TABLES or not expected:
            continue
        table, col = GROUP_TABLES[g]
        have = None
        if inspect(engine).has_table(table):
            with engine.connect() as conn:
                have = conn.execute(text(f"SELECT MAX({col}) FROM {table}")).scalar()
        out[g] = None if not have else (datetime.strptime(expected, "%Y-%m-%d")
                                         - datetime.strptime(str(have)[:10], "%Y-%m-%d")).days
    return out
```

In `check_calibrate_preflight`, after the gamelogs-vs-schedule gate:

```python
    if expected:
        from zoneinfo import ZoneInfo

        from data_engine import _get_engine
        from feature_store import ENABLED_GROUPS

        today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        for g, lag in group_data_lag_days(_get_engine(), today, ENABLED_GROUPS).items():
            if lag is None:
                _flag(f"Feature group '{g}' is enabled but {GROUP_TABLES[g][0]} is empty; run its backfill")
            elif lag > GROUP_MAX_LAG_DAYS:
                _flag(f"Feature group '{g}': {GROUP_TABLES[g][0]} is {lag}d behind the schedule; run its backfill")
            elif lag >= 1:
                warnings.append(f"Feature group '{g}': {GROUP_TABLES[g][0]} is {lag}d behind (backfills lag a day)")
```

(`expected` is `_latest_completed_scheduled_date()`, which conftest stubs to `None`, so existing tests skip this.)

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest -q && .venv/bin/python -m pyflakes slate.py data_engine.py calibrate_preflight.py pipeline/commands.py config.py settings_validate.py`
Expected: PASS; no lint output. If `test_scan_live_wiring.py` or `test_settings_validate.py` assumed the old window, update their expectations to the `SCAN_TIP_WINDOW_MIN` behaviour (not the production code).

- [ ] **Step 7: Dry-run smoke on a real slate**

Run: `FEATURE_GROUPS=travel,opp_vacuum,tracking,lineup,inactive,game_market SCAN_TIP_WINDOW_MIN= .venv/bin/python run_pipeline.py scan --date 2026-10-14 --dry-run 2>&1 | tail -30`
Expected: runs to completion; preseason markets either skipped as "lineups not confirmed" or priced; no traceback.

- [ ] **Step 8: Commit**

```bash
git add config.py settings_validate.py data_engine.py pipeline/commands.py slate.py calibrate_preflight.py tests/
git commit -m "feat: scan after confirmed lineups with live game-market and Inactive inputs"
```

---

### Task 11: Nightly backfills, 10-minute scan cron, docs

**Files:**
- Modify: `scripts/cron_job.sh` (`nightly` additions, `scan`/`scan-live` lock)
- Modify: `scripts/crontab.phase1.example` (scan cadence)
- Modify: `WORKFLOW.md` (schedule table, new commands), `SYSTEM.md` §3 (feature groups, decision time)

**Interfaces:**
- Consumes: CLI commands `tracking-backfill`, `starters-backfill`, `lineups-backfill`, `game-markets-backfill`, `injury-backfill`.

- [ ] **Step 1: Nightly additions** — in the `nightly)` case, after the `injury-backfill` line:

```bash
    # Feature-group sources for the last few days (each skips dates already stored).
    BACKFILL_START="$("$PYTHON" -c "
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
print((datetime.now(ZoneInfo('America/New_York')).date() - timedelta(days=${RECONCILE_LOOKBACK_DAYS})).isoformat())
")"
    run_py tracking-backfill --start "$BACKFILL_START" --end "$YESTERDAY_ET" || log "tracking-backfill failed"
    run_py starters-backfill --start "$BACKFILL_START" --end "$YESTERDAY_ET" || log "starters-backfill failed"
    run_py lineups-backfill --start "$BACKFILL_START" --end "$YESTERDAY_ET" || log "lineups-backfill failed"
    run_py game-markets-backfill || log "game-markets-backfill failed"
```

- [ ] **Step 2: Scan lock** — add after `SCAN_LOG=...`:

```bash
# Scans run every 10 minutes; never let two overlap (a slow stats.nba.com call can stretch one).
scan_lock() {
  local lock="${LOG_DIR}/.scan.lock"
  if [[ -f "$lock" ]] && kill -0 "$(cat "$lock")" 2>/dev/null; then
    echo "[$TS] [$JOB] previous scan (pid $(cat "$lock")) still running; skipping"
    exit 0
  fi
  echo $$ >"$lock"
  trap 'rm -f "$lock"' EXIT
}
```

and call `scan_lock` as the first line of the `scan)` and `scan-live)` cases.

- [ ] **Step 3: Cron cadence** — in `scripts/crontab.phase1.example` replace the hourly scan block with:

```
# Dry-run scans every 10 min, 11:00-23:50 ET (-> 08:00-20:50 PDT). Only games tipping in
# SCAN_TIP_WINDOW_MIN (5-60 min) are scanned, after lineups are confirmed.
*/10 8-20 * * * TZ=America/New_York @@REPO@@/scripts/cron_job.sh scan >> @@REPO@@/logs/scan.log 2>&1
```

- [ ] **Step 4: Verify the script parses**

Run: `bash -n scripts/cron_job.sh && echo ok`
Expected: `ok`.

- [ ] **Step 5: Docs** — `WORKFLOW.md`: schedule row "every 10 min 11:00–23:50 ET | `scan` (dry-run) — games tipping in 5–60 min with confirmed lineups"; data commands table gains the four backfills; mention `FEATURE_GROUPS` and that `train` must follow any change to it. `SYSTEM.md` §3: a "Feature groups" subsection (one line per group: source, PIT rule, history coverage) and the `DECISION_LEAD_MIN` rule. (§5 results are written in Task 13.)

- [ ] **Step 6: Commit**

```bash
git add scripts/cron_job.sh scripts/crontab.phase1.example WORKFLOW.md SYSTEM.md
git commit -m "chore: nightly feature-source backfills and 10-minute scan cadence"
```

(The installed crontab is switched in Task 13, after the ablation.)

---

### Task 12: Historical backfills (operational, background)

No code. Start long jobs early — they can run while Tasks 9–11 are implemented, as soon as Tasks 5–8 are committed.

- [ ] **Step 1: nba_api sources, sequential (one IP, shared rate limit)** — run in background:

```bash
cd /Users/maxgarcia/dataProjects/ml/nba_reb_line && \
.venv/bin/python run_pipeline.py starters-backfill --start 2021-10-01 --end 2026-06-30 && \
.venv/bin/python run_pipeline.py tracking-backfill --start 2021-10-01 --end 2026-06-30 && \
.venv/bin/python run_pipeline.py lineups-backfill --start 2021-10-01 --end 2026-06-30 && \
.venv/bin/python run_pipeline.py injury-backfill --start 2025-10-21 --end 2026-06-13 --lead-min 30
```

Expected ≈ 850 dates × 3 sources (~1.5–2 h) + injury PDFs (~20–30 min).

- [ ] **Step 2: Kalshi game markets, in parallel (different host)**

```bash
.venv/bin/python run_pipeline.py game-markets-backfill --workers 4
```

Expected ≈ 32k tickers at 5 req/s (~1.8 h). Resumable: re-run to continue after an interruption.

- [ ] **Step 3: Coverage checks**

```bash
sqlite3 data/nba_reb.db "select substr(GAME_DATE,1,7) m, count(*) from starters group by m order by m" | head -60
sqlite3 data/nba_reb.db "select count(distinct GAME_DATE), min(GAME_DATE), max(GAME_DATE) from tracking_daily"
sqlite3 data/nba_reb.db "select substr(GAME_DATE,1,4), roster_status, count(*) from lineups group by 1,2"
sqlite3 data/nba_reb.db "select count(*), count(distinct ticker) from kalshi_game_candles"
sqlite3 data/nba_reb.db "select source, count(*) from daily_backfill_done group by 1"
```

Expected: starters ≈ 10 per game every game date; tracking dates = starters dates; lineups non-empty for 2023-24 and 2025-26; candles for most of the ~32k tickers. Any month with zero rows → re-run that range (failed dates stay undone).

---

### Task 13: Ablation, measurements, decision, rollout

No production code beyond what the results dictate.

- [ ] **Step 1: Walk-forward ablation (c7)**

```bash
cd /Users/maxgarcia/dataProjects/ml/nba_reb_line
for cfg in "base:" "travel:travel" "opp_vacuum:opp_vacuum" "tracking:tracking" "lineup:lineup" \
           "inactive:inactive" "game_market:game_market" \
           "all:travel,opp_vacuum,tracking,lineup,inactive,game_market"; do
  tag="${cfg%%:*}"; groups="${cfg#*:}"
  FEATURE_GROUPS="$groups" .venv/bin/python bakeoff.py features
  FEATURE_GROUPS="$groups" .venv/bin/python bakeoff.py run --models c7_ens_c4_c5 --tag "$tag"
done
.venv/bin/python bakeoff.py ablation --model c7_ens_c4_c5 --base-tag base
.venv/bin/python bakeoff.py ablation --model c7_ens_c4_c5 --base-tag base --since-score 2025-12-01
```

Expected: a table of `d_ll` with CIs per tag. The `--since-score 2025-12-01` view is the one that counts for `game_market` (and `inactive`, whose history coverage is 2023-24 and 2025-26).

- [ ] **Step 2: Measurements for SYSTEM.md** (scratch scripts in the scratchpad, not committed)
  - Starter agreement, 2025-26: share of team-games where JSON `Confirmed` starters (ts ≤ tip − 30) equal `starters` table rows.
  - Hindsight-gap closure: re-run the Tier 4 hindsight diagnostic with `pregame_out ∪ pregame_inactive`; report the new share of covered team-games with box-score absences not on the pre-tip list (was 36%).
  - Game-market coverage: share of 2025-26 games with non-NaN `mkt_total` / `mkt_home_margin` at T-30.

- [ ] **Step 3: Ship decision (rule from spec §7)** — winners = tags with `ci_hi < 0`, plus tags whose `d_ll < 0` and that improve market comparison:

```bash
FEATURE_GROUPS=<winners> .venv/bin/python bakeoff.py run --models c7_ens_c4_c5 --tag winners
.venv/bin/python market_eval.py --lead 30 --models c7_ens_c4_c5__base,c7_ens_c4_c5__winners
```

Report both tables to the user and stop for their go/no-go before Step 4.

- [ ] **Step 4: Enable, retrain, refit, backtest** (after user go)

```bash
cp models/*.pkl models/*.json models/backup-2026-10-14/ 2>/dev/null || (mkdir -p models/backup-2026-10-14 && cp models/*.pkl models/*.json models/backup-2026-10-14/)
# .env: FEATURE_GROUPS=<winners>
.venv/bin/python run_pipeline.py train
.venv/bin/python run_pipeline.py refit-blend
.venv/bin/python run_pipeline.py backtest --start 2026-04-01 --end 2026-06-13
.venv/bin/python -c "from calibrate_preflight import check_calibrate_preflight as c; r=c(live=True); print(r.errors, r.warnings)"
```

Expected: preflight shows only the known "fill calibrator missing" warning/error.

- [ ] **Step 5: Switch the installed cron** (user's crontab; confirm with the user first)

```bash
scripts/install_phase1_cron.sh && crontab -l | grep cron_job.sh
```

Expected: `*/10 8-20 * * * ... scan`.

- [ ] **Step 6: Write results to SYSTEM.md §5 and commit**

```bash
git add SYSTEM.md
git commit -m "docs: feature-group ablation results and new-source measurements"
```
