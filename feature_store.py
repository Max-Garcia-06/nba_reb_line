"""
feature_store.py
----------------
Builds the player-game feature table for the rebound model.

Point-in-time by construction
-----------------------------
Every feature is computed as a *state after game g* (player state, team
state), then attached to a target row with an as-of join on games STRICTLY
BEFORE the target game's date. A target row therefore cannot see its own
game or anything later. The same code path builds tonight's scan rows (pass
`extra_rows`), so training and serving can't drift apart.

The previous version leaked: opponent shooting, pace and tracking stats were
full-season aggregates joined on the SAME season (end-of-season numbers fed
to early-season games), and the teammate-absence "vacuum" was 0 in training
but non-zero at inference. mlb_tb_line hit the same class of problem
(SYSTEM.md §9) — see tests/test_feature_store_pit.py for the guard.

Feature groups
--------------
Player form      reb/oreb/dreb rolling + EWM, per-36 rate, minutes mean/std,
                 share of team rebounds, season-to-date avg, rest
Prior season     tracking (rebound chances, contested %, deferred) + REB/G
                 from the PREVIOUS season only
Matchup          expected own/opp missed FGs, pace, opp OREB%/DREB%,
                 opp 3PA rate, expected |margin| (blowout risk)
Teammate vacuum  rebounds/minutes of rotation teammates absent from the game
                 (history: didn't appear in the box score; live: injury list)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import data_engine as de
from config import DECISION_LEAD_MIN, FEATURE_GROUPS as _CONFIG_GROUPS, MIN_GAMES, PREGAME_ABSENCES, ROLLING_WINDOW

TEAM_WINDOW = 15
EWM_HALFLIFE = 6.0
ROTATION_MIN = 15.0          # trailing MPG to count as a rotation player
ROTATION_MAX_GAP_DAYS = 21   # last appearance within this many days
MIN_TRAILING_MINUTES = 15.0  # training population, matches Kalshi prop universe

ID_COLS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "OPP_TEAM_ID", "GAME_ID", "GAME_DATE", "SEASON"]
TARGET = "REB"

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


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_NUM = ["MIN", "REB", "OREB", "DREB", "FGA", "FGM", "FG3A", "FTA", "TOV", "PTS"]


def clean_gamelogs(gl: pd.DataFrame) -> pd.DataFrame:
    gl = gl.copy()
    for c in _NUM:
        gl[c] = pd.to_numeric(gl[c], errors="coerce").fillna(0.0)
    gl["PLAYER_ID"] = gl["PLAYER_ID"].astype(int)
    gl["TEAM_ID"] = gl["TEAM_ID"].astype(int)
    gl["GAME_ID"] = gl["GAME_ID"].astype(str)
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"]).dt.normalize()
    gl["is_home"] = gl["MATCHUP"].str.contains(r"vs\.", na=False).astype(int)
    gl["is_playoffs"] = (gl.get("SEASON_TYPE", "") == "Playoffs").astype(int)
    return gl.drop_duplicates(["PLAYER_ID", "GAME_ID"]).sort_values(["GAME_DATE", "GAME_ID"])


def prev_season(season: str) -> str:
    y = int(season[:4]) - 1
    return f"{y}-{str(y + 1)[-2:]}"


# ---------------------------------------------------------------------------
# Team-game table and team state
# ---------------------------------------------------------------------------

def team_games(gl: pd.DataFrame) -> pd.DataFrame:
    """One row per (TEAM_ID, GAME_ID) with box totals and the opponent's totals."""
    agg = (
        gl.groupby(["TEAM_ID", "GAME_ID", "GAME_DATE", "SEASON"], as_index=False)
        [["FGA", "FGM", "FG3A", "FTA", "OREB", "DREB", "REB", "TOV", "PTS"]].sum()
    )
    opp = agg.rename(columns={c: f"o_{c}" for c in agg.columns if c not in ("GAME_ID",)})
    tg = agg.merge(opp, on="GAME_ID")
    tg = tg[tg["TEAM_ID"] != tg["o_TEAM_ID"]].rename(columns={"o_TEAM_ID": "OPP_TEAM_ID"})
    tg = tg.drop(columns=["o_GAME_DATE", "o_SEASON"])
    tg["poss"] = 0.5 * ((tg["FGA"] + 0.44 * tg["FTA"] - tg["OREB"] + tg["TOV"])
                        + (tg["o_FGA"] + 0.44 * tg["o_FTA"] - tg["o_OREB"] + tg["o_TOV"]))
    tg["missed"] = tg["FGA"] - tg["FGM"]
    tg["o_missed"] = tg["o_FGA"] - tg["o_FGM"]
    tg["oreb_pct"] = tg["OREB"] / (tg["OREB"] + tg["o_DREB"]).replace(0, np.nan)
    tg["dreb_pct"] = tg["DREB"] / (tg["DREB"] + tg["o_OREB"]).replace(0, np.nan)
    tg["fg3a_rate"] = tg["FG3A"] / tg["FGA"].replace(0, np.nan)
    tg["margin"] = tg["PTS"] - tg["o_PTS"]
    return tg.sort_values(["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)


def team_state(tg: pd.DataFrame, window: int = TEAM_WINDOW) -> pd.DataFrame:
    """Team state AFTER each game (inclusive rolling means)."""
    cols = {
        "missed": "t_missed",            # own offense misses -> own OREB chances
        "o_missed": "t_missed_allowed",  # misses this defense forces
        "poss": "t_pace",
        "oreb_pct": "t_oreb_pct",
        "dreb_pct": "t_dreb_pct",
        "fg3a_rate": "t_3pa_rate",
        "margin": "t_margin",
        "PTS": "t_pts",
        "o_PTS": "t_pts_allowed",
    }
    g = tg.groupby("TEAM_ID")
    st = tg[["TEAM_ID", "GAME_DATE"]].copy()
    for src, dst in cols.items():
        st[dst] = g[src].transform(lambda x: x.rolling(window, min_periods=3).mean())
    return st


# ---------------------------------------------------------------------------
# Player state
# ---------------------------------------------------------------------------

def player_state(gl: pd.DataFrame, tg: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Player state AFTER each game (inclusive)."""
    df = gl.merge(tg[["TEAM_ID", "GAME_ID", "REB"]].rename(columns={"REB": "team_reb"}),
                  on=["TEAM_ID", "GAME_ID"], how="left")
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    g = df.groupby("PLAYER_ID")

    def roll(col, fn="mean"):
        return g[col].transform(lambda x: getattr(x.rolling(window, min_periods=3), fn)())

    def rsum(col):
        return g[col].transform(lambda x: x.rolling(window, min_periods=3).sum())

    def ewm(col):
        return g[col].transform(lambda x: x.ewm(halflife=EWM_HALFLIFE, min_periods=3).mean())

    st = df[["PLAYER_ID", "GAME_DATE", "TEAM_ID", "SEASON"]].copy()
    st["reb_roll"] = roll("REB")
    st["reb_std_roll"] = roll("REB", "std")
    st["oreb_roll"] = roll("OREB")
    st["dreb_roll"] = roll("DREB")
    st["min_roll"] = roll("MIN")
    st["min_std_roll"] = roll("MIN", "std")
    st["reb_ewm"] = ewm("REB")
    st["min_ewm"] = ewm("MIN")
    st["reb36_roll"] = 36 * rsum("REB") / rsum("MIN").replace(0, np.nan)
    st["reb36_ewm"] = 36 * st["reb_ewm"] / st["min_ewm"].replace(0, np.nan)
    st["reb_share_roll"] = rsum("REB") / g["team_reb"].transform(
        lambda x: x.rolling(window, min_periods=3).sum()).replace(0, np.nan)
    gs = df.groupby(["PLAYER_ID", "SEASON"])
    st["reb_season_avg"] = gs["REB"].transform(lambda x: x.expanding().mean())
    st["games_in_season"] = gs.cumcount() + 1
    st["games_played"] = g.cumcount() + 1
    st["last_game_date"] = df["GAME_DATE"]
    st["last_team_id"] = df["TEAM_ID"]
    return st.drop(columns=["TEAM_ID", "SEASON"])


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
    for pid, tid, gid, date, live in zip(df["PLAYER_ID"], df["TEAM_ID"], df["GAME_ID"], df["GAME_DATE"], df["_live"]):
        tn = tonight(int(tid), gid, date, live)
        starts.append(np.nan if tn is None else float(int(pid) in tn))
        key = (int(tid), str(gid))
        if key not in usual_out:
            s = by_team.get(int(tid))
            prev = s[s.index < date].iloc[-USUAL_STARTER_GAMES:] if s is not None else []
            counts = pd.Series([p for ps in prev for p in ps], dtype="int64").value_counts(sort=False)
            usual = set(counts.sort_values(ascending=False, kind="stable").index[:5]) if len(counts) else set()
            usual_out[key] = np.nan if tn is None or not usual else float(len(usual - set(tn)))
    out["starts_tonight"] = starts
    out["team_usual_starters_out"] = [usual_out[(int(t), str(g))] for t, g in zip(df["TEAM_ID"], df["GAME_ID"])]
    out["role_change"] = out["starts_tonight"] - out["start_rate_roll"]
    return out[FEATURE_GROUPS["lineup"]].set_index(df.index)


# ---------------------------------------------------------------------------
# As-of joins
# ---------------------------------------------------------------------------

def _asof(left: pd.DataFrame, right: pd.DataFrame, by: str, left_by: Optional[str] = None) -> pd.DataFrame:
    """Attach the latest `right` state with GAME_DATE strictly before left's GAME_DATE."""
    left_by = left_by or by
    l = left.reset_index().rename(columns={"index": "_row"}).sort_values("GAME_DATE")
    r = right.sort_values("GAME_DATE").rename(columns={"GAME_DATE": "_state_date"})
    if left_by != by:
        r = r.rename(columns={by: left_by})
    out = pd.merge_asof(
        l, r, left_on="GAME_DATE", right_on="_state_date", by=left_by,
        allow_exact_matches=False, direction="backward",
    )
    return out.sort_values("_row").set_index("_row").drop(columns=["_state_date"])


# ---------------------------------------------------------------------------
# Teammate vacuum
# ---------------------------------------------------------------------------

def _next_team(cand: pd.DataFrame, appearances: pd.DataFrame) -> pd.Series:
    """TEAM_ID of each candidate's first game strictly after the row's date (NaN if none)."""
    left = cand[["PLAYER_ID", "GAME_DATE"]].reset_index().rename(columns={"index": "_row"})
    right = appearances[["PLAYER_ID", "GAME_DATE", "TEAM_ID"]].rename(columns={"TEAM_ID": "_next_team"})
    right = right.sort_values("GAME_DATE")
    out = pd.merge_asof(left.sort_values("GAME_DATE"), right, on="GAME_DATE", by="PLAYER_ID",
                        allow_exact_matches=False, direction="forward")
    return out.set_index("_row")["_next_team"].reindex(cand.index)


def teammate_vacuum(
    targets: pd.DataFrame,
    pstate: pd.DataFrame,
    roster_pool: pd.DataFrame,
    played: set[tuple[int, str]],
    absent_override: Optional[dict[int, set[int]]] = None,
    pregame_out: Optional[dict[tuple[str, int], set[int]]] = None,
    appearances: Optional[pd.DataFrame] = None,
    current_team: Optional[dict[int, int]] = None,
) -> pd.DataFrame:
    """
    For each (TEAM_ID, GAME_ID) in targets: rebounds and minutes of rotation
    teammates who are NOT playing, as knowable before tip.

    * Live games: listed Out (``absent_override`` {TEAM_ID: {PLAYER_ID}}, from the
      official injury report) or now on another team's roster (``current_team``).
    * Historical games the injury reports cover (``pregame_out`` {(GAME_ID, TEAM_ID):
      {PLAYER_ID}}): listed Out before the decision time, or departed — their next
      game is for another team (traded/waived; public before tip).
    * Older games without report coverage: no box-score row (includes late
      scratches — hindsight live does not have).
    """
    tgames = targets[["TEAM_ID", "GAME_ID", "GAME_DATE", "SEASON"]].drop_duplicates(["TEAM_ID", "GAME_ID"])
    cand = tgames.merge(roster_pool, on=["TEAM_ID", "SEASON"])
    cand = _asof(cand, pstate[["PLAYER_ID", "GAME_DATE", "reb_roll", "min_roll",
                               "last_game_date", "last_team_id"]], by="PLAYER_ID")
    gap = (cand["GAME_DATE"] - cand["last_game_date"]).dt.days
    rot = cand[(cand["last_team_id"] == cand["TEAM_ID"]) & (cand["min_roll"] >= ROTATION_MIN)
               & (gap <= ROTATION_MAX_GAP_DAYS)].copy()

    live_games = set(targets.loc[targets["_live"], "GAME_ID"])
    override = absent_override or {}
    reports = pregame_out or {}
    now_team = current_team or {}
    if reports and appearances is not None and len(rot):
        nxt = _next_team(rot, appearances)
        departed_hist = nxt.notna() & (nxt != rot["TEAM_ID"])
    else:
        departed_hist = pd.Series(False, index=rot.index)

    absent = []
    for pid, tid, gid, dep in zip(rot["PLAYER_ID"], rot["TEAM_ID"], rot["GAME_ID"], departed_hist):
        if gid in live_games:
            absent.append(pid in override.get(tid, set()) or now_team.get(pid, tid) != tid)
        elif (gid, tid) in reports:
            absent.append(pid in reports[(gid, tid)] or bool(dep))
        else:
            absent.append((pid, gid) not in played)
    rot["absent"] = absent

    ab = rot[rot["absent"]]
    vac = ab.groupby(["TEAM_ID", "GAME_ID"]).agg(
        vacuum_reb=("reb_roll", "sum"), vacuum_min=("min_roll", "sum"), vacuum_n=("PLAYER_ID", "size"),
    ).reset_index()
    return vac


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_feature_table(
    extra_rows: Optional[pd.DataFrame] = None,
    absent_override: Optional[dict[int, set[int]]] = None,
    gamelogs: Optional[pd.DataFrame] = None,
    tracking: Optional[pd.DataFrame] = None,
    apply_filters: bool = True,
    pregame_out: Optional[dict[tuple[str, int], set[int]]] = None,
    current_team: Optional[dict[int, int]] = None,
    groups: Optional[list[str]] = None,
    schedule: Optional[pd.DataFrame] = None,
    tracking_daily: Optional[pd.DataFrame] = None,
    starters: Optional[pd.DataFrame] = None,
    live_starters: Optional[dict[str, set[int]]] = None,
    pregame_inactive: Optional[dict[tuple[str, int], set[int]]] = None,
    game_market: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    extra_rows      : tonight's target rows (no box score yet). Columns:
                      PLAYER_ID, PLAYER_NAME, TEAM_ID, OPP_TEAM_ID, GAME_ID,
                      GAME_DATE, SEASON, is_home, is_playoffs. REB is NaN.
    absent_override : {TEAM_ID: {PLAYER_ID}} confirmed-out players for extra_rows games.
    pregame_out     : {(GAME_ID, TEAM_ID): {PLAYER_ID}} listed Out before the decision time
                      for historical games (default: loaded from the injury_reports table
                      when gamelogs come from SQLite and PREGAME_ABSENCES is on).
    current_team    : {PLAYER_ID: TEAM_ID} from tonight's rosters (live traded players).
    gamelogs, tracking : inject frames (tests); default loads from SQLite.
    apply_filters   : drop thin-history / low-minute rows (training population).
                      Extra rows are never filtered.
    groups          : optional feature groups to attach (default: ENABLED_GROUPS).
    schedule        : schedule for the travel group; default loaded from SQLite when
                      gamelogs come from SQLite.
    tracking_daily  : per-game rebound tracking for the tracking group (default: the
                      tracking_daily table when gamelogs come from SQLite).
    starters        : box-score starters (PLAYER_ID, TEAM_ID, GAME_DATE) for the lineup group
                      (default: the starters table when gamelogs come from SQLite).
    live_starters   : {GAME_ID: {PLAYER_ID}} confirmed starters for extra_rows games; a game
                      missing here gets NaN lineup features.
    pregame_inactive: {(GAME_ID, TEAM_ID): {PLAYER_ID}} on the lineups JSON Inactive list
                      before the decision time; the inactive group adds them to pregame_out
                      (default: lineups.pregame_inactive() when gamelogs come from SQLite).
    game_market     : GAME_ID, mkt_total, mkt_home_margin for the game_market group (default:
                      game_markets.history_features() when gamelogs come from SQLite).
    """
    groups = ENABLED_GROUPS if groups is None else sorted(set(groups))
    feature_cols = model_features(groups)
    gl = clean_gamelogs(de.load_gamelogs() if gamelogs is None else gamelogs)
    trk = de.load_tracking_rebounds() if tracking is None else tracking

    tg = team_games(gl)
    tstate = team_state(tg)
    pstate = player_state(gl, tg)

    targets = gl[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE", "SEASON",
                  "is_home", "is_playoffs", "REB", "MIN"]].merge(
        tg[["TEAM_ID", "GAME_ID", "OPP_TEAM_ID"]], on=["TEAM_ID", "GAME_ID"], how="left")
    targets["_live"] = False
    if extra_rows is not None and len(extra_rows):
        ex = extra_rows.copy()
        ex["GAME_DATE"] = pd.to_datetime(ex["GAME_DATE"]).dt.normalize()
        ex["GAME_ID"] = ex["GAME_ID"].astype(str)
        ex["REB"] = np.nan
        ex["MIN"] = np.nan
        ex["_live"] = True
        targets = pd.concat([targets, ex[targets.columns]], ignore_index=True)

    # Player form
    df = _asof(targets, pstate, by="PLAYER_ID")
    df["rest_days"] = (df["GAME_DATE"] - df["last_game_date"]).dt.days.clip(upper=10).fillna(10)
    df["is_b2b"] = (df["rest_days"] <= 1).astype(int)
    # Season-scoped fields reset when the last game was in a prior season.
    last_season = _asof(targets[["PLAYER_ID", "GAME_DATE"]],
                        gl[["PLAYER_ID", "GAME_DATE", "SEASON"]].rename(columns={"SEASON": "_last_season"}),
                        by="PLAYER_ID")["_last_season"]
    new_season = last_season.ne(df["SEASON"])
    df.loc[new_season, "reb_season_avg"] = np.nan
    df.loc[new_season, "games_in_season"] = 0
    df["games_in_season"] = df["games_in_season"].fillna(0)
    df["games_played"] = df["games_played"].fillna(0)

    if "tracking" in groups:
        if tracking_daily is None and gamelogs is None:
            from nba_daily import TRACKING_TABLE, load_table

            tracking_daily = load_table(TRACKING_TABLE)
        if tracking_daily is not None and len(tracking_daily):
            df = _asof(df, tracking_state(tracking_daily), by="PLAYER_ID")

    if "lineup" in groups:
        if starters is None and gamelogs is None:
            from nba_daily import STARTERS_TABLE, load_table

            starters = load_table(STARTERS_TABLE)
        if starters is not None and len(starters):
            lf = lineup_features(df, gl, starters, live_starters)
            for c in lf.columns:
                df[c] = lf[c].values

    # Prior-season tracking + REB/G
    prev = trk.copy()
    prev["SEASON"] = prev["SEASON"].map(lambda s: f"{int(s[:4]) + 1}-{str(int(s[:4]) + 2)[-2:]}")
    prev = prev.rename(columns={
        "REB": "prev_reb_pg", "REB_CHANCES": "prev_reb_chances",
        "REB_CHANCE_PCT_ADJ": "prev_reb_chance_pct_adj", "REB_CONTEST_PCT": "prev_reb_contest_pct",
        "REB_CHANCE_DEFER": "prev_reb_chance_defer", "AVG_REB_DIST": "prev_avg_reb_dist",
    })
    prev_cols = ["prev_reb_pg", "prev_reb_chances", "prev_reb_chance_pct_adj",
                 "prev_reb_contest_pct", "prev_reb_chance_defer", "prev_avg_reb_dist"]
    prev["PLAYER_ID"] = prev["PLAYER_ID"].astype(int)
    df = df.merge(prev[["PLAYER_ID", "SEASON"] + prev_cols].drop_duplicates(["PLAYER_ID", "SEASON"]),
                  on=["PLAYER_ID", "SEASON"], how="left")

    # Matchup: own team state and opponent state, both strictly before tip
    own = _asof(df[["TEAM_ID", "GAME_DATE"]], tstate, by="TEAM_ID")
    opp = _asof(df[["OPP_TEAM_ID", "GAME_DATE"]], tstate, by="TEAM_ID", left_by="OPP_TEAM_ID")
    df["exp_opp_missed"] = (opp["t_missed"].values + own["t_missed_allowed"].values) / 2
    df["exp_own_missed"] = (own["t_missed"].values + opp["t_missed_allowed"].values) / 2
    df["exp_pace"] = (own["t_pace"].values + opp["t_pace"].values) / 2
    df["opp_oreb_pct"] = opp["t_oreb_pct"].values
    df["opp_dreb_pct"] = opp["t_dreb_pct"].values
    df["opp_3pa_rate"] = opp["t_3pa_rate"].values
    df["exp_margin_abs"] = np.abs(own["t_margin"].values - opp["t_margin"].values) / 2
    df["_exp_total"] = ((own["t_pts"].values + opp["t_pts_allowed"].values) / 2
                        + (opp["t_pts"].values + own["t_pts_allowed"].values) / 2)

    if "travel" in groups:
        import travel

        sched = de.load_schedule() if schedule is None and gamelogs is None else schedule
        if sched is not None and len(sched):
            tf = travel.travel_features(sched)
            df = df.merge(tf.drop(columns=["sched_rest_days"]), on=["TEAM_ID", "GAME_ID"], how="left")
            opp = tf[["TEAM_ID", "GAME_ID", "sched_rest_days"]].rename(
                columns={"TEAM_ID": "OPP_TEAM_ID", "sched_rest_days": "opp_rest_days"})
            df = df.merge(opp, on=["OPP_TEAM_ID", "GAME_ID"], how="left")

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

    # Teammate vacuum
    pool = gl[["TEAM_ID", "SEASON", "PLAYER_ID"]].drop_duplicates()
    played = set(zip(gl["PLAYER_ID"], gl["GAME_ID"]))
    if pregame_out is None and gamelogs is None and PREGAME_ABSENCES:
        from injury_reports import pregame_out_for_gamelogs

        pregame_out = pregame_out_for_gamelogs(gl)
    if "inactive" in groups:
        if pregame_inactive is None and gamelogs is None:
            from lineups import pregame_inactive as _load_inactive

            pregame_inactive = _load_inactive()
        merged = {k: set(v) for k, v in (pregame_out or {}).items()}
        for k, v in (pregame_inactive or {}).items():
            merged.setdefault(k, set()).update(v)
        pregame_out = merged
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

    # An enabled group whose source data is missing yields NaN columns, not a KeyError.
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    if apply_filters:
        hist = ~df["_live"]
        keep = (df["games_played"] >= MIN_GAMES) & (df["min_roll"] >= MIN_TRAILING_MINUTES)
        df = df[~hist | keep]

    # MIN is an outcome (actual minutes), kept only as a training target — never a feature.
    return df[ID_COLS + ["_live", TARGET, "MIN", "last_game_date"] + feature_cols].reset_index(drop=True)
