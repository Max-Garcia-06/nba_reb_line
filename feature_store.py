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
from config import MIN_GAMES, ROLLING_WINDOW

TEAM_WINDOW = 15
EWM_HALFLIFE = 6.0
ROTATION_MIN = 15.0          # trailing MPG to count as a rotation player
ROTATION_MAX_GAP_DAYS = 21   # last appearance within this many days
MIN_TRAILING_MINUTES = 15.0  # training population, matches Kalshi prop universe

ID_COLS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "OPP_TEAM_ID", "GAME_ID", "GAME_DATE", "SEASON"]
TARGET = "REB"

MODEL_FEATURES = [
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

def teammate_vacuum(
    targets: pd.DataFrame,
    pstate: pd.DataFrame,
    roster_pool: pd.DataFrame,
    played: set[tuple[int, str]],
    absent_override: Optional[dict[int, set[int]]] = None,
) -> pd.DataFrame:
    """
    For each (TEAM_ID, GAME_ID) in targets: rebounds and minutes of rotation
    teammates who are NOT playing.

    History: 'not playing' = no box-score row for that game. In live use the
    equivalent is the official inactive/injury list (known before tip), passed
    via absent_override {TEAM_ID: {PLAYER_ID, ...}} for tonight's games.
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
    rot["absent"] = [
        (pid in override.get(tid, set())) if gid in live_games else ((pid, gid) not in played)
        for pid, tid, gid in zip(rot["PLAYER_ID"], rot["TEAM_ID"], rot["GAME_ID"])
    ]

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
) -> pd.DataFrame:
    """
    Parameters
    ----------
    extra_rows      : tonight's target rows (no box score yet). Columns:
                      PLAYER_ID, PLAYER_NAME, TEAM_ID, OPP_TEAM_ID, GAME_ID,
                      GAME_DATE, SEASON, is_home, is_playoffs. REB is NaN.
    absent_override : {TEAM_ID: {PLAYER_ID}} confirmed-out players for extra_rows games.
    gamelogs, tracking : inject frames (tests); default loads from SQLite.
    apply_filters   : drop thin-history / low-minute rows (training population).
                      Extra rows are never filtered.
    """
    gl = clean_gamelogs(de.load_gamelogs() if gamelogs is None else gamelogs)
    trk = de.load_tracking_rebounds() if tracking is None else tracking

    tg = team_games(gl)
    tstate = team_state(tg)
    pstate = player_state(gl, tg)

    targets = gl[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE", "SEASON",
                  "is_home", "is_playoffs", "REB"]].merge(
        tg[["TEAM_ID", "GAME_ID", "OPP_TEAM_ID"]], on=["TEAM_ID", "GAME_ID"], how="left")
    targets["_live"] = False
    if extra_rows is not None and len(extra_rows):
        ex = extra_rows.copy()
        ex["GAME_DATE"] = pd.to_datetime(ex["GAME_DATE"]).dt.normalize()
        ex["GAME_ID"] = ex["GAME_ID"].astype(str)
        ex["REB"] = np.nan
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

    # Teammate vacuum
    pool = gl[["TEAM_ID", "SEASON", "PLAYER_ID"]].drop_duplicates()
    played = set(zip(gl["PLAYER_ID"], gl["GAME_ID"]))
    vac = teammate_vacuum(df, pstate, pool, played, absent_override)
    df = df.merge(vac, on=["TEAM_ID", "GAME_ID"], how="left")
    for c in ("vacuum_reb", "vacuum_min", "vacuum_n"):
        df[c] = df[c].fillna(0.0)

    if apply_filters:
        hist = ~df["_live"]
        keep = (df["games_played"] >= MIN_GAMES) & (df["min_roll"] >= MIN_TRAILING_MINUTES)
        df = df[~hist | keep]

    return df[ID_COLS + ["_live", TARGET, "last_game_date"] + MODEL_FEATURES].reset_index(drop=True)
