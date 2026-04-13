"""
feature_store.py
----------------
Builds the "Glass-Eater Matrix" — a player-game level feature table
with strictly trailing windows to prevent data leakage.

Features produced
-----------------
Opportunity features (the "Pie"):
  opp_missed_fg_per_game   — how many misses the opponent generates
  opp_3pa_rate             — fraction of opponent shots that are 3PAs
                             (long rebounds bypass bigs, go to guards)
  pace_adjustment          — combined pace z-score vs. league avg
  team_reb_vacuum          — rebound share left vacant by absent teammates
  vegas_spread_abs         — expected blowout risk (from spread)

Performance features (the "Slice"):
  adj_reb_chance_pct_roll  — trailing-N rebound chance %
  box_out_rate_roll        — box-out attempts per minute (trailing)
  contested_reb_rate_roll  — contested rebs / total rebs (trailing)
  reb_per_min_roll         — trailing rebounds per 36 (pace-adjusted)
  rest_days                — days since last game
  is_b2b                   — back-to-back flag
"""

import pandas as pd
import numpy as np
from typing import Optional

from config import ROLLING_WINDOW, MIN_GAMES
import data_engine as de


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    return (a / b.replace(0, np.nan)).fillna(fill)


def _trailing(df: pd.DataFrame, col: str, window: int, groupby: str = "PLAYER_ID") -> pd.Series:
    """Strictly trailing rolling mean (excludes current game)."""
    return (
        df.groupby(groupby)[col]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
    )


# ---------------------------------------------------------------------------
# Step 1 — Enrich game logs with rest days
# ---------------------------------------------------------------------------

def add_rest_features(gl: pd.DataFrame) -> pd.DataFrame:
    gl = gl.copy()
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    gl = gl.sort_values(["PLAYER_ID", "GAME_DATE"])
    gl["rest_days"] = (
        gl.groupby("PLAYER_ID")["GAME_DATE"]
        .diff()
        .dt.days
        .fillna(3)  # assume 3 rest days for season opener
    )
    gl["is_b2b"] = (gl["rest_days"] <= 1).astype(int)
    return gl


# ---------------------------------------------------------------------------
# Step 2 — Trailing performance features from game logs
# ---------------------------------------------------------------------------

def add_trailing_performance(gl: pd.DataFrame) -> pd.DataFrame:
    gl = gl.copy()
    W = ROLLING_WINDOW

    gl["MIN"] = pd.to_numeric(gl["MIN"], errors="coerce").fillna(0)
    gl["REB"] = pd.to_numeric(gl["REB"], errors="coerce").fillna(0)
    gl["OREB"] = pd.to_numeric(gl["OREB"], errors="coerce").fillna(0)
    gl["DREB"] = pd.to_numeric(gl["DREB"], errors="coerce").fillna(0)

    # Home court: "vs." = home, "@" = away
    gl["is_home"] = gl["MATCHUP"].str.contains(r"vs\.", na=False).astype(int)

    # Playoffs flag — slower pace, longer minutes, tighter rotations
    if "SEASON_TYPE" in gl.columns:
        gl["is_playoffs"] = (gl["SEASON_TYPE"] == "Playoffs").astype(int)
    else:
        gl["is_playoffs"] = 0

    gl["reb_per_36"] = _safe_div(gl["REB"] * 36, gl["MIN"])

    gl["reb_per_min_roll"] = _trailing(gl, "reb_per_36", W)
    gl["min_roll"] = _trailing(gl, "MIN", W)

    # Raw trailing rebound avg — direct predictor of tonight's line
    gl["reb_roll"] = _trailing(gl, "REB", W)

    # OREB / DREB split — captures rebounding style
    gl["oreb_roll"] = _trailing(gl, "OREB", W)
    gl["dreb_roll"] = _trailing(gl, "DREB", W)

    # Season-to-date average rebounds (longer window for stability)
    gl["reb_season_avg"] = (
        gl.groupby(["PLAYER_ID", "SEASON"])["REB"]
        .transform(lambda x: x.shift(1).expanding(min_periods=5).mean())
    )

    # cumulative games played (for MIN_GAMES filter)
    gl["games_played"] = gl.groupby("PLAYER_ID").cumcount()

    return gl


# ---------------------------------------------------------------------------
# Step 3 — Merge tracking data (season-level, joined on PLAYER_ID + SEASON)
# ---------------------------------------------------------------------------

def merge_tracking(gl: pd.DataFrame, trk_reb: pd.DataFrame, trk_eff: pd.DataFrame) -> pd.DataFrame:
    """
    Tracking data is season-level (not game-level).
    We join it as a prior-season baseline and apply game-level rolling on top.
    trk_eff is team-level only (no PLAYER_ID) so it is not merged.
    """
    keep_reb = ["PLAYER_ID", "SEASON",
                "REB_CHANCE_PCT_ADJ",
                "OREB_CHANCES", "DREB_CHANCES",
                "OREB_CONTEST", "DREB_CONTEST",
                "REB_CHANCE_DEFER",
                ]

    reb_cols = [c for c in keep_reb if c in trk_reb.columns]
    trk_reb = trk_reb[reb_cols].copy()

    gl["PLAYER_ID"] = gl["PLAYER_ID"].astype(int)
    trk_reb["PLAYER_ID"] = trk_reb["PLAYER_ID"].astype(int)

    gl = gl.merge(trk_reb, on=["PLAYER_ID", "SEASON"], how="left")

    # Derived: contested rebound rate
    if "OREB_CONTEST" in gl.columns and "DREB_CONTEST" in gl.columns:
        gl["contested_reb_total"] = gl["OREB_CONTEST"].fillna(0) + gl["DREB_CONTEST"].fillna(0)
        gl["contested_reb_rate"] = _safe_div(gl["contested_reb_total"], gl["REB"].replace(0, np.nan))
    else:
        gl["contested_reb_rate"] = np.nan

    # box_out_rate not available from nba_api tracking endpoints
    gl["box_out_rate"] = np.nan

    return gl


# ---------------------------------------------------------------------------
# Step 4 — Opponent shooting opportunity features
# ---------------------------------------------------------------------------

def build_opp_features(opp_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a TEAM_ID-keyed frame with:
      opp_missed_fg_per_game, opp_3pa_rate, team_pace
    We join this to game logs by opponent team.
    """
    opp = opp_df.copy()
    team = team_df.copy()

    # Rename for clarity
    opp = opp.rename(columns={"TEAM_ID": "OPP_TEAM_ID"})

    # FGM made by opponent = FGA - FGM missed from team's perspective
    # API "Opponent" measure gives OPP_FGM, OPP_FGA, OPP_FG3M, OPP_FG3A
    opp_cols = {c: c for c in opp.columns}  # passthrough

    if "OPP_FGA" in opp.columns and "OPP_FGM" in opp.columns:
        opp["opp_missed_fg_per_game"] = opp["OPP_FGA"] - opp["OPP_FGM"]
    else:
        opp["opp_missed_fg_per_game"] = np.nan

    if "OPP_FGA" in opp.columns and "OPP_FG3A" in opp.columns:
        opp["opp_3pa_rate"] = _safe_div(
            opp["OPP_FG3A"].fillna(0), opp["OPP_FGA"].replace(0, np.nan)
        )
    else:
        opp["opp_3pa_rate"] = np.nan

    # Merge in team pace
    if "PACE" in team.columns:
        pace_map = team[["TEAM_ID", "PACE", "SEASON"]].copy()
        pace_map = pace_map.rename(columns={"TEAM_ID": "OPP_TEAM_ID", "PACE": "opp_pace"})
        opp = opp.merge(pace_map, on=["OPP_TEAM_ID", "SEASON"], how="left")
    else:
        opp["opp_pace"] = np.nan

    return opp[["OPP_TEAM_ID", "SEASON", "opp_missed_fg_per_game", "opp_3pa_rate", "opp_pace"]]


def add_opponent_features(gl: pd.DataFrame, opp_features: pd.DataFrame) -> pd.DataFrame:
    """
    Derive OPP_TEAM_ID from GAME_ID + TEAM_ID, then join opponent features.
    Each GAME_ID has exactly two distinct teams; the opponent is the other one.
    """
    gl = gl.copy()
    opp_features = opp_features.copy()

    # Build GAME_ID -> set of TEAM_IDs mapping, then assign opponent
    if "OPP_TEAM_ID" not in gl.columns:
        game_teams = (
            gl[["GAME_ID", "TEAM_ID"]]
            .drop_duplicates()
            .groupby("GAME_ID")["TEAM_ID"]
            .apply(list)
            .reset_index()
            .rename(columns={"TEAM_ID": "game_teams"})
        )
        gl = gl.merge(game_teams, on="GAME_ID", how="left")
        gl["OPP_TEAM_ID"] = gl.apply(
            lambda r: next(
                (t for t in r["game_teams"] if t != r["TEAM_ID"]),
                np.nan,
            ) if isinstance(r["game_teams"], list) else np.nan,
            axis=1,
        )
        gl = gl.drop(columns=["game_teams"])
        gl["OPP_TEAM_ID"] = pd.to_numeric(gl["OPP_TEAM_ID"], errors="coerce")

    opp_features["OPP_TEAM_ID"] = pd.to_numeric(opp_features["OPP_TEAM_ID"], errors="coerce")
    gl = gl.merge(opp_features, on=["OPP_TEAM_ID", "SEASON"], how="left")
    return gl


# ---------------------------------------------------------------------------
# Step 5 — Team rebound vacuum (absent teammate adjustment)
# ---------------------------------------------------------------------------

def compute_rebound_vacuum(
    gl: pd.DataFrame,
    active_players: Optional[list[int]] = None,
) -> pd.Series:
    """
    Estimate the rebound share vacated by inactive teammates.
    If active_players list is given, we compute the fraction of
    the team's historical rebounding that is absent tonight.

    Returns a Series indexed like gl with the vacuum value.
    """
    if active_players is None:
        return pd.Series(0.0, index=gl.index)

    # Per-player average REB share on team
    team_reb = (
        gl.groupby(["TEAM_ID", "PLAYER_ID"])["REB"]
        .mean()
        .reset_index()
        .rename(columns={"REB": "avg_reb"})
    )
    team_totals = team_reb.groupby("TEAM_ID")["avg_reb"].sum().rename("team_avg_reb")
    team_reb = team_reb.merge(team_totals, on="TEAM_ID")
    team_reb["reb_share"] = _safe_div(team_reb["avg_reb"], team_reb["team_avg_reb"])

    # Absent players = all roster players not in active_players
    absent = team_reb[~team_reb["PLAYER_ID"].isin(active_players)]
    absent_vacuum = absent.groupby("TEAM_ID")["reb_share"].sum().rename("team_reb_vacuum")

    gl = gl.merge(absent_vacuum.reset_index(), on="TEAM_ID", how="left")
    return gl["team_reb_vacuum"].fillna(0.0)


# ---------------------------------------------------------------------------
# Step 6 — Pace adjustment
# ---------------------------------------------------------------------------

def add_pace_adjustment(gl: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    """
    pace_adjustment = z-score of (home_pace + away_pace) vs. league avg.
    Positive = fast game → more rebound opportunities.
    """
    gl = gl.copy()

    if "PACE" not in team_df.columns or "opp_pace" not in gl.columns:
        gl["pace_adjustment"] = 0.0
        return gl

    league_avg_pace = team_df["PACE"].mean()
    league_std_pace = team_df["PACE"].std()

    # Team pace for the player's own team
    team_pace_map = (
        team_df[["TEAM_ID", "PACE", "SEASON"]]
        .rename(columns={"PACE": "team_pace"})
    )
    gl = gl.merge(team_pace_map, on=["TEAM_ID", "SEASON"], how="left")

    combined_pace = (gl["team_pace"].fillna(league_avg_pace) + gl["opp_pace"].fillna(league_avg_pace)) / 2
    gl["pace_adjustment"] = (combined_pace - league_avg_pace) / (league_std_pace + 1e-6)

    return gl


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_feature_table(
    active_players: Optional[list[int]] = None,
    spread_map: Optional[dict] = None,  # {game_id: abs_spread}
) -> pd.DataFrame:
    """
    Build the full Glass-Eater feature table from stored data.

    Parameters
    ----------
    active_players : list of PLAYER_IDs active tonight (for vacuum calc).
    spread_map     : {game_id: abs_spread} from Vegas lines.

    Returns
    -------
    DataFrame with one row per player-game, feature columns, and
    target column REB (actual rebounds — set to NaN for future games).
    """
    gl = de.load_gamelogs()
    trk_reb = de.load_tracking_rebounds()
    trk_eff = de.load_tracking_efficiency()
    opp_df = de.load_opponent_shooting()
    team_df = de.load_team_base_stats()

    # Build features step by step
    gl = add_rest_features(gl)
    gl = add_trailing_performance(gl)
    gl = merge_tracking(gl, trk_reb, trk_eff)

    opp_features = build_opp_features(opp_df, team_df)
    gl = add_opponent_features(gl, opp_features)
    gl = add_pace_adjustment(gl, team_df)

    # Rebound vacuum
    gl["team_reb_vacuum"] = compute_rebound_vacuum(gl, active_players)

    # Vegas spread as blowout risk proxy
    if spread_map and "GAME_ID" in gl.columns:
        gl["vegas_spread_abs"] = gl["GAME_ID"].map(spread_map).fillna(5.0)
    else:
        gl["vegas_spread_abs"] = 5.0  # neutral default

    # Trailing tracking features (use season-level as proxy, since tracking is season-level)
    for col in ["REB_CHANCE_PCT_ADJ", "contested_reb_rate", "box_out_rate"]:
        if col in gl.columns:
            trail_col = f"{col}_roll"
            gl[trail_col] = _trailing(gl, col, ROLLING_WINDOW)

    # Filter out players with too few games
    gl = gl[gl["games_played"] >= MIN_GAMES].copy()

    # Filter to Kalshi-relevant minutes (≥15 MPG trailing) to match prop market population
    gl = gl[gl["min_roll"] >= 15].copy()

    # Final feature set
    feature_cols = [
        # Identifiers (kept for joins, not model input)
        "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_DATE", "GAME_ID", "SEASON",
        # Target
        "REB",
        # Opportunity features
        "opp_missed_fg_per_game",
        "opp_3pa_rate",
        "pace_adjustment",
        "team_reb_vacuum",
        "vegas_spread_abs",
        # Performance features
        "reb_season_avg",
        "reb_roll",
        "oreb_roll",
        "dreb_roll",
        "reb_per_min_roll",
        "min_roll",
        "rest_days",
        "is_b2b",
        "is_home",
        "is_playoffs",
        "REB_CHANCE_PCT_ADJ_roll",
        "contested_reb_rate_roll",
    ]

    available = [c for c in feature_cols if c in gl.columns]
    return gl[available].reset_index(drop=True)


MODEL_FEATURES = [
    "opp_missed_fg_per_game",
    "opp_3pa_rate",
    "pace_adjustment",
    "team_reb_vacuum",
    "vegas_spread_abs",
    "reb_season_avg",
    "reb_roll",
    "oreb_roll",
    "dreb_roll",
    "reb_per_min_roll",
    "min_roll",
    "rest_days",
    "is_b2b",
    "is_home",
    "is_playoffs",
    "REB_CHANCE_PCT_ADJ_roll",
    "contested_reb_rate_roll",
]
