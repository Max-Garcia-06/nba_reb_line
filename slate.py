"""
slate.py
--------
Build tonight's model inputs for a set of Kalshi rebound markets.

Replaces the old scan logic, which (a) scored each player's LAST HISTORICAL
feature row — last game's opponent, rest and home/away, and trailing windows
that excluded that game — (b) forced is_playoffs=1, and (c) fabricated a
"model" probability from the market price for unmatched players, which could
only ever manufacture edge.

Here every market is resolved to (PLAYER_ID, TEAM_ID, OPP_TEAM_ID, GAME_ID)
through the schedule and identity_bridge, and its features are built by the
same point-in-time code path as training (feature_store extra_rows). Markets
that can't be resolved, or players with thin history, are skipped — never
priced off the market.

Absences for the teammate-vacuum feature come from injury_feed (OUT players)
plus an optional manual file data/absent_<date>.json:
    {"OUT": ["Joel Embiid", "Jimmy Butler"]}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import data_engine as de
from config import DATA_DIR, GAMES_FOR_LAMBDA_SANITY
from feature_store import build_feature_table
from identity_bridge import PlayerIndex, norm_player_name

log = logging.getLogger(__name__)


@dataclass
class SlateRow:
    player_id: int
    player_name: str
    team_id: int
    opp_team_id: int
    game_id: str
    is_home: int


def _event_ticker(ml) -> str:
    et = str(getattr(ml, "event_ticker", "") or "")
    if not et and getattr(ml, "ticker", ""):
        parts = str(ml.ticker).split("-")
        if len(parts) >= 2:
            et = f"{parts[0]}-{parts[1]}"
    return et


def recent_rosters(game_date: str, team_ids: set[int], gl: pd.DataFrame | None = None) -> pd.DataFrame:
    """Players whose most recent game before `game_date` was for one of `team_ids` (trades handled)."""
    gl = de.load_gamelogs() if gl is None else gl
    gl = gl[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_DATE"]].copy()
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    gl = gl[gl["GAME_DATE"] < pd.Timestamp(game_date)]
    last = gl.sort_values("GAME_DATE").groupby("PLAYER_ID").tail(1)
    last = last[last["GAME_DATE"] >= pd.Timestamp(game_date) - pd.Timedelta(days=200)]
    return last[last["TEAM_ID"].astype(int).isin(team_ids)]


def load_manual_absences(game_date: str) -> set[str]:
    p = Path(DATA_DIR) / f"absent_{game_date}.json"
    if not p.exists():
        return set()
    try:
        return {norm_player_name(n) for n in json.loads(p.read_text()).get("OUT", [])}
    except Exception as e:
        log.warning(f"Could not read {p} ({e})")
        return set()


def resolve_slate(market_lines: list, game_date: str) -> tuple[dict[str, SlateRow], pd.DataFrame, list[str]]:
    """
    Returns ({ticker: SlateRow}, rosters, unresolved_tickers).
    Every ticker of the same player/game maps to the same SlateRow.
    """
    idx = de.slate_schedule_index(game_date)
    teams = set()
    for row in idx.values():
        teams |= {row["home_team_id"], row["away_team_id"]}
    rosters = recent_rosters(game_date, teams)

    out: dict[str, SlateRow] = {}
    unresolved: list[str] = []
    cache: dict[tuple[str, str], SlateRow | None] = {}
    for ml in market_lines:
        et = _event_ticker(ml)
        mu = de.parse_kalshi_event_matchup(et)
        g = idx.get(de.matchup_slug(*mu)) if mu else None
        if not g:
            unresolved.append(ml.ticker)
            continue
        key = (norm_player_name(ml.player_name), g["game_id"])
        if key not in cache:
            pool = rosters[rosters["TEAM_ID"].isin([g["home_team_id"], g["away_team_id"])]]
            pid = PlayerIndex(zip(pool["PLAYER_ID"], pool["PLAYER_NAME"])).resolve(ml.player_name)
            if not pid:
                cache[key] = None
            else:
                team = int(pool.loc[pool["PLAYER_ID"] == pid, "TEAM_ID"].iloc[0])
                home = team == g["home_team_id"]
                cache[key] = SlateRow(
                    player_id=pid, player_name=ml.player_name, team_id=team,
                    opp_team_id=g["away_team_id"] if home else g["home_team_id"],
                    game_id=g["game_id"], is_home=int(home))
        if cache[key] is None:
            unresolved.append(ml.ticker)
        else:
            out[ml.ticker] = cache[key]
    return out, rosters, unresolved


def absences_by_team(game_date: str, rosters: pd.DataFrame) -> dict[int, set[int]]:
    from injury_feed import out_player_names

    names = out_player_names() | load_manual_absences(game_date)
    if not names:
        log.warning("No absence info (injury feed empty/unavailable, no manual file): vacuum features = 0")
    out: dict[int, set[int]] = {}
    for pid, name, team in zip(rosters["PLAYER_ID"], rosters["PLAYER_NAME"], rosters["TEAM_ID"]):
        if norm_player_name(name) in names:
            out.setdefault(int(team), set()).add(int(pid))
    return out


def build_slate_predictions(market_lines: list, game_date: str, model) -> tuple[list[dict], list, list[str]]:
    """
    Returns (predictions, kept_market_lines, skipped_reasons). predictions[i]
    corresponds to kept_market_lines[i] and carries a full PMF.
    """
    resolved, rosters, unresolved = resolve_slate(market_lines, game_date)
    skipped = [f"{t}: unresolved player/game" for t in unresolved]
    if not resolved:
        return [], [], skipped

    uniq = {(r.player_id, r.game_id): r for r in resolved.values()}
    season = de.season_for_date(game_date)
    extra = pd.DataFrame([{
        "PLAYER_ID": r.player_id, "PLAYER_NAME": r.player_name, "TEAM_ID": r.team_id,
        "OPP_TEAM_ID": r.opp_team_id, "GAME_ID": r.game_id, "GAME_DATE": game_date,
        "SEASON": season, "is_home": r.is_home, "is_playoffs": int(_is_playoffs(r.game_id)),
    } for r in uniq.values()])
    feats = build_feature_table(extra_rows=extra, absent_override=absences_by_team(game_date, rosters))
    feats = feats[feats["_live"]].set_index(["PLAYER_ID", "GAME_ID"])

    pmf = model.pmf(feats.reset_index())
    pmf_by_key = {k: pmf[i] for i, k in enumerate(feats.index)}

    preds, kept = [], []
    for ml in market_lines:
        r = resolved.get(ml.ticker)
        if r is None:
            continue
        key = (r.player_id, r.game_id)
        f = feats.loc[key] if key in feats.index else None
        gp = int(f["games_played"]) if f is not None else 0
        if f is None or gp < GAMES_FOR_LAMBDA_SANITY:
            skipped.append(f"{ml.ticker}: thin history (games_played={gp})")
            continue
        preds.append({
            "player_id": r.player_id, "player_name": ml.player_name, "game_date": game_date,
            "kalshi_line": float(ml.line), "pmf": pmf_by_key[key], "games_played": gp,
        })
        kept.append(ml)
    return preds, kept, skipped


def _is_playoffs(game_id: str) -> bool:
    # NBA game ids: 002 = regular season, 004 = playoffs, 005 = play-in. Training data
    # tags only "Playoffs" season-type games as playoffs, so play-in stays 0 here too.
    return str(game_id)[:3] == "004"
