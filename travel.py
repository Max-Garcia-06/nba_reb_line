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
    # Both offsets on tonight's date: a DST switch between the two games is not travel.
    prv_off = np.array([_utc_offset_h(v, d) if isinstance(v, str) else np.nan
                        for v, d in zip(prev_venue, tg["GAME_DATE"])])

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
