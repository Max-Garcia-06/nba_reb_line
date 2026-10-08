"""
injury_feed.py
--------------
Best-effort list of players ruled OUT for tonight, used by slate.py for the
teammate-vacuum features.

Source: ESPN's public NBA injuries JSON. It is unofficial and can be slow or
unreachable, so every failure degrades to an empty set (vacuum = 0, logged)
rather than blocking the scan. data/absent_<date>.json is the manual override.

Training uses realised absences (no box-score row); live uses this list. The
gap is late scratches after the scan — one more reason for the short
SCAN_WITHIN_HOURS entry window.
"""

from __future__ import annotations

import logging

import requests

from identity_bridge import norm_player_name

log = logging.getLogger(__name__)

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
TIMEOUT = (5, 10)
OUT_STATUSES = {"out"}


def out_player_names() -> set[str]:
    try:
        r = requests.get(ESPN_URL, timeout=TIMEOUT)
        r.raise_for_status()
        teams = r.json().get("injuries", [])
    except Exception as e:
        log.warning(f"Injury feed unavailable ({e})")
        return set()
    names = set()
    for t in teams:
        for inj in t.get("injuries", []) or []:
            if str(inj.get("status", "")).strip().lower() in OUT_STATUSES:
                name = (inj.get("athlete") or {}).get("displayName")
                if name:
                    names.add(norm_player_name(name))
    return names
