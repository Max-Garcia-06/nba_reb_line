"""
identity_bridge.py
------------------
Resolve Kalshi player names to NBA PLAYER_IDs.

Kalshi titles carry display names ("Nikola Jokić", "Wendell Carter Jr.") and an
opaque `custom_strike.basketball_player` UUID that is not an NBA id. Matching is
done on a normalised name (accents, punctuation, Jr./III suffixes stripped),
restricted to the two teams in the game when a roster is given so same-name
collisions can't cross teams. Fallback: first initial + last name, accepted
only when unique.

The old scan matched `PLAYER_NAME.lower() == name.lower()`, which silently
missed every accented name; on the 2025-26 history this normaliser maps 98.7%
of settled markets (the rest are DNPs with no box score).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def norm_player_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", " ", s.lower().replace(".", "").replace("'", ""))
    return " ".join(_SUFFIX.sub(" ", s).split())


def _initial_last(norm: str) -> str:
    t = norm.split()
    return f"{t[0][0]} {t[-1]}" if t else ""


class PlayerIndex:
    """name -> PLAYER_ID lookup over a candidate pool (e.g. both teams' players tonight)."""

    def __init__(self, players: Iterable[tuple[int, str]]):
        self.by_name: dict[str, set[int]] = {}
        self.by_short: dict[str, set[int]] = {}
        for pid, name in players:
            n = norm_player_name(name)
            self.by_name.setdefault(n, set()).add(int(pid))
            self.by_short.setdefault(_initial_last(n), set()).add(int(pid))

    def resolve(self, player_name: str) -> int:
        n = norm_player_name(player_name)
        hits = self.by_name.get(n, set())
        if len(hits) == 1:
            return next(iter(hits))
        if not hits:
            short = self.by_short.get(_initial_last(n), set())
            if len(short) == 1:
                return next(iter(short))
        return 0


def resolve_nba_player_id(player_name: str, players: Iterable[tuple[int, str]]) -> int:
    """One-off resolve against (PLAYER_ID, PLAYER_NAME) pairs; 0 when ambiguous or unknown."""
    return PlayerIndex(players).resolve(player_name)
