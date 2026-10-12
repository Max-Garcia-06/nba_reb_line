from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import DATA_DIR

LEDGER_PATH = Path(DATA_DIR) / "execution_ledger.json"
TICK_SIZE = 0.01


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round_to_tick(x: float) -> float:
    x = round(round(x / TICK_SIZE) * TICK_SIZE, 2)
    return min(0.99, max(0.01, x))


class LedgerCorruptError(RuntimeError):
    """The ledger file exists but cannot be parsed; refusing to treat it as empty."""


@dataclass(frozen=True)
class LedgerKey:
    game_date: str
    ticker: str
    side: str


class ExecutionLedger:
    def __init__(self, path: Path = LEDGER_PATH):
        self.path = path
        self._data = {"version": 1, "entries": []}
        self._load()

    def _load(self) -> None:
        # A corrupt ledger must not read as "nothing submitted" — every key would
        # look new and be re-placed, and the next save would erase the history.
        if not self.path.exists():
            return
        try:
            self._data = json.loads(self.path.read_text())
        except Exception as e:
            raise LedgerCorruptError(f"Execution ledger {self.path} is unreadable ({e}); fix or move it aside") from e

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        os.replace(tmp, self.path)

    def has(self, key: LedgerKey) -> bool:
        k = {"game_date": key.game_date, "ticker": key.ticker, "side": key.side}
        return k in (e.get("key") for e in self._data.get("entries", []))

    def has_successful_submit(self, key: LedgerKey) -> bool:
        """
        True if a post-submit for this key succeeded or ended ambiguously (network
        error after send) — either way the order may be live, so never re-send it.
        """
        k = {"game_date": key.game_date, "ticker": key.ticker, "side": key.side}
        for e in reversed(self._data.get("entries", [])):
            if e.get("key") != k:
                continue
            if str(e.get("note", "")) != "post-submit":
                continue
            if e.get("success") is True or e.get("ambiguous") is True:
                return True
        return False

    def count_submits(self, key: LedgerKey) -> int:
        """Number of post-submit entries already logged for ``key`` (client_order_id attempt index)."""
        k = {"game_date": key.game_date, "ticker": key.ticker, "side": key.side}
        return sum(
            1 for e in self._data.get("entries", []) if e.get("key") == k and str(e.get("note", "")) == "post-submit"
        )

    def add_attempt(
        self,
        key: LedgerKey,
        *,
        price: float,
        contracts: int,
        dollars: float,
        note: str = "",
        order_id: str = "",
        success: Optional[bool] = None,
        ambiguous: bool = False,
    ) -> None:
        entry = {
            "ts": _utc_now_iso(),
            "key": {"game_date": key.game_date, "ticker": key.ticker, "side": key.side},
            "price": price,
            "contracts": contracts,
            "dollars": dollars,
            "order_id": order_id,
            "success": success,
            "note": note,
        }
        if ambiguous:
            entry["ambiguous"] = True
        self._data.setdefault("entries", []).append(entry)
        self._save()


def suggest_limit_price(
    *,
    side: str,
    bid: float,
    ask: float,
    model_fair: float,
    max_cross_spread: float = 0.06,
    maker: bool = False,
) -> float:
    side = (side or "").lower()
    if side not in {"yes", "no"}:
        raise ValueError(f"Invalid side: {side!r}")

    bid = float(bid)
    ask = float(ask)
    model_fair = float(model_fair)
    spread = max(0.0, ask - bid)
    mid = (ask + bid) / 2

    if spread <= max_cross_spread:
        px = min(ask, model_fair)
    else:
        px = min(max(bid + TICK_SIZE, mid - TICK_SIZE), model_fair)

    # Maker mode: never cross — rest at least one tick inside the ask (no taker fee).
    if maker:
        px = min(px, ask - TICK_SIZE)

    return _round_to_tick(px)

