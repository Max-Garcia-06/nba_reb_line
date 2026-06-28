"""
trade_journal.py
----------------
Append-only trade journal for tracking live performance.

Writes JSONL rows under data/trades_YYYY-MM-DD.jsonl.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import DATA_DIR


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def journal_path(game_date: str) -> Path:
    return Path(DATA_DIR) / f"trades_{game_date}.jsonl"


def append_row(game_date: str, row: dict[str, Any]) -> None:
    p = journal_path(game_date)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    row.setdefault("ts", _utc_now_iso())
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


@dataclass
class TradeRow:
    game_date: str
    ticker: str
    side: str
    action: str
    contracts: int
    limit_price: float
    order_id: str
    player_name: str = ""
    kalshi_line: float = 0.0
    predicted_lambda: float = 0.0
    p_model: float = 0.0
    edge: float = 0.0
    ev: float = 0.0
    note: str = ""
    success: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

