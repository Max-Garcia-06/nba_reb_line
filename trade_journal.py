from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from config import DATA_DIR


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def journal_path(game_date: str, *, paper: bool = False) -> Path:
    """Live journal, or the dry-run paper journal under ``data/paper/``."""
    root = Path(DATA_DIR) / "paper" if paper else Path(DATA_DIR)
    return root / f"trades_{game_date}.jsonl"


def append_row(game_date: str, row: dict[str, Any], *, paper: bool = False) -> None:
    p = journal_path(game_date, paper=paper)
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
    games_played: int = 0
    predicted_lambda: float = 0.0
    p_model: float = 0.0
    p_model_raw: float = 0.0
    p_model_cal: float = 0.0  # calibrated pre-blend side prob (fit-blend input)
    model_trained_on: str = ""  # stamp of the model that priced this order
    p_market: float = 0.0
    fee_per_contract: float = 0.0  # expected Kalshi fee at the limit price
    edge: float = 0.0
    ev: float = 0.0
    expected_pnl: float = 0.0
    book_bid: float = 0.0
    book_ask: float = 0.0
    book_spread: float = 0.0
    filled_contracts: int = 0
    avg_fill_price: float = 0.0
    order_status: str = ""  # Kalshi order status at reconcile time (fill rows)
    mark_label: str = ""
    mark_yes_bid: float = 0.0
    mark_yes_ask: float = 0.0
    mark_no_bid: float = 0.0
    mark_no_ask: float = 0.0
    mark_yes_mid: float = 0.0
    mark_no_mid: float = 0.0
    note: str = ""
    success: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

