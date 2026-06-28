"""
kalshi_bridge.py
----------------
Kalshi API client with RSA request signing (v2 auth) and a mock layer
for offline development.

Auth flow
---------
Kalshi v2 uses RSA-PSS signed requests, not a simple Bearer token.
Each request is signed with your private key; Kalshi verifies with the
public key you uploaded to the dashboard.

Setup
-----
1. Generate a key pair:
       openssl genrsa -out ~/.kalshi/private_key.pem 2048
       openssl rsa -in ~/.kalshi/private_key.pem -pubout -out ~/.kalshi/public_key.pem

2. Upload public_key.pem at demo.kalshi.co -> Settings -> API -> Create Key.
   Copy the Key ID shown after upload.

3. Add to .env:
       KALSHI_API_KEY_ID=<key-id-from-dashboard>
       KALSHI_PRIVATE_KEY_PATH=/Users/you/.kalshi/private_key.pem
       KALSHI_BASE_URL=https://demo-api.kalshi.co/trade-api/v2

Real API: https://demo-api.kalshi.co/trade-api/v2  (demo / paper)
          https://trading-api.kalshi.com/trade-api/v2 (live)
"""

import base64
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from config import KALSHI_API_KEY_ID, KALSHI_PRIVATE_KEY_PATH, KALSHI_BASE_URL, KALSHI_ORDER_URL

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MarketLine:
    ticker: str
    player_name: str
    player_id: int
    game_date: str
    line: float
    yes_ask: float
    yes_bid: float
    no_ask: float
    no_bid: float
    volume: int = 0
    open_interest: int = 0

    @property
    def yes_mid(self) -> float:
        return round((self.yes_ask + self.yes_bid) / 2, 4)

    @property
    def no_mid(self) -> float:
        return round((self.no_ask + self.no_bid) / 2, 4)

    @property
    def yes_spread(self) -> float:
        return round(self.yes_ask - self.yes_bid, 4)

    @property
    def no_spread(self) -> float:
        return round(self.no_ask - self.no_bid, 4)

    @property
    def implied_prob(self) -> float:
        # Backwards-compatible: historically this code treated the YES mid as "market prob".
        # Keep that behavior, but prefer yes_mid/no_mid explicitly in new logic.
        return self.yes_mid


@dataclass
class OrderResult:
    success: bool
    order_id: str
    ticker: str
    side: str
    contracts: int
    price: float
    message: str = ""


@dataclass
class OpenOrder:
    order_id: str
    ticker: str
    side: str
    action: str
    type: str
    status: str
    price: float
    remaining_count: int
    created_time: str = ""


# ---------------------------------------------------------------------------
# RSA signing
# ---------------------------------------------------------------------------

def _load_private_key(path: str):
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _sign(private_key, timestamp_ms: int, method: str, path: str) -> str:
    """
    Kalshi v2 signature:
      message = f"{timestamp_ms}{method.upper()}/trade-api/v2{path}"
    Signed with RSA-PSS SHA-256, base64-encoded.
    """
    message = f"{timestamp_ms}{method.upper()}/trade-api/v2{path}".encode("utf-8")
    sig = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


# ---------------------------------------------------------------------------
# Live client
# ---------------------------------------------------------------------------

class KalshiClient:
    """Kalshi REST API v2 with RSA-PSS signing."""

    def __init__(
        self,
        key_id: str = KALSHI_API_KEY_ID,
        private_key_path: str = KALSHI_PRIVATE_KEY_PATH,
        base_url: str = KALSHI_BASE_URL,
        order_url: str = KALSHI_ORDER_URL,
    ):
        self.key_id = key_id
        self.base_url = base_url.rstrip("/")
        self.order_url = order_url.rstrip("/")
        self._private_key = _load_private_key(private_key_path)
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def _auth_headers(self, method: str, path: str) -> dict:
        ts = int(time.time() * 1000)
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": str(ts),
            "KALSHI-ACCESS-SIGNATURE": _sign(self._private_key, ts, method, path),
        }

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        r = self._session.get(url, params=params, headers=self._auth_headers("GET", path), timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        url = f"{self.base_url}{path}"
        r = self._session.post(url, json=body, headers=self._auth_headers("POST", path), timeout=10)
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str, params: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        r = self._session.delete(url, params=params, headers=self._auth_headers("DELETE", path), timeout=10)
        r.raise_for_status()
        return r.json()

    def get_markets(
        self,
        series_ticker: str = "KXNBAREB",
        status: str = "open",
        limit: int = 200,
        max_pages: int = 20,
    ) -> list[dict]:
        """
        Fetch markets for a series, paginating via cursor.

        Kalshi returns at most 200 markets per page. For series with lots of
        markets (multi-day slates), today's games may appear on later pages.
        """
        markets: list[dict] = []
        cursor: Optional[str] = None
        pages = 0

        while True:
            params = {"series_ticker": series_ticker, "status": status, "limit": limit}
            if cursor:
                params["cursor"] = cursor
            data = self._get("/markets", params=params)
            batch = data.get("markets", []) or []
            markets.extend(batch)

            cursor = data.get("cursor")
            pages += 1
            if not cursor or not batch or pages >= max_pages:
                break

        return markets

    def get_rebound_lines(self, game_date: Optional[str] = None) -> list[MarketLine]:
        """
        Return rebound markets, optionally filtered to a specific YYYY-MM-DD.

        Note: Kalshi's markets endpoint doesn't support filtering by event date
        for this series, so we fetch then filter based on parsed event_ticker.
        """
        lines = self.parse_rebound_markets(self.get_markets())
        if not game_date:
            return lines
        return [ml for ml in lines if ml.game_date == game_date]

    def parse_rebound_markets(self, raw_markets: list[dict]) -> list[MarketLine]:
        """
        Parse KXNBAREB markets. Title format: "Nikola Jokic: 10+ rebounds"
        Only include markets that have live prices (yes_ask is not null).
        """
        lines = []
        for m in raw_markets:
            title = m.get("title", "")
            if "rebound" not in title.lower():
                continue
            ticker = m.get("ticker", "")
            try:
                # floor_strike is the authoritative line value when present
                line = float(m["floor_strike"]) if m.get("floor_strike") is not None else self._extract_line_from_title(title)
                player_name = self._extract_player_from_title(title)

                # API returns prices in two possible formats:
                #   yes_ask_dollars / yes_bid_dollars  (0.00–1.00 USD, live)
                #   yes_ask / yes_bid                  (0–100 cents, legacy)
                def _price(dollars_key, cents_key, fallback):
                    if m.get(dollars_key) is not None:
                        v = float(m[dollars_key])
                        return v if v > 0 else fallback
                    if m.get(cents_key) is not None:
                        return float(m[cents_key]) / 100
                    return fallback

                yes_ask = _price("yes_ask_dollars", "yes_ask", 0.50)
                yes_bid = _price("yes_bid_dollars", "yes_bid", 0.48)
                no_ask  = _price("no_ask_dollars",  "no_ask",  0.52)
                no_bid  = _price("no_bid_dollars",  "no_bid",  0.50)

                # Skip if both sides are still at resting quotes (no real market yet)
                if yes_ask >= 0.99 and yes_bid <= 0.01:
                    log.debug(f"No two-sided market yet for {ticker}")
                    continue
                # event_ticker format: KXNBAREB-26APR10CLEATL → extract date
                event_ticker = m.get("event_ticker", "")
                game_date_str = self._parse_game_date(event_ticker)
                lines.append(MarketLine(
                    ticker=ticker, player_name=player_name, player_id=0,
                    game_date=game_date_str, line=line,
                    yes_ask=yes_ask, yes_bid=yes_bid,
                    no_ask=no_ask, no_bid=no_bid,
                    volume=int(m.get("volume", 0) or 0),
                    open_interest=int(m.get("open_interest", 0) or 0),
                ))
            except (ValueError, KeyError, TypeError) as e:
                log.debug(f"Skipping {ticker}: {e}")
        return lines

    @staticmethod
    def _parse_game_date(event_ticker: str) -> str:
        """
        Extract date from event_ticker like KXNBAREB-26APR10CLEATL.
        Format: YY + MMM + DD  e.g. 26APR10 → 2026-04-10
        """
        match = re.search(r"(\d{2})([A-Z]{3})(\d{2})", event_ticker)
        if match:
            year, mon, day = match.groups()
            months = {"JAN":"01","FEB":"02","MAR":"03","APR":"04","MAY":"05","JUN":"06",
                      "JUL":"07","AUG":"08","SEP":"09","OCT":"10","NOV":"11","DEC":"12"}
            return f"20{year}-{months.get(mon,'00')}-{day}"
        return datetime.today().strftime("%Y-%m-%d")

    @staticmethod
    def _extract_line_from_title(title: str) -> float:
        # Format: "Player Name: 10+ rebounds"
        match = re.search(r":?\s*(\d+\.?\d*)\+?\s*rebound", title, re.IGNORECASE)
        if match:
            return float(match.group(1)) - 0.5  # "10+" means >9.5
        match = re.search(r"(\d+\.5|\d+)", title)
        if match:
            return float(match.group(1))
        raise ValueError(f"Cannot parse line from: {title}")

    @staticmethod
    def _extract_player_from_title(title: str) -> str:
        # Format: "Player Name: 10+ rebounds" → "Player Name"
        match = re.match(r"^([^:]+):", title)
        if match:
            return match.group(1).strip().title()
        parts = re.split(r"\b(over|under|more|fewer|\d+\+)\b", title, flags=re.IGNORECASE)
        return parts[0].strip().title() if parts else title.strip()

    def _post_order(self, path: str, body: dict) -> dict:
        """POST to the order URL (demo for paper trading, live for real money)."""
        url = f"{self.order_url}{path}"
        r = self._session.post(url, json=body, headers=self._auth_headers("POST", path), timeout=10)
        r.raise_for_status()
        return r.json()

    def place_order(self, ticker: str, side: str, contracts: int, price: float, order_type: str = "limit") -> OrderResult:
        side_norm = (side or "").strip().lower()
        body = {
            "ticker": ticker,
            "action": "buy",
            "side": side_norm,
            "count": contracts,
            "type": order_type,
        }
        # Kalshi v2 expects side-specific price keys.
        if side_norm == "yes":
            body["yes_price"] = int(round(price * 100))
        elif side_norm == "no":
            body["no_price"] = int(round(price * 100))
        else:
            return OrderResult(
                success=False,
                order_id="",
                ticker=ticker,
                side=side,
                contracts=contracts,
                price=price,
                message=f"Invalid side: {side!r}",
            )
        try:
            data = self._post_order("/portfolio/orders", body)
            order = data.get("order", {})
            return OrderResult(success=True, order_id=order.get("order_id", ""),
                               ticker=ticker, side=side, contracts=contracts,
                               price=price, message="Order placed")
        except requests.HTTPError as e:
            log.error(f"Order failed for {ticker}: {e}")
            return OrderResult(success=False, order_id="", ticker=ticker, side=side,
                               contracts=contracts, price=price, message=str(e))

    def get_orders(
        self,
        status: str = "resting",
        ticker: Optional[str] = None,
        limit: int = 200,
    ) -> list[OpenOrder]:
        """
        Fetch your portfolio orders.
        Common status for open orders is "resting".
        """
        params: dict = {"status": status, "limit": limit}
        if ticker:
            params["ticker"] = ticker
        data = self._get("/portfolio/orders", params=params)
        orders = []
        for o in data.get("orders", []) or []:
            side = (o.get("side") or "").lower()
            # Price may come back as dollars or cents depending on API version.
            price = None
            if o.get("yes_price_dollars") is not None and side == "yes":
                price = float(o["yes_price_dollars"])
            elif o.get("no_price_dollars") is not None and side == "no":
                price = float(o["no_price_dollars"])
            elif o.get("yes_price") is not None and side == "yes":
                price = float(o["yes_price"]) / 100
            elif o.get("no_price") is not None and side == "no":
                price = float(o["no_price"]) / 100
            else:
                # Fallback: some responses include a unified 'price' field.
                if o.get("price") is not None:
                    price = float(o["price"])
            if price is None:
                price = 0.0
            orders.append(
                OpenOrder(
                    order_id=str(o.get("order_id", "")),
                    ticker=str(o.get("ticker", "")),
                    side=side,
                    action=str(o.get("action", "")),
                    type=str(o.get("type", "")),
                    status=str(o.get("status", "")),
                    price=float(price),
                    remaining_count=int(o.get("remaining_count", o.get("count", 0)) or 0),
                    created_time=str(o.get("created_time", "")),
                )
            )
        return orders

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order by order_id."""
        try:
            self._delete(f"/portfolio/orders/{order_id}")
            return True
        except requests.HTTPError as e:
            log.error(f"Cancel failed for order_id={order_id}: {e}")
            return False

    def get_market(self, ticker: str) -> dict:
        """Fetch a single market by ticker (used for settlement/result)."""
        data = self._get(f"/markets/{ticker}")
        # Docs return either {"market": {...}} or directly the market payload depending on version.
        return data.get("market", data)

    def get_balance(self) -> float:
        data = self._get("/portfolio/balance")
        return float(data.get("balance", {}).get("available_balance", 0)) / 100


# ---------------------------------------------------------------------------
# Mock client
# ---------------------------------------------------------------------------

class MockKalshiClient:
    MOCK_PLAYERS = [
        {"name": "Nikola Jokic",         "line": 12.5, "yes_ask": 0.58, "yes_bid": 0.56},
        {"name": "Domantas Sabonis",      "line": 13.5, "yes_ask": 0.45, "yes_bid": 0.43},
        {"name": "Anthony Davis",         "line": 11.5, "yes_ask": 0.52, "yes_bid": 0.50},
        {"name": "Giannis Antetokounmpo", "line": 11.5, "yes_ask": 0.54, "yes_bid": 0.52},
        {"name": "Rudy Gobert",           "line": 12.5, "yes_ask": 0.49, "yes_bid": 0.47},
        {"name": "Bam Adebayo",           "line": 9.5,  "yes_ask": 0.51, "yes_bid": 0.49},
        {"name": "Joel Embiid",           "line": 11.5, "yes_ask": 0.53, "yes_bid": 0.51},
        {"name": "Karl-Anthony Towns",    "line": 9.5,  "yes_ask": 0.46, "yes_bid": 0.44},
    ]

    def get_rebound_lines(self, game_date: Optional[str] = None) -> list[MarketLine]:
        date_str = game_date or datetime.today().strftime("%Y-%m-%d")
        lines = []
        for i, p in enumerate(self.MOCK_PLAYERS):
            ticker = f"NBA-REB-{p['name'].upper().replace(' ', '-')}-{date_str}"
            lines.append(MarketLine(
                ticker=ticker, player_name=p["name"], player_id=i + 1,
                game_date=date_str, line=p["line"],
                yes_ask=p["yes_ask"], yes_bid=p["yes_bid"],
                no_ask=round(1 - p["yes_bid"], 4), no_bid=round(1 - p["yes_ask"], 4),
                volume=500 + i * 200, open_interest=1000 + i * 150,
            ))
        # Mirror live behavior: if a date is requested, only return that date.
        if not game_date:
            return lines
        return [ml for ml in lines if ml.game_date == game_date]

    def place_order(self, ticker: str, side: str, contracts: int, price: float, order_type: str = "limit") -> OrderResult:
        log.info(f"[MOCK] {ticker} | {side.upper()} x{contracts} @ {price:.2f}")
        return OrderResult(
            success=True,
            order_id=f"mock-{ticker[:20]}-{datetime.now().timestamp():.0f}",
            ticker=ticker, side=side, contracts=contracts, price=price,
            message="[Paper trade] Order simulated.",
        )

    def get_orders(self, status: str = "resting", ticker: Optional[str] = None, limit: int = 200) -> list[OpenOrder]:
        return []

    def cancel_order(self, order_id: str) -> bool:
        return True

    def get_balance(self) -> float:
        return 1000.0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_client(force_mock: bool = False) -> KalshiClient | MockKalshiClient:
    if force_mock or not KALSHI_API_KEY_ID or not KALSHI_PRIVATE_KEY_PATH:
        log.info("Kalshi credentials not configured — using MockKalshiClient.")
        return MockKalshiClient()
    return KalshiClient()
