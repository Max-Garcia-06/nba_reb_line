import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Kalshi — RSA key auth
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.getenv("KALSHI_PRIVATE_KEY_PATH", "")
KALSHI_BASE_URL = os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")
KALSHI_ORDER_URL = os.getenv("KALSHI_ORDER_URL", "https://api.elections.kalshi.com/trade-api/v2")

# Edge detection
EDGE_THRESHOLD = float(os.getenv("EDGE_THRESHOLD", "0.05"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
MAX_BET_PCT = float(os.getenv("MAX_BET_PCT", "0.02"))

# Storage
DB_PATH = os.getenv("DB_PATH", str(DATA_DIR / "nba_reb.db"))

# NBA API seasons to pull (last 3)
SEASONS = ["2022-23", "2023-24", "2024-25"]

# Rolling window (games) for trailing features
ROLLING_WINDOW = 10

# Minimum games played to include a player
MIN_GAMES = 15

# Distribution to use: "poisson" or "nbinom"
DISTRIBUTION = "nbinom"
