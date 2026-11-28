"""Text a warning when scans are running on battery.

On battery the MacBook sleeps when the lid closes (caffeinate can't stop that),
and scans stop until it wakes: the 2026-11 trial lost most game evenings this
way. Called from cron_job.sh before each scan. At most one text per hour while
unplugged; plugging back in resets it. Never raises.
"""
import logging
import re
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

STATE = Path(__file__).resolve().parent / "logs" / ".battery_warned"
REPEAT_SEC = 60 * 60


def read_power(pmset_out: str) -> tuple[bool, int | None]:
    """(on_battery, percent) from `pmset -g batt`. Unknown output counts as not on battery."""
    on_batt = "Now drawing from 'Battery Power'" in pmset_out
    m = re.search(r"(\d+)%", pmset_out)
    return on_batt, int(m.group(1)) if m else None


def _pmset() -> str:
    try:
        return subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""


def check(now: float | None = None, pmset_out: str | None = None, state: Path = STATE, send=None) -> bool:
    """Send the warning if due. Returns True when a text was sent."""
    now = time.time() if now is None else now
    on_batt, pct = read_power(_pmset() if pmset_out is None else pmset_out)
    if not on_batt:
        state.unlink(missing_ok=True)
        return False
    try:
        if now - float(state.read_text().strip()) < REPEAT_SEC:
            return False
    except (OSError, ValueError):
        pass
    if send is None:
        from notify import send_imessage as send
    level = f" ({pct}%)" if pct is not None else ""
    send(f"NBA scans: Mac on battery{level}. Plug in and keep the lid open, or scans stop when it sleeps.")
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(f"{now:.0f}\n")
    return True


if __name__ == "__main__":
    try:
        if check():
            print("battery warning sent")
    except Exception as e:  # never break the scan over a warning
        logger.error(f"battery check failed: {e}")
