"""Scan notifications: email summaries (Gmail SMTP) and short per-scan iMessage texts."""
import logging
import smtplib
import subprocess
from email.mime.text import MIMEText

from config import (
    GMAIL_APP_PASSWORD,
    GMAIL_SENDER,
    IMESSAGE_RECIPIENT,
    SCAN_EMAIL_ENABLED,
    SCAN_EMAIL_RECIPIENT,
    SCAN_TEXT_ENABLED,
)

logger = logging.getLogger(__name__)

_MAX_TEXT_SIGNALS = 5
_OSASCRIPT_TIMEOUT_SEC = 20
# Text and recipient arrive as argv, never spliced into the script source.
_SEND_SCRIPT = (
    "on run argv\n"
    "  tell application \"Messages\"\n"
    "    set svc to 1st account whose service type = iMessage\n"
    "    send (item 1 of argv) to participant (item 2 of argv) of svc\n"
    "  end tell\n"
    "end run"
)


def format_scan_text(*, now_et: str, mode: str, games: list[str], n_priced: int, signals: list,
                     n_new: int | None, outcome: str = "") -> str:
    """A few lines: when, which games, how many markets priced, and the edges (YES = over)."""
    lines = [f"NBA scan {now_et} ET · {mode}", f"{', '.join(games)} · {n_priced} mkts priced"]
    if not signals:
        lines.append("No edges.")
    else:
        head = f"{len(signals)} edge{'s' if len(signals) != 1 else ''}"
        lines.append(f"{head} ({n_new} new):" if n_new is not None else f"{head}:")
        for s in signals[:_MAX_TEXT_SIGNALS]:
            ou = "o" if s.recommended_side == "yes" else "u"
            lines.append(f"{s.player_name} {ou}{s.kalshi_line:g} @{s.limit_price:.2f} "
                         f"e{s.edge:+.2f} x{s.recommended_contracts}")
        if len(signals) > _MAX_TEXT_SIGNALS:
            lines.append(f"+{len(signals) - _MAX_TEXT_SIGNALS} more")
    if outcome:
        lines.append(outcome)
    return "\n".join(lines)


def send_imessage(text: str) -> None:
    """Send ``text`` to IMESSAGE_RECIPIENT via Messages.app. No-op if off; never raises."""
    if not SCAN_TEXT_ENABLED or not IMESSAGE_RECIPIENT:
        return
    try:
        subprocess.run(["osascript", "-e", _SEND_SCRIPT, text, IMESSAGE_RECIPIENT],
                       check=True, capture_output=True, text=True, timeout=_OSASCRIPT_TIMEOUT_SEC)
        logger.info("iMessage sent")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to send iMessage: {(e.stderr or '').strip() or e}")
    except Exception as e:
        logger.error(f"Failed to send iMessage: {e}")


def send_email(subject: str, body: str) -> None:
    """Send a plaintext email via Gmail SMTP. No-ops if disabled or unconfigured."""
    if not SCAN_EMAIL_ENABLED:
        return
    if not GMAIL_APP_PASSWORD or not GMAIL_SENDER or not SCAN_EMAIL_RECIPIENT:
        logger.warning("GMAIL_APP_PASSWORD/GMAIL_SENDER/SCAN_EMAIL_RECIPIENT not set — skipping email")
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_SENDER
    msg["To"] = SCAN_EMAIL_RECIPIENT

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(GMAIL_SENDER, GMAIL_APP_PASSWORD)
            smtp.send_message(msg)
        logger.info(f"Email sent: {subject}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
