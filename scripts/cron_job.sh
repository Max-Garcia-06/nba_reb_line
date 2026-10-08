#!/usr/bin/env bash
# Run a single pipeline step for cron (logging, venv, ET dates).
# Usage: scripts/cron_job.sh <snapshot|etl|reconcile|report|nightly|refit-blend|scan|scan-platt|scan-live>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"

JOB="${1:-}"
if [[ -z "$JOB" ]]; then
  echo "Usage: $0 <snapshot|etl|reconcile|report|nightly|refit-blend|scan|scan-platt|scan-live>" >&2
  exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

LOG_FILE="${LOG_DIR}/cron.log"
TS="$(TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z')"

log() {
  echo "[$TS] [$JOB] $*"
}

run_py() {
  log "START: $*"
  set +e
  "$PYTHON" run_pipeline.py "$@" >>"$LOG_FILE" 2>&1
  local code=$?
  set -e
  if [[ $code -eq 0 ]]; then
    log "OK (exit 0)"
  else
    log "FAILED (exit $code)"
  fi
  return $code
}

# Calendar dates in US/Eastern (NBA slate context)
TODAY_ET="$("$PYTHON" -c "
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
tz = ZoneInfo('America/New_York')
print(datetime.now(tz).date().isoformat())
")"

YESTERDAY_ET="$("$PYTHON" -c "
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
tz = ZoneInfo('America/New_York')
print((datetime.now(tz).date() - timedelta(days=1)).isoformat())
")"

# Nightly runs under launchd, which fires a missed StartCalendarInterval late (on
# the next wake) rather than skipping it. A run that lands a day or two behind
# would reconcile the wrong slate if it only ever looked at YESTERDAY_ET, so walk
# back a short window instead. reconcile is idempotent per order_id.
RECONCILE_LOOKBACK_DAYS=3

recent_et_dates() {
  "$PYTHON" -c "
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
today = datetime.now(ZoneInfo('America/New_York')).date()
print(' '.join((today - timedelta(days=n)).isoformat() for n in range(1, ${RECONCILE_LOOKBACK_DAYS} + 1)))
"
}

SCAN_LOG="${LOG_DIR}/scan.log"

case "$JOB" in
  snapshot)
    # Pre-game tape for today's slate (feeds model-vs-market / refit-blend / backtest)
    run_py snapshot --date "$TODAY_ET"
    ;;
  etl)
    run_py etl --incremental
    ;;
  reconcile)
    # After games: journal for yesterday's slate (04:30 ET job runs after West Coast games end)
    run_py reconcile --date "$YESTERDAY_ET"
    ;;
  report)
    run_py report --date "$YESTERDAY_ET"
    ;;
  nightly)
    # ETL, reconcile the recent slates' fills, then report against yesterday's.
    run_py etl --incremental
    for d in $(recent_et_dates); do
      run_py reconcile --date "$d" || true
    done
    run_py report --date "$YESTERDAY_ET"
    # Piggyback the weekly blend refit on this job (Sundays only) instead of a
    # standalone 5am cron slot: that slot silently missed 2 straight Sundays
    # because the laptop was asleep at exactly 5am.
    if [[ "$(TZ=America/New_York date '+%u')" == "7" ]]; then
      log "Sunday — also running refit-blend"
      run_py refit-blend
    fi
    ;;
  refit-blend)
    # Re-fit market-blend weight (global + per-segment) from a trailing window of
    # full-slate scoring, so w tracks the model's actual recent performance vs.
    # the market instead of staying pinned wherever it was last set by hand.
    # Runs automatically as part of `nightly` on Sundays (see above); this case
    # remains for manual/ad-hoc re-fits.
    run_py refit-blend
    ;;
  scan)
    # Dry-run edge scan — logs to scan.log, no orders placed
    log "START scan (dry-run) for $TODAY_ET"
    set +e
    "$PYTHON" run_pipeline.py scan --date "$TODAY_ET" --dry-run >>"$SCAN_LOG" 2>&1
    code=$?
    set -e
    echo "[$TS] [scan] EXIT $code"
    ;;
  scan-platt)
    # Shadow dry-run with the Platt calibrator enabled, logged separately so its
    # signals can be compared against the live isotonic path before anyone flips
    # USE_PLATT_CALIBRATION. Places no orders.
    log "START scan (dry-run, Platt) for $TODAY_ET"
    set +e
    USE_PLATT_CALIBRATION=true "$PYTHON" run_pipeline.py scan --date "$TODAY_ET" --dry-run >>"${LOG_DIR}/scan_platt.log" 2>&1
    code=$?
    set -e
    echo "[$TS] [scan-platt] EXIT $code"
    ;;
  scan-live)
    # Live scan — places orders on Kalshi
    log "START scan (LIVE) for $TODAY_ET"
    set +e
    "$PYTHON" run_pipeline.py scan --date "$TODAY_ET" --live >>"$SCAN_LOG" 2>&1
    code=$?
    set -e
    echo "[$TS] [scan-live] EXIT $code"
    ;;
  *)
    log "Unknown job: $JOB"
    exit 1
    ;;
esac
