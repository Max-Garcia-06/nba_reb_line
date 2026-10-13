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
    # Each step runs even if an earlier one fails (set -e used to skip everything
    # after a flaky stats.nba.com ETL, including reconcile and the weekly retrain).
    etl_ok=1
    run_py etl --incremental || { etl_ok=0; log "etl failed — continuing; weekly retrain/refit skipped"; }
    # Official injury reports for yesterday's games (pre-tip absences for features).
    run_py injury-backfill --start "$YESTERDAY_ET" --end "$YESTERDAY_ET" || log "injury-backfill failed"
    for d in $(recent_et_dates); do
      run_py reconcile --date "$d" || true
    done
    run_py report --date "$YESTERDAY_ET" || log "report failed"
    # Weekly retrain + blend refit, due when the last successful one is >6 days old
    # (a stamp file, not "is it Sunday": a launchd catch-up run on Monday used to
    # skip the week). AUTO_RETRAIN=0 in .env turns the retrain off (e.g. once a fill
    # calibrator exists: every retrain invalidates it, and REQUIRE_FILL_CALIB_FOR_LIVE
    # then blocks live). A failed fit leaves the previous model in place and
    # calibrate_preflight blocks live once it falls MODEL_MAX_DATA_LAG_DAYS behind.
    stamp="${LOG_DIR}/.last_weekly_refit"
    now_s="$(date +%s)"
    last_s="$( [[ -f "$stamp" ]] && stat -f %m "$stamp" || echo 0 )"
    if (( now_s - last_s > 6 * 86400 )); then
      if [[ "$etl_ok" == "1" ]]; then
        auto_retrain="$(grep -E '^AUTO_RETRAIN=' .env 2>/dev/null | tail -1 | cut -d= -f2 | tr -d '"' || true)"
        if [[ "${auto_retrain:-1}" != "0" ]]; then
          log "Weekly — retraining model"
          run_py train || log "train failed — previous model stays in place"
        fi
        log "Weekly — running refit-blend"
        run_py refit-blend && touch "$stamp"
      else
        log "Weekly retrain/refit due but ETL failed — will retry next run"
      fi
    fi
    ;;
  refit-blend)
    # Re-fit market-blend weight (global + per-segment) from a trailing window of
    # full-slate scoring, so w tracks the model's actual recent performance vs.
    # the market instead of staying pinned wherever it was last set by hand.
    # Runs automatically as part of `nightly` once a week (see above); this case
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
