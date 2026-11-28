#!/usr/bin/env bash
# Run a single pipeline step for cron (logging, venv, ET dates).
# Usage: scripts/cron_job.sh <snapshot|etl|reconcile|report|nightly|refit-blend|mark|scan|scan-platt|scan-live>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/logs"
mkdir -p "$LOG_DIR"

JOB="${1:-}"
if [[ -z "$JOB" ]]; then
  echo "Usage: $0 <snapshot|etl|reconcile|report|nightly|refit-blend|mark|scan|scan-platt|scan-live>" >&2
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

# Scans run every 10 minutes; never let two overlap (a slow stats.nba.com call can stretch one).
# Lock file holds "<pid> <start epoch>". A holder older than SCAN_MAX_MIN is treated as hung
# (on 2026-11-27 one blocked 4.5h on a socket left dead by Mac sleep and skipped every later
# scan): its python child is killed and this scan runs. Only the scan's direct child is
# killed, so the mark sleepers it spawned keep their schedule.
SCAN_MAX_MIN="${SCAN_MAX_MIN:-25}"

# Running and not a zombie (kill -0 also succeeds for an unreaped zombie).
pid_alive() {
  local st
  st="$(ps -o stat= -p "$1" 2>/dev/null)" && [[ -n "$st" && "$st" != Z* ]]
}

scan_lock() {
  local lock="${LOG_DIR}/.scan.lock"
  local holder="" started="" age
  [[ -f "$lock" ]] && read -r holder started <"$lock" || true
  if [[ -n "$holder" ]] && pid_alive "$holder"; then
    # Old pid-only locks: fall back to the file's mtime (BSD stat, then GNU).
    started="${started:-$(stat -f %m "$lock" 2>/dev/null || stat -c %Y "$lock")}"
    age=$(( $(date +%s) - started ))
    if (( age < SCAN_MAX_MIN * 60 )); then
      echo "[$TS] [$JOB] previous scan (pid $holder) still running; skipping"
      exit 0
    fi
    echo "[$TS] [$JOB] previous scan (pid $holder) hung for $(( age / 60 ))m; killing it"
    pkill -TERM -P "$holder" 2>/dev/null || true
    local i
    for i in $(seq 1 20); do pid_alive "$holder" || break; sleep 0.5; done
    if pid_alive "$holder"; then
      pkill -KILL -P "$holder" 2>/dev/null || true
      kill -KILL "$holder" 2>/dev/null || true
    fi
  fi
  echo "$$ $(date +%s)" >"$lock"
  # Expand now: $lock is local and out of scope when the EXIT trap fires. Remove the
  # lock only while it is still ours, so a killed holder can't drop a newer scan's lock.
  trap "[[ \"\$(cut -d' ' -f1 '$lock' 2>/dev/null)\" == $$ ]] && rm -f '$lock'; true" EXIT
}

# Text a warning (max once an hour) while scans run on battery: unplugged, the Mac
# sleeps when the lid closes and scans stop. Never fails the scan.
battery_check() {
  "$PYTHON" power_warning.py 2>&1 || true
}

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
    # Feature-group sources for the last few days (each skips dates already stored).
    BACKFILL_START="$("$PYTHON" -c "
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
print((datetime.now(ZoneInfo('America/New_York')).date() - timedelta(days=${RECONCILE_LOOKBACK_DAYS})).isoformat())
")"
    run_py tracking-backfill --start "$BACKFILL_START" --end "$YESTERDAY_ET" || log "tracking-backfill failed"
    run_py starters-backfill --start "$BACKFILL_START" --end "$YESTERDAY_ET" || log "starters-backfill failed"
    run_py lineups-backfill --start "$BACKFILL_START" --end "$YESTERDAY_ET" || log "lineups-backfill failed"
    run_py game-markets-backfill || log "game-markets-backfill failed"
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
  mark)
    # CLV/paper-fill marks on wall-clock time. The sleepers a scan spawns stall while
    # the Mac sleeps (time.sleep pauses) and then fire after tip, when nothing is
    # markable; this catches the 15m and close marks on the next run after a wake.
    # mark is idempotent. Skip days with no journal to keep the log quiet.
    for kind in paper live; do
      if [[ "$kind" == paper ]]; then j="data/paper/trades_${TODAY_ET}.jsonl"; flag=(--paper); else j="data/trades_${TODAY_ET}.jsonl"; flag=(); fi
      [[ -f "$j" ]] || continue
      set +e
      "$PYTHON" run_pipeline.py mark --date "$TODAY_ET" "${flag[@]+"${flag[@]}"}" >>"${LOG_DIR}/mark.log" 2>&1
      code=$?
      set -e
      echo "[$TS] [mark] $kind EXIT $code"
    done
    ;;
  scan)
    battery_check
    scan_lock
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
    battery_check
    scan_lock
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
