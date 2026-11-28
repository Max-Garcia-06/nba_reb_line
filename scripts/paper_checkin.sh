#!/usr/bin/env bash
# Paper-trial health check: scan cadence (gaps = Mac asleep), marks landed, paper P&L.
# Writes logs/paper_checkin_<end>.txt and emails it (NO_EMAIL=1 to skip).
# Usage: scripts/paper_checkin.sh [START_DATE]   (default 2026-11-23, end = today ET)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${REPO_ROOT}/.venv/bin/python"
START="${1:-2026-11-23}"
END="$(TZ=America/New_York date +%F)"
OUT="${REPO_ROOT}/logs/paper_checkin_${END}.txt"

{
  echo "Paper trial check-in: ${START} -> ${END}"
  echo
  echo "== Scan runs per ET day (full 11:00-23:50 day = 78; fewer = Mac asleep or off) =="
  grep "\[scan\] START" logs/scan.log | sed -E 's/^\[([0-9-]+) .*/\1/' \
    | awk -v s="$START" '$1 >= s' | sort | uniq -c
  echo
  echo "== Scan texts sent per ET day (one per scan that priced markets) =="
  grep "iMessage sent" logs/scan.log | cut -c1-10 | awk -v s="$START" '$1 >= s' | sort | uniq -c
  echo
  echo "== Scans skipped behind a running scan / hung scans killed (both should stay near 0) =="
  for pat in "still running; skipping" "hung for .*killing it"; do
    n="$(grep -hE "$pat" logs/scan.log logs/cron.log 2>/dev/null | sed -E 's/^\[([0-9-]+) .*/\1/' \
      | awk -v s="$START" '$1 >= s' | wc -l | tr -d ' ')"
    echo "  ${n}  ${pat//.\*/…}"
  done
  echo
  echo "== Nightly steps since ${START} (FAILED lines; none = all OK) =="
  grep -E "^\[[0-9-]+ .*\] \[nightly\] (START|FAILED)" logs/cron.log | sed -E 's/^\[([0-9-]+) .*/\1/' \
    | awk -v s="$START" '$1 >= s' | sort | uniq -c | sed 's/$/ nightly step(s) started/'
  grep -E "\[nightly\] FAILED" logs/cron.log | awk -v s="[$START" '$1 >= s' || true
  echo
  echo "== Paper journal (orders, mark rows by label, orders with a pre-tip mark) =="
  "$PYTHON" - "$START" <<'EOF'
import collections, glob, json, os, sys
start = sys.argv[1]
orders = 0; marked = set(); labels = collections.Counter()
for f in sorted(glob.glob("data/paper/trades_*.jsonl")):
    d = os.path.basename(f)[len("trades_"):-len(".jsonl")]
    if d < start:
        continue
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if r.get("note") == "mark":
            labels[r.get("mark_label", "")] += 1
            marked.add(r.get("order_id"))
        elif r.get("order_id"):
            orders += 1
print(f"paper orders: {orders}")
print(f"orders with >=1 mark: {len(marked)}")
print(f"mark rows by label: {dict(labels) or '{}'}")
EOF
  echo
  echo "== report-range --paper =="
  COLUMNS=110 "$PYTHON" run_pipeline.py report-range --paper --start "$START" --end "$END" 2>&1 \
    | grep -v -E "WARNING|INFO"
  echo
  echo "Go-live rule: ~30+ paper fills with net P&L >= 0 and close CLV >= 0."
} >"$OUT" 2>&1

echo "wrote $OUT"

if [[ "${NO_EMAIL:-0}" != "1" ]]; then
  "$PYTHON" -c "
import logging, sys; logging.basicConfig(level=logging.INFO)
import notify
notify.send_email('nba_reb_line: paper trial check-in ${END}', open(sys.argv[1]).read())
" "$OUT"
fi

# One-shot when launched by a com.nba_reb_line.paper-checkin* launchd agent: unload and
# remove that agent after running.
LABEL="${XPC_SERVICE_NAME:-}"
if [[ "$LABEL" == com.nba_reb_line.paper-checkin* ]]; then
  rm -f "${HOME}/Library/LaunchAgents/${LABEL}.plist"
  launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
fi
