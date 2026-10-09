#!/usr/bin/env bash
# Launch this supervisor using nohup in its own session.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/../.." && pwd)"
RUN_ID="${1:?Run ID required}"
LOG_ROOT="$PROJECT/logs/r0_r5/$RUN_ID"
source "$PROJECT/env/activate.sh"
cd "$HERE/.."
nohup env RESUME_FROM=R2_BlockAP bash "$HERE/run_r0_r5.sh" full "$RUN_ID" >"$LOG_ROOT/nohup_driver.log" 2>&1 </dev/null &
DRIVER_PID=$!
printf '%s\n' "$DRIVER_PID" >"$LOG_ROOT/recovery_driver.pid"
nohup python "$HERE/monitor_resources.py" "$DRIVER_PID" "$LOG_ROOT/resources_nohup.tsv" >"$LOG_ROOT/resources_monitor.log" 2>&1 </dev/null &
MONITOR_PID=$!
wait "$DRIVER_PID"
RUN_RC=$?
wait "$MONITOR_PID" || true
printf 'FORMAL_EXIT\t%s\t%s\n' "$RUN_RC" "$(date -u +%FT%TZ)" >"$LOG_ROOT/supervisor_exit.tsv"
# Publish diagnostics on failure too; retry transient connection failures.
for attempt in {1..12}; do
  if nohup python "$HERE/sync_experiment_reports.py" >"$LOG_ROOT/publish_attempt_$attempt.log" 2>&1 </dev/null; then
    printf 'PUBLISHED\t%s\n' "$(date -u +%FT%TZ)" >"$LOG_ROOT/REPORTS_PUSHED"
    unset REPORT_GITHUB_TOKEN
    exit "$RUN_RC"
  fi
  sleep 30
done
printf 'PUSH_FAILED\t%s\n' "$(date -u +%FT%TZ)" >"$LOG_ROOT/REPORTS_PUSH_FAILED"
unset REPORT_GITHUB_TOKEN
exit 1
