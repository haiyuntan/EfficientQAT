#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/../.." && pwd)"
RUN_ID="$1"
LOG_ROOT="$PROJECT/logs/r0_r5/$RUN_ID"
mkdir -p "$LOG_ROOT"
source "$PROJECT/env/activate.sh"
nohup python "$HERE/prepare_r5_data.py" >"$LOG_ROOT/prepare_data.log" 2>&1 </dev/null &
DATA_PID=$!
echo "$DATA_PID" >"$LOG_ROOT/prepare_data.pid"
wait "$DATA_PID"
nohup timeout --signal=TERM --kill-after=30s 7100s bash "$HERE/run_r0_r5.sh" smoke "${RUN_ID}_smoke" >"$LOG_ROOT/smoke_driver.log" 2>&1 </dev/null &
SMOKE_PID=$!
echo "$SMOKE_PID" >"$LOG_ROOT/smoke.pid"
echo "Smoke started PID=$SMOKE_PID"
wait "$SMOKE_PID"
test -s "$PROJECT/logs/r0_r5/${RUN_ID}_smoke/SMOKE_OK"
nohup bash "$HERE/run_r0_r5.sh" full "${RUN_ID}_full" >"$LOG_ROOT/full_driver.log" 2>&1 </dev/null &
FULL_PID=$!
echo "$FULL_PID" >"$LOG_ROOT/full.pid"
echo "Full started PID=$FULL_PID"
wait "$FULL_PID"
