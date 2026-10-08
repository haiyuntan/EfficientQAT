#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="$1"
timeout --signal=TERM --kill-after=30s 7100s bash "$HERE/run_r0_r5.sh" smoke "${RUN_ID}_smoke"
bash "$HERE/run_r0_r5.sh" full "${RUN_ID}_full"
