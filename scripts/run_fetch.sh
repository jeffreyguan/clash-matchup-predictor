#!/bin/bash
# Wrapper for the daily launchd run of fetch_games.py.
# Runs the fetch with the project venv, timestamps start/end, and appends
# a per-run log. Absolute paths only — launchd starts with a bare environment.

set -uo pipefail

PROJECT_DIR="/Users/jeffreyguan/projects/clash-matchup-predictor"
PYTHON="$PROJECT_DIR/venv/bin/python3"
SCRIPT="$PROJECT_DIR/data/fetch_games.py"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/fetch_games.log"

mkdir -p "$LOG_DIR"

start=$(date +%s)
echo "===== run start $(date '+%Y-%m-%d %H:%M:%S') =====" >> "$LOG_FILE"

# Run from the project dir so any relative paths resolve correctly.
# caffeinate -s holds a power assertion (effective on AC power) so the Mac
# stays awake for the whole job, then sleeps normally once python exits.
cd "$PROJECT_DIR" || exit 1
caffeinate -s "$PYTHON" "$SCRIPT" >> "$LOG_FILE" 2>&1
status=$?

end=$(date +%s)
elapsed=$(( end - start ))
echo "===== run end   $(date '+%Y-%m-%d %H:%M:%S')  exit=$status  elapsed=${elapsed}s =====" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

exit $status
