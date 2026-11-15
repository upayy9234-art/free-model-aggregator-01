#!/bin/bash
# Local practice-trial harness for the free-model-aggregator entry.
# Usage: ./test_harness.sh <trial> [seed]
#   trial: 0 | 1 | 2  (practice/instance in the kit)
# Runs the env server (node server.mjs) + this entry (python3 main.py),
# waits for the entry to exit 0, SIGTERMs the env, scores with score.mjs.
set -u
KIT="$HOME/workspace/makefun/free-model-aggregator-01-kit"
ENTRY="$HOME/workspace/makefun/free-model-aggregator-01"
TRIAL="${1:-0}"
SEED="${2:-1}"
TRIAL_DIR="$KIT/practice/trial-$TRIAL"
STATE="$ENTRY/test-state-$TRIAL"
SCORES="$ENTRY/test-scores-$TRIAL"
PORT=$(( 9500 + TRIAL ))
rm -rf "$STATE" "$SCORES"
mkdir -p "$STATE" "$SCORES"

echo "=== env (trial $TRIAL, port $PORT) ==="
MAKEFUN_ENV_DATA_DIR="$TRIAL_DIR/input/env" \
MAKEFUN_STATE_DIR="$STATE" \
MAKEFUN_ENV_PORT="$PORT" \
node "$KIT/program/env/server.mjs" > "$ENTRY/test-env-$TRIAL.log" 2>&1 &
ENV_PID=$!
for i in $(seq 1 50); do
  [ -f "$STATE/ready" ] && break
  sleep 0.1
done
[ -f "$STATE/ready" ] || { echo "ENV FAILED TO START"; cat "$ENTRY/test-env-$TRIAL.log"; kill $ENV_PID 2>/dev/null; exit 1; }

echo "=== entry ==="
MAKEFUN_ENV_URL="http://127.0.0.1:$PORT" \
MAKEFUN_PROMPT_FILE="$TRIAL_DIR/input/prompt.txt" \
MAKEFUN_TRIAL_SEED="$SEED" \
timeout 170 python3 "$ENTRY/main.py" > "$ENTRY/test-entry-$TRIAL.log" 2>&1
EC=$?
echo "entry exit code: $EC"
cat "$ENTRY/test-entry-$TRIAL.log"

echo "=== stop env, score ==="
kill -TERM $ENV_PID 2>/dev/null
sleep 1
wait $ENV_PID 2>/dev/null
MAKEFUN_OUTPUT_DIR="$STATE/export" \
MAKEFUN_REFERENCE_DIR="$TRIAL_DIR/reference" \
MAKEFUN_SCORES_DIR="$SCORES" \
node "$KIT/program/scorer/score.mjs" 2>&1 | tee "$ENTRY/test-score-$TRIAL.txt"
echo "=== summary ==="
cat "$SCORES/summary.json" 2>/dev/null || echo "(no summary.json)"
