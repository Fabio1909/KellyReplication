#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?Usage: $0 DATASET_NAME}"

ENDPOINT="https://s3api-eur-is-1.runpod.io"
BUCKET="wa3d8umwnt"
SRC="_images/${DATASET}/traning"
DEST_PREFIX="KellyReplication/_images/${DATASET}/traning"

ulimit -n 65535 || true

JOBS="$(mktemp -t s5jobs.XXXXXX)"
LOG="$(mktemp -t s5log.XXXXXX)"

# --- Preflight Check ---
echo "Checking connectivity to $ENDPOINT ..."
MAX_RETRIES=5
for attempt in $(seq 1 $MAX_RETRIES); do
    if s5cmd --endpoint-url="$ENDPOINT" ls "s3://$BUCKET/" >/dev/null 2>&1; then
        echo "✅ Connection successful (attempt $attempt)"
        break
    else
        echo "⚠️  Preflight check failed (attempt $attempt/$MAX_RETRIES), retrying in 5s..."
        sleep 5
    fi
    if [[ "$attempt" -eq "$MAX_RETRIES" ]]; then
        echo "❌ Could not connect to $ENDPOINT after $MAX_RETRIES attempts. Exiting."
        exit 1
    fi
done

# --- Build job list ---
find "$SRC" -type f ! -name '.DS_Store' -print0 |
while IFS= read -r -d '' f; do
    rel="${f#"$SRC"/}"
    printf "cp %q s3://%s/%s/%s\n" "$f" "$BUCKET" "$DEST_PREFIX" "$rel"
done > "$JOBS"

TOTAL_FILES=$(wc -l < "$JOBS" | tr -d ' ')
echo "Uploading dataset: $DATASET"
echo "Source: $SRC"
echo "Destination: s3://$BUCKET/$DEST_PREFIX/"
echo "Total files: $TOTAL_FILES"

# --- Progress Monitor ---
(
    while kill -0 "$$" 2>/dev/null; do
        UPLOADED=$(grep -c '^cp ' "$LOG" 2>/dev/null || true)
        PERCENT=$(( 100 * UPLOADED / TOTAL_FILES ))
        echo -ne "Progress: $UPLOADED / $TOTAL_FILES files (${PERCENT}%)\r"
        sleep 2
    done
) &
PROGRESS_PID=$!

# --- Upload with s5cmd ---
s5cmd --endpoint-url="$ENDPOINT" run "$JOBS" 2>&1 \
    | tee /dev/tty \
    | grep '^cp ' >> "$LOG" || true

# --- Cleanup ---
kill "$PROGRESS_PID" 2>/dev/null || true
echo -e "\n✅ Upload complete for $DATASET."