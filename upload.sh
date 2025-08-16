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

# Progress monitor in the background
(
    while kill -0 "$$" 2>/dev/null; do
        UPLOADED=$(grep -c '^cp ' "$LOG" 2>/dev/null || true)
        PERCENT=$(( 100 * UPLOADED / TOTAL_FILES ))
        echo -ne "Progress: $UPLOADED / $TOTAL_FILES files (${PERCENT}%)\r"
        sleep 1
    done
) &
PROGRESS_PID=$!

# Run s5cmd, filter output, and log successes for the progress counter
s5cmd --endpoint-url="$ENDPOINT" run "$JOBS" 2>&1 \
    | tee /dev/tty \
    | grep '^cp ' >> "$LOG" || true

# Cleanup
kill "$PROGRESS_PID" 2>/dev/null || true
echo -e "\nUpload complete for $DATASET."