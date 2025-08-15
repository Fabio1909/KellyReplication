#!/usr/bin/env bash
set -euo pipefail

# Usage: ./upload_images.sh I5-R5
DATASET="${1:?Usage: $0 DATASET_NAME}"

ENDPOINT="https://s3api-eur-is-1.runpod.io"
BUCKET="wa3d8umwnt"
SRC="_images/${DATASET}/traning"
DEST_PREFIX="KellyReplication/_images/${DATASET}/traning"

ulimit -n 65535 || true

JOBS="$(mktemp -t s5jobs.XXXXXX)"
LOG="s5cmd_upload_${DATASET}_$(date +%Y%m%d_%H%M%S).log"

find "$SRC" -type f ! -name '.DS_Store' -print0 |
while IFS= read -r -d '' f; do
    rel="${f#"$SRC"/}"
    printf "cp %q s3://%s/%s/%s\n" "$f" "$BUCKET" "$DEST_PREFIX" "$rel"
done > "$JOBS"

TOTAL_FILES=$(wc -l < "$JOBS")
echo "Uploading dataset: $DATASET"
echo "Source: $SRC"
echo "Destination: s3://$BUCKET/$DEST_PREFIX/"
echo "Total files: $TOTAL_FILES"

# Progress loop
(
    while true; do
        if [[ -f "$LOG" ]]; then
            UPLOADED=$(wc -l < "$LOG" || true)
            PERCENT=$(( 100 * UPLOADED / TOTAL_FILES ))
            echo -ne "Progress: $UPLOADED / $TOTAL_FILES files (${PERCENT}%)\r"
        fi
        sleep 1
    done
) & PROGRESS_PID=$!

# Run s5cmd silently (only errors shown), log each successful file for progress counter
s5cmd --endpoint-url="$ENDPOINT" run "$JOBS" \
  2>&1 | grep -v '^cp ' | tee -a "$LOG.tmp"

# Extract only successful cp lines into $LOG for counting
grep '^cp ' "$LOG.tmp" > "$LOG"

kill "$PROGRESS_PID" 2>/dev/null || true
echo -e "\nUpload complete."
echo "Log saved to $LOG"