#!/usr/bin/env bash
set -euo pipefail

# Usage: ./upload_images.sh I5-R5 [concurrency]
# Example: ./upload_images.sh I5-R5 512

# ----- Arguments -----
DATASET="${1:?Usage: $0 DATASET_NAME [concurrency]}"
CONCURRENCY="${2:-512}"  # Default concurrency = 512 if not specified

# ----- Fixed settings -----
ENDPOINT="https://s3api-eur-is-1.runpod.io"
BUCKET="wa3d8umwnt"
SRC="_images/${DATASET}/traning"
DEST_PREFIX="KellyReplication/_images/${DATASET}/traning"

# ----- System tuning -----
ulimit -n 65535 || true

# ----- Job list & log -----
JOBS="$(mktemp -t s5jobs.XXXXXX)"
LOG="s5cmd_upload_${DATASET}_$(date +%Y%m%d_%H%M%S).log"

# ----- Build job file -----
find "$SRC" -type f ! -name '.DS_Store' -print0 |
while IFS= read -r -d '' f; do
  rel="${f#"$SRC"/}"  # relative path after $SRC
  printf "cp %q s3://%s/%s/%s\n" "$f" "$BUCKET" "$DEST_PREFIX" "$rel"
done > "$JOBS"

echo "Uploading dataset: $DATASET"
echo "Source: $SRC"
echo "Destination: s3://$BUCKET/$DEST_PREFIX/"
echo "Files: $(wc -l < "$JOBS")"
echo "Concurrency: $CONCURRENCY"

# ----- Run uploads -----
time s5cmd --endpoint-url="$ENDPOINT" --concurrency "$CONCURRENCY" run "$JOBS" | tee -a "$LOG"

echo "Log saved to $LOG"