#!/usr/bin/env bash
set -euo pipefail

# Array of dataset names
DATASETS=("I20-R5" "I60-R5" "I60-R20" "I60-R60")

for dataset in "${DATASETS[@]}"; do
    echo "=== Starting upload for $dataset ==="
    ./upload.sh "$dataset"
    echo "=== Finished upload for $dataset ==="
    echo
done