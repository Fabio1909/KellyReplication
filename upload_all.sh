#!/usr/bin/env bash
set -euo pipefail

# Array of dataset names
DATASETS=("I5-R5" "I5-R20" "I5-R60" "I20-R5" "I20-R20" "I20-R60" "I60-R5" "I60-R20" "I60-R60")

for dataset in "${DATASETS[@]}"; do
    echo "=== Starting upload for $dataset ==="
    ./upload.sh "$dataset"
    echo "=== Finished upload for $dataset ==="
    echo
done