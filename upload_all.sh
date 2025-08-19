#!/usr/bin/env bash
set -euo pipefail

# Array of dataset names
DATASETS=("I60-R60")

for dataset in "${DATASETS[@]}"; do
    echo "=== Starting upload for $dataset ==="
    ./upload.sh "$dataset"
    echo "=== Finished upload for $dataset ==="
    echo
done