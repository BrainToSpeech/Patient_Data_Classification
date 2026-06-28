#!/usr/bin/env bash
set -euo pipefail

PATIENT="sub1_hjlee" # sub1_hjlee    sub2_yjkim
MODEL="eegnet"
RUN_NAME="v3_sub1_eegnet_all_days"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$ROOT_DIR/data/processed/$PATIENT"
TRAIN_SCRIPT="$ROOT_DIR/trainers/train_v3_braindecode_eeg.py"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Error: train script not found: $TRAIN_SCRIPT" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Error: data directory not found: $DATA_ROOT" >&2
  exit 1
fi

for day_dir in "$DATA_ROOT"/*; do
  [[ -d "$day_dir" ]] || continue
  day="$(basename "$day_dir")"

  if [[ ! -f "$day_dir/X_eeg.npy" || ! -f "$day_dir/y.npy" ]]; then
    echo "Skipping $PATIENT/$day: missing X_eeg.npy or y.npy"
    continue
  fi

  echo "=================================================="
  echo "Running v3: patient=$PATIENT day=$day model=$MODEL run_name=$RUN_NAME"

  python3 "$TRAIN_SCRIPT" \
    --patient "$PATIENT" \
    --day "$day" \
    --model "$MODEL" \
    --run-name "$RUN_NAME"
done
