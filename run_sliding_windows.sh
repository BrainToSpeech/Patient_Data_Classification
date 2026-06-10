#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# windows=(
#   "0 2"
#   "1 3"
#   "2 4"
#   "3 5"
#   "4 6"
#   "5 7"
#   "6 8"
#   "0 8"
# )

windows=(
  "2.25 2.75"
  "2.50 3.00"
  "2.75 3.25"
  "3.00 3.50"
  "3.25 3.75"
  "3.50 4.00"
  "3.75 4.25"
)

for window in "${windows[@]}"; do
  read -r start end <<< "$window"
  echo "Running window: ${start}s - ${end}s"
  # python train_10_binary_spdnet.py --start "$start" --end "$end"
  python train_10_binary_eegnet.py \
    --input-file X_emg_raw.npy \
    --checkpoint-root checkpoints_emg_eegnet \
    --start-s "$start" \
    --end-s "$end" \
    --lr 2e-5

done

echo "All sliding-window runs completed."


    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    # parser.add_argument("--input-file", default="X_eeg_raw.npy")
    # parser.add_argument("--checkpoint-root", type=Path, default=BASE_DIR / "checkpoints_eegnet")
    # parser.add_argument("--epochs", type=int, default=1000)
    # parser.add_argument("--batch-size", type=int, default=8)
    # parser.add_argument("--lr", type=float, default=1e-5)
    # parser.add_argument("--weight-decay", type=float, default=1e-4)
    # parser.add_argument("--patience", type=int, default=100)
    # parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--val_test_ratio", type=float, default=0.2)
    # parser.add_argument("--start-s", type=float, default=0.0)
    # parser.add_argument("--end-s", type=float, default=None)