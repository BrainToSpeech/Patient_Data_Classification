from itertools import combinations
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
NPZ_PATH = BASE_DIR / "data" / "260602_sub1_hjlee.npz"
OUT_DIR = BASE_DIR / "data" / "processed" / "260602_sub1_hjlee_raw"

LABEL_OFFSET = 1

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(NPZ_PATH)
    X = data["X"]  # (time, channels, trials)
    y_original = data["y"].astype(np.int64)  # 1..5

    X_eeg = np.transpose(X, (2, 1, 0)).astype(np.float32)  # (trials, channels, time)
    y = y_original - LABEL_OFFSET  # 0..4

    np.save(OUT_DIR / "X_eeg_raw.npy", X_eeg)
    np.save(OUT_DIR / "y.npy", y)
    np.save(OUT_DIR / "y_trigger.npy", y_original)

    for a, b in combinations(range(5), 2): #[0,1,2,3,4]
        print(f"label{a}, {b} -> 0")
        y_binary = (~np.isin(y, [a, b])).astype(np.int64)
        np.save(OUT_DIR / f"y_{a}{b}.npy", y_binary)

    print(f"Saved to {OUT_DIR}")
    print(f"X_eeg_raw.npy: {X_eeg.shape}")
    print(f"y.npy: {np.bincount(y, minlength=5).tolist()}")


if __name__ == "__main__":
    main()
