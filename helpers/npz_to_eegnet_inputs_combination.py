from itertools import combinations
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent

# 입력/출력 경로만 필요하면 여기만 수정
NPZ_PATH = BASE_DIR / "data" / "260623_sub2_yjkim.npz" # 260602_sub1_hjlee.npz, 260623_sub2_yjkim.npz
OUT_DIR = BASE_DIR / "data" / "processed" / "260623_sub2_yjkim_raw" # 260602_sub1_hjlee_raw, 260623_sub2_yjkim_raw

# 원본 label이 1..5이면 1, 이미 0..4이면 0으로 설정
LABEL_OFFSET = 1

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(NPZ_PATH)
    X = data["X"]  # (time, channels, trials)
    y_original = data["y"].astype(np.int64)  # 1..5

    # EEGNet 입력 형태로 변환
    X_eeg = np.transpose(X, (2, 1, 0)).astype(np.float32)  # (trials, channels, time)
    y = y_original - LABEL_OFFSET  # 0..4

    np.save(OUT_DIR / "X_eeg_raw.npy", X_eeg)
    np.save(OUT_DIR / "y.npy", y)
    np.save(OUT_DIR / "y_trigger.npy", y_original)

    # 10개 binary task 생성: y_01은 label 0/1 vs 나머지 label을 의미
    for a, b in combinations(range(5), 2): #[0,1,2,3,4]
        print(f"label{a}, {b} -> 0")
        # 선택된 두 label은 0, 나머지 세 label은 1
        y_binary = (~np.isin(y, [a, b])).astype(np.int64)
        np.save(OUT_DIR / f"y_{a}{b}.npy", y_binary)

    print(f"Saved to {OUT_DIR}")
    print(f"X_eeg_raw.npy: {X_eeg.shape}")
    print(f"y.npy: {np.bincount(y, minlength=5).tolist()}")


if __name__ == "__main__":
    main()
