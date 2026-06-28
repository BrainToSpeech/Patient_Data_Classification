from collections import defaultdict
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "npz"
OUT_ROOT = BASE_DIR / "data" / "processed"

def get_date(npz_path):
    return npz_path.stem.split("_", 1)[0]


def get_subject(npz_path):
    stem = npz_path.stem
    if stem.endswith("_emg"):
        stem = stem[:-4]
    return stem.split("_", 1)[1]


def get_modality(npz_path):
    return "emg" if npz_path.stem.endswith("_emg") else "eeg"


def load_eegnet_arrays(npz_path, modality):
    data = np.load(npz_path)
    X = data["X"]  # (time, channels, trials)

    LABEL_OFFSET = 1

    y = data["y"].astype(np.int64) - LABEL_OFFSET
    X_eegnet = np.transpose(X, (2, 1, 0)).astype(np.float32)  # (trials, channels, time)
    if modality == "emg":
        X_eegnet = X_eegnet[:, :2, :]
    return X_eegnet, y


def concat_npz_files(npz_files, modality):
    X_list = []
    y_list = []

    for npz_path in npz_files:
        X, y = load_eegnet_arrays(npz_path, modality)
        X_list.append(X)
        y_list.append(y)
        print(f"  loaded {npz_path.name}: X={X.shape}, y={y.shape}")

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def main():
    npz_files = sorted(DATA_DIR.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No npz files found in {DATA_DIR}")

    grouped = defaultdict(lambda: {"eeg": [], "emg": []})
    for npz_path in npz_files:
        grouped[(get_subject(npz_path), get_date(npz_path))][get_modality(npz_path)].append(npz_path)

    for (subject, date), files_by_modality in sorted(grouped.items()):
        out_dir = OUT_ROOT / subject / date
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{subject}/{date}]")
        y_by_modality = {}

        if files_by_modality["eeg"]:
            X_eeg, y_eeg = concat_npz_files(files_by_modality["eeg"], "eeg")
            np.save(out_dir / "X_eeg.npy", X_eeg)
            y_by_modality["eeg"] = y_eeg
            print(f"  saved X_eeg.npy: {X_eeg.shape}")

        if files_by_modality["emg"]:
            X_emg, y_emg = concat_npz_files(files_by_modality["emg"], "emg")
            np.save(out_dir / "X_emg.npy", X_emg)
            y_by_modality["emg"] = y_emg
            print(f"  saved X_emg.npy: {X_emg.shape}")

        if "eeg" in y_by_modality and "emg" in y_by_modality:
            if not np.array_equal(y_by_modality["eeg"], y_by_modality["emg"]):
                print("  warning: EEG and EMG labels differ; saving EEG labels to y.npy")
            y = y_by_modality["eeg"]
        elif "eeg" in y_by_modality:
            y = y_by_modality["eeg"]
        else:
            y = y_by_modality["emg"]

        np.save(out_dir / "y.npy", y)
        print(f"  saved y.npy: {y.shape}")


if __name__ == "__main__":
    main()
