"""Create subwindow channel-covariance features from preprocessed EEG epochs."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.covariance import ledoit_wolf


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/processed/260602_sub1_hjlee_raw"

AUDITORY_CHANNELS = ["T7", "T8", "TP7", "TP8", "TP9", "TP10"]
VISUAL_CHANNELS = ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8", "P7", "P8"]
IMAGERY_CHANNELS = AUDITORY_CHANNELS + VISUAL_CHANNELS

START_S = 2.25
STOP_S = 4.25
N_SUBWINDOWS = 8
WINDOW_S = 0.5
EPS = 1e-4

LABEL_FILES = [
    "y.npy",
    "y_01.npy",
    "y_02.npy",
    "y_03.npy",
    "y_04.npy",
    "y_12.npy",
    "y_13.npy",
    "y_14.npy",
    "y_23.npy",
    "y_24.npy",
    "y_34.npy",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--start-s", type=float, default=START_S)
    parser.add_argument("--stop-s", type=float, default=STOP_S)
    parser.add_argument("--n-subwindows", type=int, default=N_SUBWINDOWS)
    parser.add_argument("--window-s", type=float, default=WINDOW_S)
    args = parser.parse_args()

    X = np.load(args.data_dir / "X_eeg_raw.npy", mmap_mode="r")
    with (args.data_dir / "preprocess_meta.json").open(encoding="utf-8") as file:
        meta = json.load(file)

    sfreq = float(meta["epoch"]["sfreq"])
    data_start_s = float(meta["epoch"]["crop_tmin"])
    data_stop_s = data_start_s + (X.shape[-1] - 1) / sfreq

    if args.n_subwindows < 1 or args.window_s <= 0:
        raise ValueError("n-subwindows and window-s must be positive.")

    channel_names = meta["eeg"]["channels"]
    missing = [channel for channel in IMAGERY_CHANNELS if channel not in channel_names]
    if missing:
        raise ValueError(f"Missing imagery channels: {missing}")
    channel_indices = [channel_names.index(channel) for channel in IMAGERY_CHANNELS]

    start_idx = int(round((args.start_s - data_start_s) * sfreq))
    stop_idx = int(round((args.stop_s - data_start_s) * sfreq))
    X = X[:, channel_indices, start_idx:stop_idx]

    window_samples = min(int(round(args.window_s * sfreq)), X.shape[-1])
    max_start = X.shape[-1] - window_samples
    subwindow_starts = np.round(
        np.linspace(0, max_start, args.n_subwindows)
    ).astype(int)
    subwindows = [(int(start), int(start + window_samples)) for start in subwindow_starts]

    n_trials, n_channels, _ = X.shape
    X_cov = np.empty(
        (n_trials, len(subwindows), n_channels, n_channels),
        dtype=np.float32,
    )
    eye = np.eye(n_channels, dtype=np.float32)

    for window_idx, (window_start, window_stop) in enumerate(subwindows):
        segment = np.asarray(X[:, :, window_start:window_stop], dtype=np.float64)
        for trial_idx in range(n_trials):
            trial = segment[trial_idx].T
            trial -= trial.mean(axis=0, keepdims=True)
            covariance, _ = ledoit_wolf(trial, assume_centered=True)
            X_cov[trial_idx, window_idx] = covariance.astype(np.float32) + EPS * eye

    for label_file in LABEL_FILES:
        labels = np.load(args.data_dir / label_file, mmap_mode="r")
        if len(labels) != n_trials:
            raise ValueError(f"{label_file} has {len(labels)} labels, expected {n_trials}.")

    output_path = args.data_dir / "X_eeg_cov.npy"
    np.save(output_path, X_cov)

    print(f"Input EEG shape : {tuple(X.shape)}")
    print(f"Channels        : {IMAGERY_CHANNELS}")
    print(f"Time range      : {args.start_s:g}-{args.stop_s:g} s")
    print(f"Subwindows      : {subwindows}")
    print(f"Output shape    : {X_cov.shape}")
    print(f"Saved           : {output_path.resolve()}")


if __name__ == "__main__":
    main()
