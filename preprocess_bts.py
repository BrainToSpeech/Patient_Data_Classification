"""
Preprocess the current integrated XDF and save raw EEGNet and EMG epochs.

Current data assumptions, verified by 00_sanity_check_xdf_markers.py:
  - XDF contains one EEG stream; its sampling rate is read from the stream.
  - EMG channels are embedded in the EEG stream.
  - Onset triggers are marker values "101", "102", "103", "104", "105".
  - Each onset trigger defines an epoch from -0.2 to 4.25 seconds.
  - No external CSV is used.

Outputs:
  The configured output directory contains:
    X_eeg_raw.npy
    X_emg_raw.npy
    y.npy
    y_trigger.npy
    preprocess_meta.json

Downsampling:
  - Set DOWNSAMPLE_SFREQ to a target Hz value, or use --downsample-sfreq.
  - Leave it as None to preserve the original sampling rate.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pyxdf

XDF_PATH = Path("260602_sub1_hjlee.xdf")
OUT_DIR = Path("data") / "processed" / "260602_sub1_hjlee_raw_toy"  # Example output directory

EEG_STREAM_TYPE = "EEG"
MARKER_STREAM_TYPE = "Markers"

ONSET_TRIGGERS = ["101", "102", "103", "104", "105"]
EVENT_ID = {trigger: int(trigger) for trigger in ONSET_TRIGGERS}
TRIGGER_TO_LABEL = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4}

EMG_CHANNELS = ["EMG1", "EMG2"]
EMG_PREFIX = "EMG"
EEGNET_REF_CHANNEL = "FCz"
EEGNET_HIGHPASS = 0.5
LINE_FREQ = 60.0

DOWNSAMPLE_SFREQ = None

EPOCH_TMIN = -0.2
EPOCH_TMAX = 4.25
BASELINE = (-0.2, 0.0)
CROP_TMIN = 0.0
CROP_TMAX = 4.25


def scalar(x: Any, default: Any = "") -> Any:
    if x is None:
        return default
    while isinstance(x, (list, tuple)) and x:
        x = x[0]
    if isinstance(x, (list, tuple)) and not x:
        return default
    return x


def channel_labels(stream: dict[str, Any]) -> list[str]:
    info = stream.get("info", {}) or {}
    desc = info.get("desc") or [{}]
    desc0 = desc[0] if isinstance(desc, list) and desc else {}
    desc0 = desc0 or {}
    channels_group = desc0.get("channels") or [{}]
    channels0 = (
        channels_group[0] if isinstance(channels_group, list) and channels_group else {}
    )
    channels0 = channels0 or {}
    channels = channels0.get("channel") or []

    labels = []
    for idx, ch in enumerate(channels):
        ch = ch or {}
        labels.append(str(scalar(ch.get("label"), default=f"ch{idx + 1}")))
    return labels


def find_stream(streams: list[dict[str, Any]], stream_type: str) -> dict[str, Any]:
    matches = [
        stream
        for stream in streams
        if str(scalar(stream.get("info", {}).get("type"))).lower()
        == stream_type.lower()
    ]
    if not matches:
        raise RuntimeError(f"No stream with type={stream_type!r} found.")
    return matches[0]


def make_raw_from_xdf(eeg_stream: dict[str, Any]) -> tuple[mne.io.RawArray, np.ndarray]:
    ch_names = channel_labels(eeg_stream)
    data = np.asarray(eeg_stream["time_series"], dtype=np.float64).T
    eeg_ts = np.asarray(eeg_stream["time_stamps"], dtype=np.float64)

    if not ch_names:
        ch_names = [f"ch{i + 1}" for i in range(data.shape[0])]
    if len(ch_names) != data.shape[0]:
        raise RuntimeError(
            f"Channel label count ({len(ch_names)}) does not match data channels "
            f"({data.shape[0]})."
        )

    sfreq = float(scalar(eeg_stream["info"].get("nominal_srate"), default=0.0))
    if sfreq <= 0:
        sfreq = float(1.0 / np.median(np.diff(eeg_ts)))

    ch_types = ["emg" if name.startswith(EMG_PREFIX) else "eeg" for name in ch_names]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose=False)
    return raw, eeg_ts


def make_events(
    marker_stream: dict[str, Any],
    eeg_ts: np.ndarray,
) -> np.ndarray:
    vals = np.asarray(marker_stream["time_series"], dtype=object)
    if vals.ndim == 1:
        vals = vals.astype(str)
    else:
        vals = np.asarray(["|".join(map(str, row.ravel())) for row in vals], dtype=str)
    ts = np.asarray(marker_stream["time_stamps"], dtype=np.float64)

    events = []
    for value, lsl_time in zip(vals, ts):
        value = str(value)
        if value not in EVENT_ID:
            continue
        insert_at = int(np.searchsorted(eeg_ts, lsl_time))
        candidates = [i for i in (insert_at - 1, insert_at) if 0 <= i < len(eeg_ts)]
        sample = min(
            candidates,
            key=lambda i: abs(float(eeg_ts[i]) - float(lsl_time)),
        )
        events.append([sample, 0, EVENT_ID[value]])

    if not events:
        raise RuntimeError("No onset triggers found.")
    return np.asarray(events, dtype=np.int64)


########### Downsample -> Epoching -> Cropping ############


def make_epochs(
    raw: mne.io.RawArray,
    events: np.ndarray,
    event_id: dict[str, int],
    downsample_sfreq: float | None = None,
) -> mne.Epochs:

    if downsample_sfreq is not None:
        raw, events = raw.copy().resample(
            sfreq=downsample_sfreq,
            events=events,
            verbose=False,
        )

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=EPOCH_TMIN,
        tmax=EPOCH_TMAX,
        baseline=BASELINE,
        detrend=None,
        preload=True,
        reject_by_annotation=False,
        event_repeated="merge",
        on_missing="raise",
        verbose=False,
    )
    epochs.crop(tmin=CROP_TMIN, tmax=CROP_TMAX)
    return epochs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xdf", type=Path, default=XDF_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--downsample-sfreq",
        type=float,
        default=DOWNSAMPLE_SFREQ,
        help="Optional target sampling rate in Hz. Omit to keep the original rate.",
    )
    args = parser.parse_args()

    if not args.xdf.exists():
        raise FileNotFoundError(args.xdf.resolve())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    streams, _ = pyxdf.load_xdf(str(args.xdf))
    eeg_stream = find_stream(streams, EEG_STREAM_TYPE)
    marker_stream = find_stream(streams, MARKER_STREAM_TYPE)

    raw, eeg_ts = make_raw_from_xdf(eeg_stream)
    original_sfreq = float(raw.info["sfreq"])
    output_sfreq = args.downsample_sfreq or original_sfreq
    if args.downsample_sfreq is not None:
        if args.downsample_sfreq <= 0:
            raise ValueError("--downsample-sfreq must be greater than 0.")
        if args.downsample_sfreq >= original_sfreq:
            raise ValueError(
                "--downsample-sfreq must be lower than the original sampling rate "
                f"({original_sfreq:g} Hz)."
            )
    if EEGNET_HIGHPASS >= output_sfreq / 2.0:
        raise ValueError("EEG high-pass cutoff must be below the output Nyquist frequency.")
    notch_freqs = np.arange(LINE_FREQ, output_sfreq / 2.0, LINE_FREQ)
    eeg_channels = [ch for ch in raw.ch_names if not ch.startswith(EMG_PREFIX)]
    missing_emg = [ch for ch in EMG_CHANNELS if ch not in raw.ch_names]
    if missing_emg:
        raise RuntimeError(f"Missing EMG channels: {missing_emg}")

    if EEGNET_REF_CHANNEL not in eeg_channels:
        raise RuntimeError(f"EEG reference channel not found: {EEGNET_REF_CHANNEL}")

    raw_eeg = raw.copy().pick_channels(eeg_channels, ordered=True)
    raw_emg = raw.copy().pick_channels(EMG_CHANNELS, ordered=True)

    raw_eeg.filter(
        l_freq=EEGNET_HIGHPASS,
        h_freq=None,
        picks="eeg",
        verbose=False,
    )

    raw_emg.filter(
        l_freq=EEGNET_HIGHPASS,
        h_freq=None,
        picks="emg",
        verbose=False,
    )

    if notch_freqs.size:
        raw_eeg.notch_filter(freqs=notch_freqs, picks="eeg", verbose=False)
        raw_emg.notch_filter(freqs=notch_freqs, picks="emg", verbose=False)

    raw_eeg.set_eeg_reference(ref_channels=[EEGNET_REF_CHANNEL], verbose=False)
    raw_eeg.drop_channels([EEGNET_REF_CHANNEL])


    events = make_events(marker_stream, eeg_ts)

    eeg_epochs = make_epochs(
        raw_eeg,
        events,
        EVENT_ID,
        downsample_sfreq=args.downsample_sfreq,
    )
    emg_epochs = make_epochs(
        raw_emg,
        events,
        EVENT_ID,
        downsample_sfreq=args.downsample_sfreq,
    )

    X_eeg_raw = eeg_epochs.get_data().astype("float32")
    np.save(args.out_dir / "X_eeg_raw.npy", X_eeg_raw)

    X_emg_raw = emg_epochs.get_data().astype("float32")
    np.save(args.out_dir / "X_emg_raw.npy", X_emg_raw)

    y_trigger = eeg_epochs.events[:, 2].astype(np.int64)
    y = np.asarray([TRIGGER_TO_LABEL[int(code)] for code in y_trigger], dtype=np.int64)

    if not np.array_equal(eeg_epochs.events[:, 0], emg_epochs.events[:, 0]):
        raise RuntimeError("EEG and EMG epoch event samples do not match.")

    np.save(args.out_dir / "y.npy", y)
    np.save(args.out_dir / "y_trigger.npy", y_trigger)
    for a, b in combinations(range(len(TRIGGER_TO_LABEL)), 2):
        y_binary = (~np.isin(y, [a, b])).astype(np.int64)
        np.save(args.out_dir / f"y_{a}{b}.npy", y_binary)

    meta = {
        "xdf_path": str(args.xdf.resolve()),
        "epoch": {
            "tmin": EPOCH_TMIN,
            "tmax": EPOCH_TMAX,
            "baseline": BASELINE,
            "crop_tmin": CROP_TMIN,
            "crop_tmax": CROP_TMAX,
            "sfreq": float(eeg_epochs.info["sfreq"]),
        },
        "eeg": {
            "channels": eeg_epochs.ch_names,
            "shape": list(X_eeg_raw.shape),
            "reference": EEGNET_REF_CHANNEL,
            "highpass": EEGNET_HIGHPASS,
            "notch_freqs": notch_freqs.tolist(),
        },
        "emg": {
            "channels": emg_epochs.ch_names,
            "shape": list(X_emg_raw.shape),
        },
        "trigger_to_label": TRIGGER_TO_LABEL,
    }
    with (args.out_dir / "preprocess_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved preprocessed data to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
