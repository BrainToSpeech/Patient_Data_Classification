# Binary EEGNet Training

This project preprocesses one integrated EEG/EMG XDF recording and trains ten
binary EEGNet models over multiple time windows.

## Pipeline

```text
260602_sub1_hjlee.xdf
    -> preprocess_bts.py
    -> EEG/EMG preprocessed data and binary labels
    -> run_sliding_windows.sh
    -> ten binary EEGNet models per time window
    -> check_results.ipynb
```

The ten binary tasks are `01`, `02`, `03`, `04`, `12`, `13`, `14`, `23`,
`24`, and `34`. For task `01`, original classes `0` and `1` become binary
label `0`; the remaining classes become binary label `1`.

## Setup

Use Python 3.11 and install the dependencies:

```bash
pip install -r requirements.txt
```

Place the recording in the repository root:

```text
260602_sub1_hjlee.xdf
```

The XDF must contain:

- an `EEG` stream containing EEG channels, `EMG1`, and `EMG2`
- a `Markers` stream containing onset triggers `1` to `5`
- the `FCz` EEG reference channel

## 1. Preprocess

```bash
python preprocess_bts.py \
  --xdf 260602_sub1_hjlee.xdf \
  --out-dir data/processed/260602_sub1_hjlee_raw
```

This creates:

```text
data/processed/260602_sub1_hjlee_raw/
├── X_eeg_raw.npy
├── X_emg_raw.npy
├── y.npy
├── y_trigger.npy
├── y_01.npy ... y_34.npy
└── preprocess_meta.json
```

The script separates EEG and EMG channels from the integrated XDF recording.
It applies a `0.5 Hz` high-pass filter and `60 Hz` harmonic notch filters to
both signals. EEG is re-referenced to `FCz`, after which the `FCz` channel is
removed.

Epochs are extracted from `-0.2` to `4.25` seconds around triggers `101` to
`105`. Baseline correction is applied using `-0.2` to `0.0` seconds, and the
epochs are cropped to `0.0` to `4.25` seconds. The script then saves EEG and
EMG epochs separately and generates the five-class and ten binary label files.

## 2. Train Sliding Windows

```bash
bash run_sliding_windows.sh
```

The current script trains on `X_eeg_raw.npy` over seven overlapping
0.5-second windows. Each window trains all ten binary models sequentially and
saves them under:

```text
checkpoints_eeg_eegnet/<run_timestamp>/
```

To train EMG instead, change the following arguments in

```bash
bash run_sliding_windows.sh \
  --input-file X_emg_raw.npy \
  --checkpoint-root checkpoints_emg_eegnet
```

The trainer uses a chronological `60% / 20% / 20%` train/validation/test split,
training-set normalization, class-weighted cross-entropy, balanced accuracy,
and validation-loss early stopping.

## 3. Inspect Results

Open `check_results.ipynb` to inspect binary results and combine the ten binary
models into a five-class ensemble.

## Files

| File | Purpose |
|---|---|
| `preprocess_bts.py` | Convert the XDF into EEG/EMG epochs and labels |
| `train_10_binary_eegnet.py` | Train and test ten binary EEGNet models |
| `run_sliding_windows.sh` | Run training sequentially over multiple windows |
| `check_results.ipynb` | Inspect checkpoints and five-class ensemble results |
| `requirements.txt` | Python dependencies |

Raw data, processed arrays, and checkpoints are intentionally excluded from
Git.
