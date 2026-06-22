# Binary EEGNet Training

This project preprocesses one integrated EEG/EMG XDF recording and trains ten
binary EEGNet models over multiple time windows.

## Pipeline

```text
260602_sub1_hjlee.xdf
    -> preprocess_bts.py
    -> EEG/EMG epochs and binary labels
    -> run_sliding_windows.sh
    -> ten binary EEGNet models per time window
    -> check_results.ipynb
```

The ten binary tasks are `01`, `02`, `03`, `04`, `12`, `13`, `14`, `23`,
`24`, and `34`. For task `01`, original classes `0` and `1` become binary
label `0`; the remaining classes become binary label `1`.

## Example JSON Outputs

The repository includes three example JSON outputs:

- `preprocess_meta.json`: preprocessing settings and saved EEG/EMG shapes,
  channels, sampling rate, reference, filters, epoch range, and label mapping
- `run_config.json`: training arguments, selected input and time range, input
  shape, sampling rate, and train/validation/test split sizes
- `results.json`: best validation checkpoint and test metrics for each of the
  ten binary tasks

The example `run_config.json` and `results.json` are from the
`20260611_184817` run using `X_eeg_raw.npy` over `0` to `2` seconds.

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

The script extracts epochs from `-0.2` to `4.25` seconds, applies baseline
correction, crops them to `0` to `4.25` seconds, and saves EEG and EMG
separately.

## 2. Train Sliding Windows

```bash
bash run_sliding_windows.sh
```

Set `--input-file` in `run_sliding_windows.sh` to `X_eeg_raw.npy` or
`X_emg_raw.npy` to train on EEG or EMG. The script trains seven overlapping
0.5-second windows, with all ten binary models trained sequentially for each
window. Set `--checkpoint-root` to choose where each run is saved.

To train one custom window directly:

```bash
python train_10_binary_eegnet.py \
  --data-dir data/processed/260602_sub1_hjlee_raw \
  --input-file X_emg_raw.npy \
  --start-s 0 \
  --end-s 0.5
```

The trainer uses one shared stratified random `60% / 20% / 20%`
train/validation/test split, training-set normalization, a
`WeightedRandomSampler` for balanced training batches, balanced accuracy, and
validation-loss early stopping.

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
