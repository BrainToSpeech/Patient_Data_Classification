# Demo Binary EEG Decoding

This repository contains a small EEG decoding pipeline for XDF recordings.
The current training workflow is based on `train_10_binary_eegnet_randomsplit_v2.py`.

Ignored files such as raw XDF recordings, generated NPZ/NPY arrays, `*.pt`
model weights, `backup/`, `scripts/`, and `*_nogit` checkpoint folders are not
described as versioned artifacts here.

## Files

### Preprocessing

- `xdf_to_np.py`
  - Loads each `*.xdf` file in the repository root.
  - Selects EEG and `trigger_stream`.
  - Keeps only markers in `select_epoch`.
  - Epochs each selected marker from `-0.2` to `2.0` sec, applies baseline, then crops to `0` to `2.0` sec.
  - Saves one NPZ file under `data/`.
  - Current use: EEG imagery-cue epoching. For `260623_sub2_yjkim.xdf`, markers `1~10` appear to be imagery cues, while `101~110` appear to be preceding visual/cognitive cues.

- `xdf_to_np_emg.py`
  - EMG-only XDF conversion.
  - Uses the last 8 channels, applies EMG-style filtering, and saves an NPZ file.

- `npz_to_eegnet_inputs.py`
  - Converts an already-epoched NPZ into NPY files used by the trainers.
  - Converts `X` from `(time, channels, trials)` to `(trials, channels, time)`.
  - Converts labels from `1..5` to `0..4`.
  - Creates `X_eeg_raw.npy`, `y.npy`, `y_trigger.npy`, and pairwise binary label files such as `y_01.npy`.

### Training

- `train_10_binary_eegnet_randomsplit_v2.py`
  - Current main trainer.
  - Reads `train_config_v2.json`.
  - Loads `X_eeg_raw.npy` and `y.npy`.
  - Runs three tasks:
    - `0_vs_rest`
    - `2_vs_rest`
    - `1_vs_3`
  - Can run either one train/validation/test split or 5-fold cross-validation.
  - Supports validation selection by `balanced_accuracy` or `balanced_loss`.
  - Saves logs and JSON/CSV summaries under `checkpoint_root`.

- `train_10_binary_eegnet_randomsplit.py`
  - Older v1 trainer.
  - Reads `train_config.json`.
  - Trains ten binary tasks using precomputed label files `y_01.npy` through `y_34.npy`.
  - Saves confusion matrices and `results.json`.

### Config

- `train_config_v2.json`
  - Current config for the v2 trainer.
  - Currently points to:

```json
"data_dir": "data/processed/260623_sub2_yjkim_raw",
"input_file": "X_eeg_raw.npy",
"checkpoint_root": "checkpoints_eegnet_randomsplit/sub2_yjkim_nogit"
```

- `train_config.json`
  - Config for the older v1 trainer.

## v1 vs v2

`v1` trains all ten pair-vs-rest binary tasks:

```text
01, 02, 03, 04, 12, 13, 14, 23, 24, 34
```

For each task, the selected pair is binary class `0`, and the remaining three
labels are binary class `1`.

`v2` is the current experiment script. It no longer trains all ten tasks by
default. Instead, it focuses on:

```text
0_vs_rest
2_vs_rest
1_vs_3
```

It also adds cross-validation support, sliding-window diagnostics, and explicit
model-selection control through:

```json
"selection_metric": "balanced_loss"
```

or:

```json
"selection_metric": "balanced_accuracy"
```

## Current v2 Workflow

### 1. Convert XDF to NPZ

Edit `select_epoch` in `xdf_to_np.py` if needed.

For imagery-cue epoching in the current `260623_sub2_yjkim.xdf` data, use the
imagery markers:

```python
select_epoch = ["1", "2", "3", "4", "5"]
```

Then run:

```bash
python xdf_to_np.py
```

Output:

```text
data/<subject>.npz
```

The NPZ contains:

```text
X: (time, channels, trials)
y: (trials,)
```

### 2. Convert NPZ to Trainer Inputs

Set the input/output paths at the top of `npz_to_eegnet_inputs.py`.

For the current yjkim data:

```python
NPZ_PATH = BASE_DIR / "data" / "260623_sub2_yjkim.npz"
OUT_DIR = BASE_DIR / "data" / "processed" / "260623_sub2_yjkim_raw"
LABEL_OFFSET = 1
```

Then run:

```bash
python npz_to_eegnet_inputs.py
```

Output folder:

```text
data/processed/260623_sub2_yjkim_raw/
```

Expected files:

```text
X_eeg_raw.npy
y.npy
y_trigger.npy
y_01.npy
y_02.npy
y_03.npy
y_04.npy
y_12.npy
y_13.npy
y_14.npy
y_23.npy
y_24.npy
y_34.npy
```

`v2` mainly uses `X_eeg_raw.npy` and `y.npy`.

### 3. Configure v2 Training

Edit `train_config_v2.json`.

Important fields:

```json
{
  "data_dir": "data/processed/260623_sub2_yjkim_raw",
  "input_file": "X_eeg_raw.npy",
  "checkpoint_root": "checkpoints_eegnet_randomsplit/sub2_yjkim_nogit",
  "epochs": 200,
  "batch_size": 16,
  "lr": 3e-5,
  "weight_decay": 0.0001,
  "patience": 50,
  "seed": 42,
  "val_test_ratio": 0.2,
  "weighted_sampler": true,
  "run_5fold_cross_validation": true,
  "selection_metric": "balanced_loss"
}
```

If `run_5fold_cross_validation` is `true`, the script runs 5-fold CV and writes
`cross_validation.csv` plus `cross_validation_results.json`.

If `run_5fold_cross_validation` is `false`, the script runs one stratified
train/validation/test split and writes `results.json`.

### 4. Run v2 Training

Use the `podcast` environment if the default Python environment does not have
NumPy, scikit-learn, or PyTorch.

```bash
python train_10_binary_eegnet_randomsplit_v2.py --config train_config_v2.json
```

The script:

1. Loads `X_eeg_raw.npy`.
2. Loads `y.npy`.
3. Stratifies by the original 5-class labels.
4. Normalizes EEG using only the training split mean and std.
5. Builds binary labels inside the script:
   - `0_vs_rest`
   - `2_vs_rest`
   - `1_vs_3`
6. Trains one EEGNet model per task.
7. Saves the best checkpoint according to `selection_metric`.
8. Writes logs and result summaries.

## v2 Output Structure

For a non-CV run:

```text
<checkpoint_root>/<YYYYMMDD_HHMMSS>/
├── run_config.json
├── results.json
├── train.log
├── y_0/best_model.pt
├── y_1_vs_3/best_model.pt
└── y_2/best_model.pt
```

For a 5-fold CV run:

```text
<checkpoint_root>/<YYYYMMDD_HHMMSS>/
├── run_config.json
├── train.log
├── cross_validation.csv
├── cross_validation_results.json
└── cross_validation/
    ├── fold_1/
    ├── fold_2/
    ├── fold_3/
    ├── fold_4/
    └── fold_5/
```

`*.pt` files are ignored by Git, so model weights are local-only unless the
ignore rule is changed.

## Git Tracking Policy

The current `.gitignore` excludes:

```text
*.xdf
*.npz
*.npy
backup/
scripts/
changes.md
__pycache__/
*.pt
checkpoints_eegnet_randomsplit/sub1_hjlee_nogit
checkpoints_eegnet_randomsplit/sub2_yjkim_nogit
```

This means:

- Raw recordings are local-only.
- Generated NPZ/NPY arrays are local-only.
- Model weight files are local-only.
- Checkpoint folders ending in `_nogit` are local-only.
- Tracked checkpoint summaries can still live under non-ignored folders such as:

```text
checkpoints_eegnet_randomsplit/sub1_hjlee/
checkpoints_eegnet_randomsplit/sub2_yjkim/
```

For Git-friendly experiment records, keep small files such as:

```text
train.log
run_config.json
results.json
cross_validation.csv
cross_validation_results.json
confusion_matrices.png
```

under a non-ignored checkpoint folder, and keep large `*.pt` files out of Git.

## Quick Commands

Current yjkim preprocessing and v2 training:

```bash
python xdf_to_np.py
python npz_to_eegnet_inputs.py
python train_10_binary_eegnet_randomsplit_v2.py --config train_config_v2.json
```

Older v1 training:

```bash
python train_10_binary_eegnet_randomsplit.py --config train_config.json
```
