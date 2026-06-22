# Binary EEGNet Classification

This folder contains a simple EEG classification pipeline:

```text
XDF recording
  -> NPZ file
  -> NPY training files
  -> ten binary EEGNet models
```

The current workflow uses three main scripts:

```text
xdf_to_np.py
npz_to_eegnet_inputs.py
train_10_binary_eegnet_randomsplit.py
```

Generated data files, raw XDF recordings, and checkpoint folders are not meant
to be tracked in Git.

## 1. Convert XDF To NPZ

Use `xdf_to_np.py` to convert the raw XDF recording into one NPZ file.

```bash
cd /home/bts_sh/jihoon/Demo_binary
python xdf_to_np.py
```

Input:

```text
260602_sub1_hjlee.xdf
```

Output:

```text
data/260602_sub1_hjlee.npz
```

The NPZ file stores already-epoched EEG data:

```text
X: (time, channels, trials)
y: (trials,)
```

In this dataset, `y` is stored as labels `1..5`.

## 2. Convert NPZ To NPY Training Files

Use `npz_to_eegnet_inputs.py` to prepare the files used by the training script.

```bash
cd /home/bts_sh/jihoon/Demo_binary
python npz_to_eegnet_inputs.py
```

This script does not perform epoching or cropping. The NPZ file is already
epoched. This step only changes the array layout and creates binary labels.

Input:

```text
data/260602_sub1_hjlee.npz
```

Outputs:

```text
data/processed/260602_sub1_hjlee_raw/
├── X_eeg_raw.npy
├── y.npy
├── y_trigger.npy
├── y_01.npy
├── y_02.npy
├── y_03.npy
├── y_04.npy
├── y_12.npy
├── y_13.npy
├── y_14.npy
├── y_23.npy
├── y_24.npy
└── y_34.npy
```

Array layout conversion:

```text
NPZ X:      (time, channels, trials)
training X: (trials, channels, time)
```

Label conversion:

```text
y_trigger.npy: original labels 1..5
y.npy:         original labels shifted to 0..4
y_ab.npy:      binary labels for task ab
```

The ten binary tasks are:

```text
01, 02, 03, 04, 12, 13, 14, 23, 24, 34
```

For task `ab`:

```text
original label a or b -> binary label 0
all other labels      -> binary label 1
```

For example, `y_01.npy` maps original labels `0` and `1` to binary label `0`,
and original labels `2`, `3`, and `4` to binary label `1`.

## 3. Train Binary EEGNet Models

Use `train_10_binary_eegnet_randomsplit.py` to train one EEGNet model for each
binary task.

```bash
cd /home/bts_sh/jihoon/Demo_binary
python train_10_binary_eegnet_randomsplit.py
```

The script reads:

```text
data/processed/260602_sub1_hjlee_raw/X_eeg_raw.npy
data/processed/260602_sub1_hjlee_raw/y.npy
data/processed/260602_sub1_hjlee_raw/y_01.npy ... y_34.npy
```

### Split Strategy

The train/validation/test split is a random trial-level split. It is not a
time-axis split.

The split is stratified using `y.npy`, the original 5-class labels.

Default split:

```text
train: 60%
val:   20%
test:  20%
```

The same `train_idx`, `val_idx`, and `test_idx` are reused for all ten binary
tasks. Each task changes only the label file, such as `y_01.npy`, `y_02.npy`,
and so on.

### Training Details

For each binary task, the script:

- loads the corresponding binary label file
- logs the original-label and binary-label distributions
- normalizes `X_eeg_raw.npy` once using train-set channel-wise mean and std
- trains an EEGNet binary classifier
- uses `WeightedRandomSampler` to reduce binary class imbalance during training
- evaluates validation loss and balanced accuracy after each epoch
- saves the best checkpoint based on validation loss

The reported accuracy is balanced accuracy:

```text
(class 0 accuracy + class 1 accuracy) / 2
```

## 4. Outputs And Checkpoints

Each run creates one timestamped output folder:

```text
checkpoints_eegnet_randomsplit/<YYYYMMDD_HHMMSS>/
```

Inside that folder:

```text
run_config.json
results.json
train.log
y_01/best_model.pt
y_02/best_model.pt
...
y_34/best_model.pt
```

`best_model.pt` is selected by the lowest validation loss, not by validation
accuracy.

`train.log` contains:

- hyperparameters
- input shape and device
- train/validation/test split sizes
- original label distribution
- binary label distribution for each task
- train loss/accuracy and validation loss/accuracy every 10 epochs
- final test accuracy for each binary model

`results.json` stores the final validation and test metrics for all ten binary
tasks.

## Recommended Command

```bash
cd /home/bts_sh/jihoon/Demo_binary
python train_10_binary_eegnet_randomsplit.py
```

Common optional arguments:

```bash
python train_10_binary_eegnet_randomsplit.py \
  --epochs 500 \
  --patience 50 \
  --batch-size 16
```

## Git Notes

The repository should track source code and documentation, not large generated
files.

Typical ignored files:

```text
*.xdf
*.npz
*.npy
data/
checkpoints*/
__pycache__/
```
