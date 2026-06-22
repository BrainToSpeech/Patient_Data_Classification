# Binary EEG Classification Pipeline

This project converts one XDF recording into NumPy arrays, builds binary labels,
and trains ten binary classifiers using one shared random stratified split.

## Pipeline

```text
260602_sub1_hjlee.xdf
    -> xdf_to_np.py
    -> data/260602_sub1_hjlee.npz
    -> npz_to_eegnet_inputs.py
    -> data/processed/260602_sub1_hjlee_raw/*.npy
    -> train_10_binary_randomsplit.py
    -> checkpoints_<model>_randomsplit/<run_id>/
```

The ten binary tasks are:

```text
01, 02, 03, 04, 12, 13, 14, 23, 24, 34
```

For example, in task `01`, original labels `0` and `1` become binary label
`0`, while original labels `2`, `3`, and `4` become binary label `1`.

## 1. Convert XDF To NPZ

Use `xdf_to_np.py` to read the XDF file, extract EEG epochs, and save one NPZ
file.

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

The NPZ file contains:

```text
X: (time, channels, trials)
y: (trials,)
```

In the current data flow, `y` is stored as original labels `1..5`.

## 2. Convert NPZ To Training NPY Files

Use `npz_to_eegnet_inputs.py` to convert the already-epoched NPZ data into the
NPY files expected by the training script.

```bash
cd /home/bts_sh/jihoon/Demo_binary

python npz_to_eegnet_inputs.py
```

This script does not epoch or crop the data. It only changes the array layout
and creates binary label files.

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

Layout conversion:

```text
NPZ X:        (time, channels, trials)
X_eeg_raw:   (trials, channels, time)
```

Label conversion:

```text
y_trigger.npy: original labels 1..5
y.npy:         original labels shifted to 0..4
y_ab.npy:      binary labels for task ab
```

For each binary task `ab`:

```text
original label a or b -> binary label 0
all other labels      -> binary label 1
```

## 3. Train Ten Binary Models

Use `train_10_binary_randomsplit.py` to train one binary model for each of the
ten binary label files.

Minimal EEGNet run:

```bash
cd /home/bts_sh/jihoon/Demo_binary

python train_10_binary_randomsplit.py --model eegnet
```

Minimal SVM+CSP run:

```bash
cd /home/bts_sh/jihoon/Demo_binary

python train_10_binary_randomsplit.py --model svm_csp
```

The script reads:

```text
data/processed/260602_sub1_hjlee_raw/X_eeg_raw.npy
data/processed/260602_sub1_hjlee_raw/y.npy
data/processed/260602_sub1_hjlee_raw/y_01.npy ... y_34.npy
```

### Data Split

The split is random at the trial level. It is not a time-axis split.

The script first splits trial indices using `y.npy`, the original 5-class label,
as the stratification target.

Default split:

```text
train: 60%
val:   20%
test:  20%
```

The same `train_idx`, `val_idx`, and `test_idx` are reused for all ten binary
tasks. Each task only changes the label array from `y_01.npy` to `y_02.npy`,
and so on.

### EEGNet Mode

With `--model eegnet`, the script:

- normalizes `X_eeg_raw.npy` once using train-set channel-wise mean and std
- trains one EEGNet per binary task
- uses `WeightedRandomSampler` to balance binary classes in the train loader
- evaluates validation loss and balanced accuracy after each epoch
- saves the checkpoint with the lowest validation loss

The reported accuracy is balanced accuracy:

```text
(class 0 accuracy + class 1 accuracy) / 2
```

### SVM+CSP Mode

With `--model svm_csp`, the script uses raw epoch data for CSP. It does not
apply the EEGNet train-set normalization before CSP fitting.

The SVM+CSP flow is:

```text
raw epoched EEG
    -> CSP feature extraction
    -> StandardScaler
    -> SVM classifier
```

CSP is fit only on the train split for each binary task. Validation and test
sets are transformed using the fitted CSP pipeline.

## 4. Model And Output Saving

Each training run creates a timestamped run directory.

For EEGNet:

```text
checkpoints_eegnet_randomsplit/<YYYYMMDD_HHMMSS>/
```

For SVM+CSP:

```text
checkpoints_svm_csp_randomsplit/<YYYYMMDD_HHMMSS>/
```

Inside each run directory:

```text
run_config.json
results.json
train.log
y_01/
y_02/
...
y_34/
```

For EEGNet, each task directory contains:

```text
y_01/best_model.pt
y_02/best_model.pt
...
```

`best_model.pt` is selected by the lowest validation loss, not by validation
accuracy.

`train.log` records:

- hyperparameters
- input shape and device
- train/validation/test split sizes
- original label distribution
- binary label distribution for each task
- training and validation loss/accuracy every 10 epochs for EEGNet
- final validation/test result for each binary task

`results.json` stores the final metrics for all ten binary tasks.

## Useful Commands

Create the NPZ from XDF:

```bash
python xdf_to_np.py
```

Create training NPY files:

```bash
python npz_to_eegnet_inputs.py
```

Train EEGNet:

```bash
python train_10_binary_randomsplit.py --model eegnet
```

Train SVM+CSP:

```bash
python train_10_binary_randomsplit.py --model svm_csp
```
