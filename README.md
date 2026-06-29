# Demo Binary EEG Decoding

## 한국어

### 개괄 설명

이 저장소는 XDF EEG/EMG 기록을 학습용 NumPy 배열로 변환하고, 여러 binary EEG decoding 실험을 실행하기 위한 코드입니다. README는 `.gitignore`에 걸리는 원천 데이터와 생성물을 제외한 현재 폴더/파일 구조를 기준으로 설명합니다.

현재 설명 범위는 `v1`, `v2`, `v3`까지입니다. 이후 버전 관련 파일이 저장소에 있더라도 이 문서에서는 다루지 않습니다.

전체 흐름은 다음과 같습니다.

```text
XDF recordings
-> helpers/xdf_to_np.py or helpers/xdf_to_np_emg.py
-> data/npz/*.npz
-> helpers/npz_to_eegnet_inputs.py
-> data/processed/<patient>/<day>/{X_eeg.npy, X_emg.npy, y.npy}
-> trainers/train_v*.py
-> checkpoints/<patient...>/{train.log, run_config.json, results.json, cross_validation.csv}
```


### Files

```text
.
├── README.md
├── .gitignore
├── requirements_podcast.txt
├── run_v3_all_days.sh
├── configs/
│   ├── train_config_v1.json
│   ├── train_config_v2.json
│   └── train_config_v3.json
├── helpers/
│   ├── xdf_to_np.py
│   ├── xdf_to_np_emg.py
│   ├── npz_to_eegnet_inputs.py
│   ├── npz_to_eegnet_inputs_combination.py
│   └── train_v2_folds.png
├── trainers/
│   ├── train_v1_combination.py
│   ├── train_v2_ovr.py
│   ├── train_v2_setup.txt
│   └── train_v3_braindecode_eeg.py
└── checkpoints/
    ├── sub1_hjlee/
    └── sub2_yjkim/
```

#### `configs/`

- `train_config_v1.json`
  - `train_v1_combination.py`의 기본 설정입니다.
  - epoch, batch size, dropout, learning rate, weight decay, patience, seed, weighted sampler 여부를 제어합니다.

- `train_config_v2.json`
  - `train_v2_ovr.py`의 기본 설정입니다.
  - `run_5fold_cross_validation`, `balance_rest_to_target`, `selection_metric`, sliding-window diagnostic 옵션을 포함합니다.

- `train_config_v3.json`
  - `train_v3_braindecode_eeg.py`의 기본 설정입니다.
  - 설정 구조는 v2와 거의 같고, 실제 모델 종류는 실행 시 `--model eegnet` 또는 `--model shallownet`으로 고릅니다.

#### `helpers/`

- `xdf_to_np.py`
  - EEG XDF 파일을 읽고 MNE epoch를 만든 뒤 NPZ로 저장합니다.
  - EEG stream과 `trigger_stream`을 사용합니다.
  - `select_epoch`에 지정한 marker만 선택합니다.
  - `-0.2`초부터 `2.0`초까지 epoching하고 baseline을 적용한 뒤, 실제 분석 구간 `0`초부터 `2.0`초만 남깁니다.
  - 저장되는 `X` 형태는 `(time, channels, trials)`입니다.

- `xdf_to_np_emg.py`
  - EMG용 XDF 변환 스크립트입니다.
  - EMG 채널을 선택하고 EMG용 필터링을 적용한 뒤 NPZ로 저장합니다.

- `npz_to_eegnet_inputs.py`
  - `data/npz/*.npz` 파일을 날짜와 subject 기준으로 묶어 trainer 입력 배열을 만듭니다.
  - EEG는 `X_eeg.npy`, EMG는 `X_emg.npy`, label은 `y.npy`로 저장합니다.
  - 입력 `X`를 `(time, channels, trials)`에서 `(trials, channels, time)`으로 바꿉니다.
  - label은 기본적으로 `1..5`에서 `0..4`로 변환합니다.

- `npz_to_eegnet_inputs_combination.py`
  - 하나의 NPZ 파일에서 `X_eeg_raw.npy`, `y.npy`, `y_trigger.npy`, 그리고 pair-combination label 파일 `y_01.npy` 등 10개 binary label 파일을 만드는 보조 스크립트입니다.
  - v1의 10개 combination task와 맞물리는 형태입니다.

#### `trainers/`

- `train_v1_combination.py`
  - 직접 구현한 EEGNet으로 10개 pair-combination binary task를 학습합니다.
  - task 이름은 `01`, `02`, `03`, `04`, `12`, `13`, `14`, `23`, `24`, `34`입니다.
  - 예를 들어 `y_01`은 원본 label `0/1`을 binary class 0, 나머지 label `2/3/4`를 binary class 1로 둡니다.

- `train_v2_ovr.py`
  - 직접 구현한 EEGNet으로 현재 주요 task 3개를 학습합니다.
  - task는 `0_vs_rest`, `2_vs_rest`, `1_vs_3`입니다.
  - single split 또는 5-fold cross-validation을 실행할 수 있습니다.
  - `selection_metric`으로 `balanced_accuracy` 또는 `balanced_loss`를 사용할 수 있습니다.

- `train_v3_braindecode_eeg.py`
  - v2와 같은 task/split/evaluation 구조를 사용하지만, 모델을 Braindecode 구현으로 바꾼 버전입니다.
  - `--model eegnet`은 `braindecode.models.EEGNet`을 사용합니다.
  - `--model shallownet`은 `braindecode.models.ShallowFBCSPNet`을 사용합니다.

- `train_v2_setup.txt`
  - v2 실행 또는 환경 설정 관련 메모 파일입니다.

#### 기타 파일

- `requirements_podcast.txt`
  - 현재 실험 환경을 재현하기 위한 Python package 목록입니다.
  - PyTorch CUDA 12.4 wheel index와 MNE, PyXDF, scikit-learn, Braindecode 관련 의존성이 포함됩니다.

- `run_v3_all_days.sh`
  - 특정 patient의 모든 day 폴더를 순회하면서 v3 trainer를 실행하는 shell script입니다.
  - 기본값은 `PATIENT="sub1_hjlee"`, `MODEL="eegnet"`, `RUN_NAME="v3_sub1_eegnet_all_days"`입니다.

- `checkpoints/`
  - 추적 중인 `run_config.json`, `train.log`, `results.json`, `cross_validation.csv`, `cross_validation_results.json`, 일부 figure 파일이 들어 있습니다.
  - `*.pt` 모델 weight와 `_nogit` 출력 폴더는 Git에서 제외됩니다.

### 학습에 대한 간단 설명

학습 입력은 기본적으로 다음 파일입니다.

```text
data/processed/<patient>/<day>/X_eeg.npy
data/processed/<patient>/<day>/y.npy
```

`X_eeg.npy`의 shape은 `(trials, channels, time)`이고, `y.npy`는 원본 5-class label입니다. trainer는 이 5-class label을 이용해 binary task를 내부에서 구성합니다.

v1은 10개 combination task를 학습합니다.

```text
01, 02, 03, 04, 12, 13, 14, 23, 24, 34
```

v2와 v3는 현재 주요 task 3개를 학습합니다.

```text
0_vs_rest
2_vs_rest
1_vs_3
```

single split에서는 전체 trial을 original 5-class label 기준으로 stratified train/validation/test로 나눕니다. 5-fold cross-validation에서는 fold마다 test fold를 정하고, 남은 fold에서 validation split을 다시 만듭니다.

정규화는 train split의 channel-wise mean/std를 기준으로 적용합니다. 5-fold cross-validation에서는 fold마다 train fold 기준 mean/std를 다시 계산합니다.

`weighted_sampler=true`이면 train loader에서 class count 기반 `WeightedRandomSampler`를 사용합니다. validation/test split은 sampling하지 않고 그대로 평가합니다.

`balanced_loss`는 class별 평균 cross-entropy loss를 구한 뒤 두 class 평균을 다시 평균낸 값입니다.

```text
balanced_loss = mean(loss_class0_mean, loss_class1_mean)
```

`selection_metric="balanced_loss"`이면 validation balanced loss가 가장 낮은 checkpoint를 best model로 저장합니다. `selection_metric="balanced_accuracy"`이면 validation balanced accuracy가 가장 높은 checkpoint를 저장합니다.

### 파일 실행 코드

#### 1. 환경 준비

```bash
pip install -r requirements_podcast.txt
```

이미 `podcast` conda 환경을 사용 중이면 해당 환경에서 실행합니다.

#### 2. XDF -> NPZ 변환

EEG 변환:

```bash
python3 helpers/xdf_to_np.py
```

EMG 변환:

```bash
python3 helpers/xdf_to_np_emg.py
```

#### 3. NPZ -> trainer 입력 변환

여러 NPZ 파일을 subject/day별 trainer 입력으로 변환:

```bash
python3 helpers/npz_to_eegnet_inputs.py
```

v1 combination label 파일까지 만드는 보조 변환:

```bash
python3 helpers/npz_to_eegnet_inputs_combination.py
```

#### 4. v1 학습

```bash
python3 trainers/train_v1_combination.py \
  --patient sub1_hjlee \
  --day 260602 \
  --config configs/train_config_v1.json
```

#### 5. v2 학습

```bash
python3 trainers/train_v2_ovr.py \
  --patient sub1_hjlee \
  --day 260602 \
  --config configs/train_config_v2.json
```

run 이름을 명시하려면:

```bash
python3 trainers/train_v2_ovr.py \
  --patient sub1_hjlee \
  --day 260602 \
  --config configs/train_config_v2.json \
  --run-name bal_loss_100_400
```

#### 6. v3 학습

Braindecode EEGNet:

```bash
python3 trainers/train_v3_braindecode_eeg.py \
  --patient sub1_hjlee \
  --day 260602 \
  --config configs/train_config_v3.json \
  --model eegnet
```

Braindecode ShallowFBCSPNet:

```bash
python3 trainers/train_v3_braindecode_eeg.py \
  --patient sub1_hjlee \
  --day 260602 \
  --config configs/train_config_v3.json \
  --model shallownet
```

특정 patient의 모든 day를 v3로 반복 실행:

```bash
bash run_v3_all_days.sh
```

### 학습 파일 상세 설명

#### `train_v1_combination.py`

v1은 10개 pair-combination task를 한 번의 공통 split으로 학습합니다. 먼저 `X_eeg.npy`와 `y.npy`를 읽고, original label 기준으로 train/validation/test를 stratified split합니다. 이후 각 task별로 `y_01.npy` 같은 binary label 파일을 읽어 모델을 학습합니다.

모델은 파일 내부에 정의된 EEGNet입니다. 구조는 temporal convolution, depthwise spatial convolution, separable convolution, average pooling, dropout, linear classifier로 구성됩니다. 출력 class 수는 항상 2개입니다.

주요 특징은 다음과 같습니다.

```text
입력: X_eeg.npy, y.npy, y_01.npy ... y_34.npy
task: 10개 pair-combination binary task
model: local EEGNet
split: original 5-class label 기준 stratified train/val/test
selection: validation loss 최소
output: train.log, run_config.json, results.json, confusion_matrices.png, best_model.pt
```

`weighted_sampler=true`이면 train loader에 `WeightedRandomSampler`를 적용합니다. `weighted_sampler=false`이면 task별 original label 수를 맞춰 train/validation/test index를 별도로 구성하는 코드 경로가 있습니다.

#### `train_v2_ovr.py`

v2는 v1보다 task를 줄이고, 현재 실험에서 보는 핵심 binary 문제에 집중합니다.

```text
0_vs_rest: original label 0 vs 나머지
2_vs_rest: original label 2 vs 나머지
1_vs_3: original label 1 vs original label 3
```

모델은 v1과 마찬가지로 파일 내부에 정의된 local EEGNet입니다. 다만 평가 지표와 checkpoint 선택이 더 명시적입니다. `evaluate()`는 일반 loss, balanced loss, balanced accuracy, accuracy를 모두 계산합니다.

single split 모드에서는 `results.json`을 저장합니다. 5-fold cross-validation 모드에서는 fold별 `results.json`, 전체 `cross_validation.csv`, 평균/표준편차 요약을 담은 `cross_validation_results.json`을 저장합니다.

주요 특징은 다음과 같습니다.

```text
입력: X_eeg.npy, y.npy
task: 0_vs_rest, 2_vs_rest, 1_vs_3
model: local EEGNet
split: single stratified split or 5-fold cross-validation
selection: balanced_accuracy 또는 balanced_loss
optional: sliding-window diagnostic
output: train.log, run_config.json, results.json 또는 cross_validation.csv/cross_validation_results.json, best_model.pt
```

`balance_rest_to_target=true`이면 one-vs-rest task의 train split에서 rest class를 target label 수에 맞춰 downsample합니다. 현재 코드는 validation/test에는 이 balancing을 적용하지 않아 평가 population을 더 자연스럽게 유지합니다.

#### `train_v3_braindecode_eeg.py`

v3는 v2의 실험 구조를 유지하면서 모델 구현만 Braindecode로 바꾼 trainer입니다. 실행 시 `--model`로 모델을 선택합니다.

```text
--model eegnet      -> braindecode.models.EEGNet
--model shallownet  -> braindecode.models.ShallowFBCSPNet
```

데이터 split, task 구성, balanced loss, balanced accuracy, `WeightedRandomSampler`, 5-fold cross-validation, `selection_metric` 처리 방식은 v2와 거의 같습니다. 따라서 v2와 v3 결과를 비교하면 local EEGNet 구현과 Braindecode 모델 구현의 차이를 보는 실험이 됩니다.

주요 특징은 다음과 같습니다.

```text
입력: X_eeg.npy, y.npy
task: 0_vs_rest, 2_vs_rest, 1_vs_3
model: Braindecode EEGNet 또는 ShallowFBCSPNet
split: single stratified split or 5-fold cross-validation
selection: balanced_accuracy 또는 balanced_loss
output: train.log, run_config.json, results.json 또는 cross_validation.csv/cross_validation_results.json, best_model.pt
```

`run_v3_all_days.sh`는 v3 trainer를 여러 day에 반복 적용하기 위한 wrapper입니다. `PATIENT`, `MODEL`, `RUN_NAME` 값을 script 상단에서 바꾸면 다른 subject나 모델로 같은 반복 실험을 실행할 수 있습니다.

---

## English

### Overview

This repository contains a compact EEG/EMG decoding workflow. It converts XDF recordings into NumPy arrays and runs several binary EEG decoding experiments. This README describes the current folder/file structure while excluding files covered by `.gitignore`.

The scope of this document is `v1`, `v2`, and `v3` only. Files for later versions may exist in the repository, but they are intentionally not covered here.

The workflow is:

```text
XDF recordings
-> helpers/xdf_to_np.py or helpers/xdf_to_np_emg.py
-> data/npz/*.npz
-> helpers/npz_to_eegnet_inputs.py
-> data/processed/<patient>/<day>/{X_eeg.npy, X_emg.npy, y.npy}
-> trainers/train_v*.py
-> checkpoints/<patient...>/{train.log, run_config.json, results.json, cross_validation.csv}
```

`data/`, `*.xdf`, `*.npz`, `*.npy`, `*.pt`, `backup/`, `scripts/`, `__pycache__/`, and `_nogit` checkpoint folders are ignored by Git. This README therefore focuses on code, configs, runnable scripts, and tracked result summaries.

### Files

```text
.
├── README.md
├── .gitignore
├── requirements_podcast.txt
├── run_v3_all_days.sh
├── configs/
│   ├── train_config_v1.json
│   ├── train_config_v2.json
│   └── train_config_v3.json
├── helpers/
│   ├── xdf_to_np.py
│   ├── xdf_to_np_emg.py
│   ├── npz_to_eegnet_inputs.py
│   ├── npz_to_eegnet_inputs_combination.py
│   └── train_v2_folds.png
├── trainers/
│   ├── train_v1_combination.py
│   ├── train_v2_ovr.py
│   ├── train_v2_setup.txt
│   └── train_v3_braindecode_eeg.py
└── checkpoints/
    ├── sub1_hjlee/
    └── sub2_yjkim/
```

#### `configs/`

- `train_config_v1.json`
  - Default config for `train_v1_combination.py`.
  - Controls epochs, batch size, dropout, learning rate, weight decay, patience, seed, and weighted sampling.

- `train_config_v2.json`
  - Default config for `train_v2_ovr.py`.
  - Includes `run_5fold_cross_validation`, `balance_rest_to_target`, `selection_metric`, and sliding-window diagnostic options.

- `train_config_v3.json`
  - Default config for `train_v3_braindecode_eeg.py`.
  - The config structure is almost the same as v2. The model is selected at runtime with `--model eegnet` or `--model shallownet`.

#### `helpers/`

- `xdf_to_np.py`
  - Converts EEG XDF files into NPZ files.
  - Uses the EEG stream and `trigger_stream`.
  - Keeps only markers listed in `select_epoch`.
  - Epochs from `-0.2` to `2.0` seconds, applies baseline correction, then keeps the analysis window from `0` to `2.0` seconds.
  - Saves `X` as `(time, channels, trials)`.

- `xdf_to_np_emg.py`
  - Converts EMG XDF files into NPZ files.
  - Selects EMG channels and applies EMG-oriented filtering.

- `npz_to_eegnet_inputs.py`
  - Converts `data/npz/*.npz` files into trainer-ready arrays grouped by subject/day.
  - Saves EEG as `X_eeg.npy`, EMG as `X_emg.npy`, and labels as `y.npy`.
  - Converts `X` from `(time, channels, trials)` to `(trials, channels, time)`.
  - Converts labels from `1..5` to `0..4` by default.

- `npz_to_eegnet_inputs_combination.py`
  - Helper script for generating `X_eeg_raw.npy`, `y.npy`, `y_trigger.npy`, and ten pair-combination binary label files such as `y_01.npy`.
  - This format matches the v1 combination tasks.

#### `trainers/`

- `train_v1_combination.py`
  - Trains ten pair-combination binary tasks with a locally defined EEGNet.
  - Task names are `01`, `02`, `03`, `04`, `12`, `13`, `14`, `23`, `24`, and `34`.
  - For example, `y_01` maps original labels `0/1` to binary class 0 and original labels `2/3/4` to binary class 1.

- `train_v2_ovr.py`
  - Trains the current three main tasks with a locally defined EEGNet.
  - Tasks are `0_vs_rest`, `2_vs_rest`, and `1_vs_3`.
  - Supports either a single train/validation/test split or 5-fold cross-validation.
  - Supports checkpoint selection by `balanced_accuracy` or `balanced_loss`.

- `train_v3_braindecode_eeg.py`
  - Uses the same task/split/evaluation structure as v2, but swaps the model implementation to Braindecode.
  - `--model eegnet` uses `braindecode.models.EEGNet`.
  - `--model shallownet` uses `braindecode.models.ShallowFBCSPNet`.

- `train_v2_setup.txt`
  - Notes related to v2 setup or execution.

#### Other Files

- `requirements_podcast.txt`
  - Python package list for reproducing the current experiment environment.
  - Includes the PyTorch CUDA 12.4 wheel index and dependencies such as MNE, PyXDF, scikit-learn, and Braindecode-related packages.

- `run_v3_all_days.sh`
  - Runs the v3 trainer over every day folder for a selected patient.
  - Defaults are `PATIENT="sub1_hjlee"`, `MODEL="eegnet"`, and `RUN_NAME="v3_sub1_eegnet_all_days"`.

- `checkpoints/`
  - Contains tracked files such as `run_config.json`, `train.log`, `results.json`, `cross_validation.csv`, `cross_validation_results.json`, and some figures.
  - `*.pt` model weights and `_nogit` output folders are ignored by Git.

### Training Summary

The main training inputs are:

```text
data/processed/<patient>/<day>/X_eeg.npy
data/processed/<patient>/<day>/y.npy
```

`X_eeg.npy` has shape `(trials, channels, time)`, and `y.npy` contains the original 5-class labels. The trainers build binary tasks internally from these 5-class labels.

v1 trains ten combination tasks:

```text
01, 02, 03, 04, 12, 13, 14, 23, 24, 34
```

v2 and v3 train the current three main tasks:

```text
0_vs_rest
2_vs_rest
1_vs_3
```

For single-split runs, trials are split into train/validation/test sets with stratification by the original 5-class labels. For 5-fold cross-validation, each fold becomes the test fold once, and a validation split is created from the remaining folds.

Normalization uses channel-wise mean/std from the training split only. In 5-fold cross-validation, the mean/std is recomputed separately for each fold.

When `weighted_sampler=true`, the train loader uses a class-count-based `WeightedRandomSampler`. Validation and test sets are evaluated without sampling.

`balanced_loss` is the average of the two class-wise mean cross-entropy losses.

```text
balanced_loss = mean(loss_class0_mean, loss_class1_mean)
```

If `selection_metric="balanced_loss"`, the best checkpoint is selected by the lowest validation balanced loss. If `selection_metric="balanced_accuracy"`, it is selected by the highest validation balanced accuracy.

### Run Commands

#### 1. Prepare environment

```bash
pip install -r requirements_podcast.txt
```

If the `podcast` conda environment is already available, run the commands inside that environment.

#### 2. Convert XDF to NPZ

EEG conversion:

```bash
python3 helpers/xdf_to_np.py
```

EMG conversion:

```bash
python3 helpers/xdf_to_np_emg.py
```

#### 3. Convert NPZ files to trainer inputs

Convert multiple NPZ files into subject/day trainer inputs:

```bash
python3 helpers/npz_to_eegnet_inputs.py
```

Generate v1 combination label files:

```bash
python3 helpers/npz_to_eegnet_inputs_combination.py
```

#### 4. Run v1

```bash
python3 trainers/train_v1_combination.py \
  --patient sub1_hjlee \
  --day 260602 \
  --config configs/train_config_v1.json
```

#### 5. Run v2

```bash
python3 trainers/train_v2_ovr.py \
  --patient sub1_hjlee \
  --day 260602 \
  --config configs/train_config_v2.json
```

With an explicit run name:

```bash
python3 trainers/train_v2_ovr.py \
  --patient sub1_hjlee \
  --day 260602 \
  --config configs/train_config_v2.json \
  --run-name bal_loss_100_400
```

#### 6. Run v3

Braindecode EEGNet:

```bash
python3 trainers/train_v3_braindecode_eeg.py \
  --patient sub1_hjlee \
  --day 260602 \
  --config configs/train_config_v3.json \
  --model eegnet
```

Braindecode ShallowFBCSPNet:

```bash
python3 trainers/train_v3_braindecode_eeg.py \
  --patient sub1_hjlee \
  --day 260602 \
  --config configs/train_config_v3.json \
  --model shallownet
```

Run v3 for all days of the configured patient:

```bash
bash run_v3_all_days.sh
```

### Training File Details

#### `train_v1_combination.py`

v1 trains ten pair-combination tasks using one shared split. It loads `X_eeg.npy` and `y.npy`, creates a stratified train/validation/test split by the original labels, then loads task-specific binary label files such as `y_01.npy`.

The model is a locally defined EEGNet. It consists of temporal convolution, depthwise spatial convolution, separable convolution, average pooling, dropout, and a linear classifier. The output dimension is always 2.

Key properties:

```text
input: X_eeg.npy, y.npy, y_01.npy ... y_34.npy
tasks: ten pair-combination binary tasks
model: local EEGNet
split: stratified train/val/test by original 5-class labels
selection: minimum validation loss
output: train.log, run_config.json, results.json, confusion_matrices.png, best_model.pt
```

When `weighted_sampler=true`, the train loader uses `WeightedRandomSampler`. When `weighted_sampler=false`, the script has a separate path that balances original-label counts for each task split.

#### `train_v2_ovr.py`

v2 reduces the task set and focuses on the current main binary decoding problems.

```text
0_vs_rest: original label 0 vs all other labels
2_vs_rest: original label 2 vs all other labels
1_vs_3: original label 1 vs original label 3
```

The model is the same locally defined EEGNet style as v1. The evaluation logic is more explicit: `evaluate()` reports ordinary loss, balanced loss, balanced accuracy, and accuracy.

Single-split mode writes `results.json`. 5-fold cross-validation mode writes per-fold `results.json`, a global `cross_validation.csv`, and `cross_validation_results.json` with mean/std summaries.

Key properties:

```text
input: X_eeg.npy, y.npy
tasks: 0_vs_rest, 2_vs_rest, 1_vs_3
model: local EEGNet
split: single stratified split or 5-fold cross-validation
selection: balanced_accuracy or balanced_loss
optional: sliding-window diagnostic
output: train.log, run_config.json, results.json or cross_validation.csv/cross_validation_results.json, best_model.pt
```

When `balance_rest_to_target=true`, the one-vs-rest training split downsamples the rest labels to match the target-label count. The current code does not apply that balancing to validation/test splits, so evaluation keeps the natural validation/test population.

#### `train_v3_braindecode_eeg.py`

v3 keeps the v2 experiment structure but replaces the model implementation with Braindecode models. The model is selected at runtime with `--model`.

```text
--model eegnet      -> braindecode.models.EEGNet
--model shallownet  -> braindecode.models.ShallowFBCSPNet
```

Data splitting, task construction, balanced loss, balanced accuracy, `WeightedRandomSampler`, 5-fold cross-validation, and `selection_metric` behavior are almost the same as v2. Comparing v2 and v3 therefore mainly compares the local EEGNet implementation against Braindecode model implementations.

Key properties:

```text
input: X_eeg.npy, y.npy
tasks: 0_vs_rest, 2_vs_rest, 1_vs_3
model: Braindecode EEGNet or ShallowFBCSPNet
split: single stratified split or 5-fold cross-validation
selection: balanced_accuracy or balanced_loss
output: train.log, run_config.json, results.json or cross_validation.csv/cross_validation_results.json, best_model.pt
```

`run_v3_all_days.sh` is a wrapper for applying the v3 trainer to multiple day folders. Change `PATIENT`, `MODEL`, and `RUN_NAME` at the top of the script to run the same repeated experiment for a different subject or model.
