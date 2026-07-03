# Demo Binary EEG Decoding

## 개괄 설명

이 저장소는 XDF로 저장된 EEG/EMG 데이터를 학습용 NumPy 배열로 변환하고, Braindecode 기반 binary decoding 모델을 학습하는 실험 코드입니다.

전체 흐름은 아래처럼 보면 됩니다. 파일 이름만 나열한 것이 아니라, 각 단계에서 무엇을 하는지도 같이 적었습니다.

```text
1. 원본 기록 준비
   XDF recording

2. XDF를 epoch 단위 NPZ로 변환
   helpers/xdf_to_np.py      -> EEG epoch NPZ 생성
   helpers/xdf_to_np_emg.py  -> EMG epoch NPZ 생성

3. NPZ를 trainer 입력 형태로 변환
   helpers/npz_to_eegnet_inputs.py
   -> data/processed/<patient>/<day>/X_eeg.npy
   -> data/processed/<patient>/<day>/X_emg.npy
   -> data/processed/<patient>/<day>/y.npy

4. 학습 실행
   trainers/train_v3_braindecode_eeg.py                  -> day별 EEG only
   trainers/train_v4_braindecode_eeg_emg.py              -> day별 EEG + EMG
   trainers/train_v5_braindecode_eeg_all_sessions.py     -> 모든 day 합친 EEG only
   trainers/train_v6_braindecode_eeg_emg_all_sessions.py -> 모든 day 합친 EEG + EMG

5. 결과 확인
   checkpoints/<patient>_nogit/<run_name>/<timestamp>/
   -> train.log
   -> run_config.json
   -> results.json 또는 cross_validation.csv / cross_validation_results.json
```

학습 입력 배열의 기본 shape은 다음과 같습니다.

```text
X_eeg.npy: (trials, eeg_channels, time)
X_emg.npy: (trials, emg_channels, time)
y.npy:     (trials,)
```

현재 binary task는 주로 다음 세 가지입니다.

```text
0_vs_rest
2_vs_rest
1_vs_3
```

`--model eegnet`을 주면 Braindecode EEGNet을 쓰고, `--model shallownet`을 주면 Braindecode ShallowFBCSPNet을 씁니다.

## 버전별 학습 파일

`v1`, `v2`는 이전 실험용 코드입니다.

- `trainers/train_v1_combination.py`: 10개 pair-combination binary task를 학습하던 초기 버전입니다.
- `trainers/train_v2_ovr.py`: `0_vs_rest`, `2_vs_rest`, `1_vs_3` 중심으로 정리한 local EEGNet 버전입니다.

현재 주요 흐름은 `v3`부터 보면 됩니다.

| 버전 | 파일 | 데이터 범위 | 입력 modality | 요약 |
| --- | --- | --- | --- | --- |
| v3 | `train_v3_braindecode_eeg.py` | day separate | EEG only | 한 session/day씩 EEG만 학습 |
| v4 | `train_v4_braindecode_eeg_emg.py` | day separate | EEG + EMG | 한 session/day씩 EEG feature와 EMG feature를 concat해서 학습 |
| v5 | `train_v5_braindecode_eeg_all_sessions.py` | all sessions | EEG only | 한 patient의 모든 day를 합쳐 EEG만 학습 |
| v6 | `train_v6_braindecode_eeg_emg_all_sessions.py` | all sessions | EEG + EMG | 한 patient의 모든 day를 합쳐 EEG+EMG fusion 학습 |

`v4`, `v6`의 EEG+EMG 방식은 raw signal을 처음부터 붙이는 방식이 아닙니다. EEG는 Braindecode 모델에서 feature를 뽑고, EMG는 간단한 MLP에서 feature를 뽑은 뒤, 두 feature vector를 concat해서 최종 binary classifier에 넣습니다.

```text
X_eeg -> Braindecode EEGNet/ShallowNet -> EEG feature
X_emg -> Flatten + MLP                 -> EMG feature

concat(EEG feature, EMG feature)
-> final classifier
-> binary logits
```

## helpers 파일 설명

- `helpers/xdf_to_np.py`: EEG XDF 파일을 읽어 marker 기준 epoch를 만들고 NPZ로 저장합니다.
- `helpers/xdf_to_np_emg.py`: EMG 채널을 읽어 EMG용 filtering/epoching 후 NPZ로 저장합니다.
- `helpers/npz_to_eegnet_inputs.py`: NPZ 파일들을 `X_eeg.npy`, `X_emg.npy`, `y.npy` 형태로 변환해 `data/processed/<patient>/<day>/`에 저장합니다.
- `helpers/npz_to_eegnet_inputs_combination.py`: v1용 pair-combination label 파일(`y_01.npy` 등)을 만드는 보조 변환 스크립트입니다.
- `helpers/summarize_cv_bal_acc.py`: 여러 run의 `cross_validation.csv`를 모아 balanced accuracy 평균/표준편차를 요약합니다.
- `helpers/plot_result_csvs.py`: 요약 CSV를 figure로 그립니다.
- `helpers/train_v2_folds.png`: v2 fold 구조 설명용 이미지입니다.

## 실행 예시

아래 예시는 repo root(`/home/bts_sh/jihoon/Demo_binary`)에서 실행한다고 가정합니다.

### 1. 환경 준비

```bash
pip install -r requirements_podcast.txt
```

이미 `podcast` conda 환경을 쓰고 있다면 해당 환경을 activate한 뒤 실행하면 됩니다.

```bash
conda activate podcast
```

### 2. XDF를 NPZ로 변환

EEG 변환:

```bash
python3 helpers/xdf_to_np.py
```

EMG 변환:

```bash
python3 helpers/xdf_to_np_emg.py
```

### 3. NPZ를 trainer 입력으로 변환

```bash
python3 helpers/npz_to_eegnet_inputs.py
```

실행 후 day별 폴더에 아래 파일들이 생기는지 확인합니다.

```text
data/processed/<patient>/<day>/X_eeg.npy
data/processed/<patient>/<day>/X_emg.npy
data/processed/<patient>/<day>/y.npy
```

EMG가 없는 subject/day는 `X_emg.npy`가 없을 수 있습니다. 그런 경우 `v4`, `v6`은 바로 실행할 수 없고, EEG only인 `v3`, `v5`를 사용해야 합니다.

### 4. v3 실행: day별 EEG only

한 day만 실행하는 예시입니다.

```bash
python3 trainers/train_v3_braindecode_eeg.py \
  --patient sub1_hjlee \
  --day 260602 \
  --model eegnet \
  --run-name v3_sub1_260602_eegnet
```

ShallowNet으로 실행하려면:

```bash
python3 trainers/train_v3_braindecode_eeg.py \
  --patient sub1_hjlee \
  --day 260602 \
  --model shallownet \
  --run-name v3_sub1_260602_shallownet
```

### 5. v4 실행: day별 EEG + EMG

`X_eeg.npy`, `X_emg.npy`, `y.npy`가 모두 있는 day에서 실행합니다.

```bash
python3 trainers/train_v4_braindecode_eeg_emg.py \
  --patient sub1_hjlee \
  --day 260602 \
  --model eegnet \
  --run-name v4_sub1_260602_eeg_emg
```

EMG 파일명이 기본값 `X_emg.npy`가 아니라면 `--emg-file`로 지정합니다.

```bash
python3 trainers/train_v4_braindecode_eeg_emg.py \
  --patient sub1_hjlee \
  --day 260602 \
  --model eegnet \
  --emg-file X_emg.npy \
  --run-name v4_sub1_260602_eeg_emg
```

### 6. v5 실행: 모든 day 합친 EEG only

한 patient 아래의 모든 day를 합쳐서 학습합니다.

```bash
python3 trainers/train_v5_braindecode_eeg_all_sessions.py \
  --patient sub1_hjlee \
  --model eegnet \
  --run-name v5_sub1_all_days_eegnet
```

이 파일은 `data/processed/sub1_hjlee/<day>/X_eeg.npy`와 `y.npy`가 있는 모든 day를 모아서 사용합니다.

### 7. v6 실행: 모든 day 합친 EEG + EMG

한 patient 아래의 모든 day를 합치되, EEG와 EMG를 함께 사용합니다.

```bash
python3 trainers/train_v6_braindecode_eeg_emg_all_sessions.py \
  --patient sub1_hjlee \
  --model eegnet \
  --run-name v6_sub1_all_days_eeg_emg
```

`v6`은 각 day 폴더에 `X_eeg.npy`, `X_emg.npy`, `y.npy`가 모두 있어야 합니다.

### 8. 여러 day를 반복 실행

`run_all_days.sh`는 patient 아래의 day 폴더를 순회하면서 day별 trainer를 반복 실행하는 shell script입니다. 스크립트 상단에서 `PATIENT`, `MODEL`, `RUN_NAME`, `TRAIN_SCRIPT`를 바꾼 뒤 실행합니다.

```bash
bash run_all_days.sh
```

예를 들어 `TRAIN_SCRIPT`가 `trainers/train_v4_braindecode_eeg_emg.py`로 되어 있으면 각 day에 대해 v4 EEG+EMG 학습을 반복합니다.

### 9. 결과 요약

5-fold cross-validation 결과를 요약하려면 run root를 넣습니다.

```bash
python3 helpers/summarize_cv_bal_acc.py \
  checkpoints/sub1_hjlee_nogit/v6_sub1_all_days_eeg_emg \
  --output-csv checkpoints/sub1_hjlee_nogit/v6_summary.csv
```

요약 CSV를 그림으로 만들려면:

```bash
python3 helpers/plot_result_csvs.py checkpoints/sub1_hjlee_nogit/v6_summary.csv
```

## 출력 파일

학습 결과는 보통 아래 위치에 저장됩니다.

```text
checkpoints/<patient>_nogit/<run_name>/<timestamp>/
```

주요 파일은 다음과 같습니다.

- `train.log`: 학습 설정, split 분포, epoch별 metric, 최종 결과 로그입니다.
- `run_config.json`: 실행 당시 config, 입력 shape, day 목록, task 목록을 저장합니다.
- `results.json`: single split 실행 결과입니다.
- `cross_validation.csv`: fold별 결과를 표 형태로 저장합니다.
- `cross_validation_results.json`: fold별 결과와 평균/표준편차 요약입니다.
- `best_model.pt`: 각 task/fold의 best checkpoint입니다. 이 파일은 Git에서 제외됩니다.

## 참고

- raw XDF, NPZ/NPY 배열, `*.pt` 모델 weight, `_nogit` checkpoint 폴더는 Git에서 제외됩니다.
- `v3`, `v4`는 day별 실험을 볼 때 사용합니다.
- `v5`, `v6`은 한 patient의 여러 day를 모두 합쳐 학습할 때 사용합니다.
- `v3`, `v5`는 EEG only입니다.
- `v4`, `v6`은 EEG + EMG fusion입니다.
