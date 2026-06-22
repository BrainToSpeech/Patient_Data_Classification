# xdf_to_np.py
#
# 목적:
#   XDF 파일 안에 들어있는 EEG stream과 trigger_stream을 읽어서
#   모델/분석에 쓰기 쉬운 NPZ 파일(X, y)로 저장한다.
#
# 최종 저장 형태:
#   X: (time, channels, trials)
#      예: (1000, 64, 500)
#      - time     : 한 trial 안의 시간 sample
#      - channels : EEG 채널
#      - trials   : trial 개수
#   y: (trials,)
#      예: [1, 2, 3, 4, 5, ...]
#      - 각 trial의 class label
#
# 큰 흐름:
#   1. pyxdf로 XDF 파일 읽기
#   2. streams 안에서 EEG stream과 trigger_stream 찾기
#   3. EEG stream을 MNE RawArray로 변환
#   4. 필터링 / notch / average reference 적용
#   5. trigger timestamp를 EEG sample index로 변환
#   6. MNE Epochs로 trial 단위 자르기
#   7. X, y를 npz로 저장
import mne
import numpy as np
import os
import glob
from scipy.io import savemat
import pyxdf

def load_xdf_file(file_path):
    # pyxdf.load_xdf()는 XDF 파일 전체를 읽고 두 값을 반환한다.
    #
    # streams:
    #   XDF 안에 들어있는 여러 데이터 stream의 list.``
    #   보통 EEG stream, trigger_stream 같은 것들이 각각 하나의 dict로 들어있다.
    #
    # header:
    #   XDF 파일 전체 header 정보.
    #   여기서는 직접 쓰지 않지만, 파일 메타데이터 확인용으로 받을 수 있다.
    streams, header = pyxdf.load_xdf(file_path)
    
    # 아래 변수들은 streams를 돌면서 채워진다.
    # 처음에는 None으로 두고, 원하는 stream을 찾으면 실제 값을 넣는다.
    eeg_data = None
    eeg_times = None
    ch_names = None
    event_ids = None
    event_times = None
    sfreq = None
    
    for stream in streams:
        # stream["info"]에는 stream 이름, 타입, 샘플링레이트, 채널 정보가 들어있다.
        # stream["time_series"]에는 실제 데이터가 들어있다.
        # stream["time_stamps"]에는 각 데이터 sample/event가 발생한 시간이 들어있다.

        if stream['info']['type'][0] == 'EEG' and eeg_data is None: # EEG 데이터 선택
            # EEG stream의 time_series는 보통 (samples, channels) 형태다.
            # MNE RawArray는 (channels, samples) 형태를 기대하므로 .T로 transpose한다.
            eeg_data = np.array(stream['time_series']).T

            # EEG sample마다 붙어있는 LSL timestamp.
            # trigger timestamp와 맞춰서 epoch 시작 위치를 찾는 데 사용한다.
            eeg_times = np.array(stream['time_stamps'])

            # nominal_srate: XDF에 기록된 샘플링레이트.
            # 예: 500 Hz이면 1초에 500 sample.
            sfreq = float(stream['info']['nominal_srate'][0])

            # 간혹 timestamp가 비어 있으면 sample index / sfreq로 시간축을 임시 생성한다.
            # 정상 XDF라면 eeg_times가 들어있는 것이 보통이다.
            if not eeg_times.size:
                eeg_times = np.arange(eeg_data.shape[1]) / sfreq

            # 채널 이름 목록을 XDF metadata에서 꺼낸다.
            # 예: ["Fp1", "Fz", "F3", ...]
            ch_names = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
        
        elif stream['info']['name'][0] == 'trigger_stream' and event_ids is None: 
            # trigger_stream은 실험 이벤트 marker가 들어있는 stream이다.
            # 예: "101", "1", "102", "2", ...
            #
            # event_ids:
            #   marker 값들.
            #
            # event_times:
            #   각 marker가 발생한 timestamp.
            event_ids = np.array(stream['time_series'])
            event_times = np.array(stream['time_stamps'])

    if eeg_data is None or event_ids is None or sfreq is None:
        # EEG 또는 trigger가 없으면 epoch를 만들 수 없으므로 바로 중단한다.
        raise ValueError("EEG data, event trigger stream, or sfreq not found in the XDF file.")
    
    # 이후 함수에서 필요한 원본 데이터와 메타데이터를 반환한다.
    return eeg_data, eeg_times, ch_names, event_ids, event_times, sfreq

def mainEEG(file_path, channels, select_epoch, upper_bound_freq):
    # 이 함수가 실제 EEG 전처리의 중심이다.
    #
    # 입력:
    #   file_path        : 변환할 XDF 파일 경로
    #   channels         : 사용할 채널 번호 목록. 1부터 시작한다고 가정한다.
    #   select_epoch     : trial로 사용할 marker 목록. 예: ["1", "2", "3", "4", "5"]
    #   upper_bound_freq : bandpass filter의 high cutoff frequency

    # XDF에서 EEG 데이터, EEG timestamp, 채널명, trigger marker, trigger timestamp를 읽는다.
    # print(f"Processing file: {file_path}")
    
    eeg_data, eeg_times, ch_names, event_ids, event_times, sfreq = load_xdf_file(file_path)
    
    # 샘플링레이트가 0 이하이면 시간축을 만들 수 없으므로 오류 처리한다.
    if sfreq <= 0:
        raise ValueError(f"Invalid sfreq: {sfreq}. Sampling frequency must be positive.")
    
    # MNE에서 EEG 데이터를 다루려면 RawArray 객체로 감싸야 한다.
    #
    # info:
    #   채널 이름, 샘플링레이트, 채널 타입 등을 담은 MNE metadata.
    #
    # raw:
    #   연속 EEG 신호 전체.
    #   아직 trial로 잘리지 않은 상태다.
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)
    
    # channels는 1부터 시작하는 번호라고 가정한다.
    # Python list index는 0부터 시작하므로 ch-1을 사용한다.
    #
    # 예:
    #   channels = [1, 2, 3]
    #   -> raw.ch_names[0], raw.ch_names[1], raw.ch_names[2] 선택
    raw.pick_channels([raw.ch_names[ch-1] for ch in channels])

    # 나중에 저장 전후 채널 순서가 바뀌지 않았는지 확인하기 위해 저장해둔다.
    original_channel_order = raw.ch_names.copy()
    
    # Bandpass filtering:
    #   0.5 Hz 아래의 느린 drift를 줄이고,
    #   upper_bound_freq 위의 높은 주파수 성분을 제거한다.
    raw.filter(l_freq=0.5, h_freq=upper_bound_freq, fir_design='firwin')
    
    # Notch filtering:
    #   전원 잡음인 60 Hz 및 그 배수 성분을 제거한다.
    #
    # np.arange(60, upper_bound_freq, 60):
    #   upper_bound_freq가 100이면 [60]
    #   upper_bound_freq가 180이면 [60, 120]
    notch_freq = 60
    freqs = np.arange(notch_freq, upper_bound_freq, notch_freq)
    raw.notch_filter(freqs=freqs, fir_design='firwin')
    
    # Average reference:
    #   선택된 EEG 채널들의 평균을 reference로 사용한다.
    #   projection=True라서 projection 형태로 추가된다.
    raw.set_eeg_reference('average', projection=True)

    # MNE Epochs는 event code가 숫자여야 한다.
    # select_epoch marker 문자열을 1,2,3,... 숫자로 매핑한다.
    #
    # 예:
    #   select_epoch = ["1", "2", "3", "4", "5"]
    #   event_id_dict = {"1": 1, "2": 2, ...}
    event_id_dict = {str(e): i+1 for i, e in enumerate(select_epoch)}

    # event_ids는 [["1"], ["101"], ...]처럼 2차원일 수 있으므로 1차원으로 편다.
    flat_ids = event_ids.flatten()

    # trigger marker가 select_epoch에 있으면 해당 숫자로 바꾸고,
    # 없으면 -1로 표시한다.
    #
    # 주의:
    #   여기서는 모든 event를 events 배열에 넣고, MNE Epochs 단계에서
    #   event_id_dict에 해당하는 event만 epoch로 사용한다.
    mapped_event_ids = np.vectorize(event_id_dict.get)(flat_ids, -1)
    
    # trigger timestamp를 EEG sample index로 변환한다.
    #
    # event_times - eeg_times[0]:
    #   EEG 시작 시점 기준으로 trigger가 몇 초 뒤에 발생했는지.
    #
    # * sfreq:
    #   초 단위를 sample index로 변환.
    #
    # events는 MNE가 요구하는 형식:
    #   [sample_index, previous_event_value, event_code]
    #   여기서는 previous_event_value를 쓰지 않으므로 0으로 둔다.
    event_samples = (event_times - eeg_times[0]) * sfreq
    events = np.column_stack([event_samples.astype(int), np.zeros(len(event_samples), dtype=int), mapped_event_ids])
    
    # Epoching:
    #   연속 raw 데이터에서 event 기준으로 trial을 잘라낸다.
    #
    # tmin=-0.2, tmax=2:
    #   trigger 0.2초 전부터 trigger 2초 후까지 자른다.
    #
    # baseline=(-0.2, 0):
    #   trigger 직전 0.2초 구간 평균을 baseline으로 빼준다.
    #
    # preload=True:
    #   데이터를 메모리에 실제로 로드해서 이후 crop/get_data를 빠르게 한다.
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=-0.2, tmax=2, baseline=(-0.2, 0), detrend=None, preload=True)

    # Drop log 출력: 어떤 epoch이 왜 제거됐는지 확인
    # 예를 들어 epoch 구간이 raw 범위를 벗어나면 MNE가 해당 epoch를 drop할 수 있다.
    # 여기서 Total events와 Kept 개수를 보면 trial이 예상보다 줄었는지 확인할 수 있다.
    print(f"  [Drop log] Total events: {len(events)}, Kept: {len(epochs)}, Dropped: {len(events) - len(epochs)}")
    drop_reasons = {}
    for reason in epochs.drop_log:
        if reason:
            key = reason[0]
            drop_reasons[key] = drop_reasons.get(key, 0) + 1
    for reason, count in drop_reasons.items():
        print(f"    - {reason}: {count}")

    # Artifact removal / bad channel interpolation:
    #   현재 코드에서는 자동으로 bad channel을 찾지는 않는다.
    #   다만 epochs.info["bads"]에 이미 bad channel이 들어있다면 보간한다.
    # epochs = epochs.copy().apply_baseline(baseline=(-0.2, 0))
    if len(epochs.info['bads']) > 0:  # 나쁜 채널이 있는 경우에만 보간 수행
        epochs = epochs.copy().interpolate_bads()
    
    # baseline 계산을 위해 처음에는 -0.2~2초를 잘랐고,
    # 저장할 때는 실제 분석 구간인 0~2초만 남긴다.
    epochs = epochs.crop(tmin=0, tmax=2)  # baseline 구간(-0.2~0) 제거 후 실제 분석 구간만 유지

    # epochs:
    #   MNE Epochs 객체. shape 개념상 (trials, channels, time)
    #
    # original_channel_order:
    #   저장 전 채널 순서 확인용.
    return epochs, original_channel_order

def concatEEG(file_path, channels, select_epoch, upper_bound_freq):
    # 예전에는 여러 run/session의 epochs를 합치는 용도였을 수 있다.
    # 현재 코드는 mainEEG 결과 하나만 그대로 반환한다.
    epochs_1, original_channel_order = mainEEG(file_path, channels, select_epoch, upper_bound_freq)
    
    # nTrial을 지정하지 않고, 선택한 에폭의 수만큼 처리
    epochs = epochs_1
    
    return epochs, original_channel_order

def save_data(root_dir, target_dir, subject_filename, X, y):
    # 최종 npz 저장 함수.
    #
    # 저장 경로:
    #   root_dir / target_dir / subject_filename.npz
    #
    # npz 안에는 X, y 두 배열이 들어간다.
    os.makedirs(os.path.join(root_dir, target_dir), exist_ok=True)  # target_dir이 존재하지 않으면 생성
    save_path = os.path.join(root_dir, target_dir, f"{subject_filename}.npz")
    np.savez(save_path, X=X, y=y)
    print(f"Data saved to {save_path}")

def process_data(root_dir, target_dir, file_path, select_epoch, channels, upper_bound_freq, trials_per_trigger=4):
    # 한 개 XDF 파일을 읽어서 한 개 npz 파일로 저장하는 함수.
    #
    # trials_per_trigger는 현재 코드에서는 사용하지 않는다.
    # 예전 트리거별 저장 코드에서 쓰던 인자로 보인다.

    # 원본 파일명에서 확장자를 제거하여 subject_filename 생성
    subject_filename = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(root_dir, target_dir, f"{subject_filename}.npz")
    
    # 이미 변환된 파일이 존재하는지 확인
    save_path = save_path.strip()
    print(f"** Processing file: {subject_filename}")
    
    try:
        epochs, original_channel_order = concatEEG(file_path, channels, select_epoch, upper_bound_freq)
    except FileNotFoundError as e:
        print(e)
        print(f"Skipping file: {subject_filename} due to missing file.")
        return
    
    # MNE Epochs에서 numpy 배열을 꺼낸다.
    #
    # epochs.get_data()의 기본 shape:
    #   (trials, channels, time)
    #
    # 기존 processed npz는:
    #   (time, channels, trials)
    #
    # 이 형식에 맞추기 위해 아래에서 transpose한다.
    X = epochs.get_data(copy=True)  # shape: (trials, channels, time)

    # crop 결과는 0초 sample을 포함한다.
    # 기존 저장 형식에 맞추기 위해 첫 sample 하나를 제거한다.
    #
    # 예:
    #   500 Hz에서 0~2초 crop은 보통 1001 samples가 될 수 있다.
    #   첫 sample 제거 후 1000 samples.
    X = X[:, :, 1:]  # Remove the first time sample to keep only 5000 samples

    # 저장 축을 기존 npz 스타일로 바꾼다.
    #
    # 변경 전: (trials, channels, time)
    # 변경 후: (time, channels, trials)
    X = np.transpose(X, (2, 1, 0))  # Change shape to (time, channels, trials)
    
    # Verify channel order
    # transpose는 축 순서만 바꾸고 채널 이름 순서를 바꾸지는 않는다.
    # 그래서 original_channel_order가 그대로인지 확인한다.
    channel_order_after_transpose = original_channel_order  # Channels order should be unchanged
    assert original_channel_order == channel_order_after_transpose, "Channel order has changed!"
    
    # y는 각 epoch의 event code를 저장한다.
    # event_id_dict에서 "1"->1, "2"->2로 매핑했으므로 y도 1~5 label이 된다.
    y = np.array([epochs.events[i, -1] for i in range(len(epochs))])
    
    # Save data
    save_data(root_dir, target_dir, subject_filename, X, y)

# Main execution
# 이 아래는 스크립트를 직접 실행했을 때 돌아가는 부분이다.
#
# 현재 설정:
#   - 이 파일이 있는 폴더(root_dir)에서 *.xdf를 모두 찾는다.
#   - 각 XDF를 EEG 폴더에 .npz로 저장한다.
root_dir = os.path.dirname(os.path.abspath(__file__))  # root directory
target_dir = os.path.join('EEG')  # save preprocessed data
os.makedirs(os.path.join(root_dir, target_dir), exist_ok=True)

# trial로 사용할 marker.
# 여기서는 trigger_stream의 "1", "2", "3", "4", "5"를 class label로 사용한다.
select_epoch = ['1', '2', '3', '4', '5']

# 앞 64개 채널을 사용한다.
# 이 코드에서는 1-based 번호로 지정하므로 1~64.
channels = list(range(1, 65))

# bandpass filter의 상한 주파수.
upper_bound_freq = 100

# 지정된 폴더 내의 모든 XDF 파일을 처리
# root_dir, 즉 이 스크립트가 있는 폴더의 모든 .xdf 파일을 변환한다.
xdf_files = glob.glob(os.path.join(root_dir, f"*.xdf"))
for xdf_file in xdf_files:
    print("==================================================")
    process_data(root_dir, target_dir, xdf_file, select_epoch, channels, upper_bound_freq)
    print("\n\n\n")
