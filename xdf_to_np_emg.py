# xdf_to_np_emg.py
import mne
import numpy as np
import os
import glob
from scipy.io import savemat
import pyxdf

def load_xdf_file(file_path):
    streams, header = pyxdf.load_xdf(file_path)
    
    eeg_data = None
    eeg_times = None
    ch_names = None
    event_ids = None
    event_times = None
    sfreq = None
    
    for stream in streams:
        if stream['info']['type'][0] == 'EEG' and eeg_data is None: # EEG 데이터 선택
            eeg_data = np.array(stream['time_series']).T
            eeg_times = np.array(stream['time_stamps'])
            sfreq = float(stream['info']['nominal_srate'][0])
            if not eeg_times.size:
                eeg_times = np.arange(eeg_data.shape[1]) / sfreq
            ch_names = [ch['label'][0] for ch in stream['info']['desc'][0]['channels'][0]['channel']]
        
        elif stream['info']['name'][0] == 'trigger_stream' and event_ids is None: 
            event_ids = np.array(stream['time_series'])
            event_times = np.array(stream['time_stamps'])

    if eeg_data is None or event_ids is None or sfreq is None:
        raise ValueError("EEG data, event trigger stream, or sfreq not found in the XDF file.")
    
    return eeg_data, eeg_times, ch_names, event_ids, event_times, sfreq

def mainEEG(file_path, channels, select_epoch, lower_bound_freq, upper_bound_freq):
    # Load the XDF data
    # print(f"Processing file: {file_path}")
    
    eeg_data, eeg_times, ch_names, event_ids, event_times, sfreq = load_xdf_file(file_path)
    
    # Ensure sfreq is valid
    if sfreq <= 0:
        raise ValueError(f"Invalid sfreq: {sfreq}. Sampling frequency must be positive.")
    
    # Create MNE Raw object
    ch_types = ["eeg"] * len(ch_names)
    for ch in channels:
        ch_types[ch - 1] = "emg"
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(eeg_data, info)
    
    # Select channels
    raw.pick([raw.ch_names[ch-1] for ch in channels])
    original_channel_order = raw.ch_names.copy()
    
    # Filtering
    raw.filter(l_freq=lower_bound_freq, h_freq=upper_bound_freq, picks="emg", fir_design='firwin')
    
    # Notch filtering
    notch_freq = 60
    freqs = np.arange(notch_freq, upper_bound_freq, notch_freq)
    freqs = freqs[freqs > lower_bound_freq]
    raw.notch_filter(freqs=freqs, picks="emg", fir_design='firwin')

    # Map event IDs
    event_id_dict = {str(e): i+1 for i, e in enumerate(select_epoch)}
    flat_ids = event_ids.flatten()
    mapped_event_ids = np.vectorize(event_id_dict.get)(flat_ids, -1)
    
    # Convert event times to samples
    event_samples = (event_times - eeg_times[0]) * sfreq
    events = np.column_stack([event_samples.astype(int), np.zeros(len(event_samples), dtype=int), mapped_event_ids])
    
    # Epoching with initial range for baseline correction
    epochs = mne.Epochs(raw, events, event_id=event_id_dict, tmin=-0.2, tmax=2, baseline=(-0.2, 0), detrend=None, preload=True)

    # Drop log 출력: 어떤 epoch이 왜 제거됐는지 확인
    print(f"  [Drop log] Total events: {len(events)}, Kept: {len(epochs)}, Dropped: {len(events) - len(epochs)}")
    drop_reasons = {}
    for reason in epochs.drop_log:
        if reason:
            key = reason[0]
            drop_reasons[key] = drop_reasons.get(key, 0) + 1
    for reason, count in drop_reasons.items():
        print(f"    - {reason}: {count}")

    # Artifact removal
    # epochs = epochs.copy().apply_baseline(baseline=(-0.2, 0))
    if len(epochs.info['bads']) > 0:  # 나쁜 채널이 있는 경우에만 보간 수행
        epochs = epochs.copy().interpolate_bads()
    
    # Crop epochs to the desired range (0 to 2 seconds)
    epochs = epochs.crop(tmin=0, tmax=2)  # baseline 구간(-0.2~0) 제거 후 실제 분석 구간만 유지

    return epochs, original_channel_order

def concatEEG(file_path, channels, select_epoch, lower_bound_freq, upper_bound_freq):
    epochs_1, original_channel_order = mainEEG(file_path, channels, select_epoch, lower_bound_freq, upper_bound_freq)
    
    # nTrial을 지정하지 않고, 선택한 에폭의 수만큼 처리
    epochs = epochs_1
    
    return epochs, original_channel_order

def save_data(root_dir, target_dir, subject_filename, X, y):
    os.makedirs(os.path.join(root_dir, target_dir), exist_ok=True)  # target_dir이 존재하지 않으면 생성
    save_path = os.path.join(root_dir, target_dir, f"{subject_filename}.npz")
    np.savez(save_path, X=X, y=y)
    print(f"Data saved to {save_path}")

def process_data(root_dir, target_dir, file_path, select_epoch, channels, lower_bound_freq, upper_bound_freq, trials_per_trigger=4):
    # 원본 파일명에서 확장자를 제거하여 subject_filename 생성
    subject_filename = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(root_dir, target_dir, f"{subject_filename}.npz")
    
    # 이미 변환된 파일이 존재하는지 확인
    save_path = save_path.strip()
    print(f"** Processing file: {subject_filename}")
    
    try:
        epochs, original_channel_order = concatEEG(file_path, channels, select_epoch, lower_bound_freq, upper_bound_freq)
    except FileNotFoundError as e:
        print(e)
        print(f"Skipping file: {subject_filename} due to missing file.")
        return
    
    # Prepare data for saving
    X = epochs.get_data(copy=True)  # shape: (trials, channels, time)
    X = X[:, :, 1:]  # Remove the first time sample to keep only 5000 samples
    X = np.transpose(X, (2, 1, 0))  # Change shape to (time, channels, trials)
    
    # Verify channel order
    channel_order_after_transpose = original_channel_order  # Channels order should be unchanged
    assert original_channel_order == channel_order_after_transpose, "Channel order has changed!"
    
    y = np.array([epochs.events[i, -1] for i in range(len(epochs))])
    
    # Save data
    save_data(root_dir, target_dir, subject_filename, X, y)

# Main execution
root_dir = os.path.dirname(os.path.abspath(__file__))  # root directory
target_dir = os.path.join('EMG')  # save preprocessed data
os.makedirs(os.path.join(root_dir, target_dir), exist_ok=True)

select_epoch = ['1', '2', '3', '4', '5']

channels = list(range(65, 73))
lower_bound_freq = 100
upper_bound_freq = 300

# 지정된 폴더 내의 모든 XDF 파일을 처리
xdf_files = glob.glob(os.path.join(root_dir, f"*.xdf"))
for xdf_file in xdf_files:
    print("==================================================")
    process_data(root_dir, target_dir, xdf_file, select_epoch, channels, lower_bound_freq, upper_bound_freq)
    print("\n\n\n")
