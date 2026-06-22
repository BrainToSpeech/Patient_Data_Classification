# Changes

- Simplified `Demo_binary/npz_to_eegnet_inputs.py` to only convert the already-epoched NPZ data into training NPY files.
- Removed unnecessary crop, sampling-rate, and metadata generation from the NPZ conversion script.
- Updated `Demo_binary/train_10_binary_eegnet_randomsplit.py` to use the full `X_eeg_raw.npy` input directly.
- Removed `--start-s`, `--end-s`, `preprocess_meta.json` loading, and time-axis slicing from the random-split EEGNet training script.
