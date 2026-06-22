"""Train and test five one-vs-rest LDA models with one shared stratified split."""

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import train_test_split

LABEL_NAMES = ["0", "1", "2", "3", "4"]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/processed/260602_sub1_hjlee_raw"


def make_lda_model():
    return LinearDiscriminantAnalysis(
        solver="lsqr",
        shrinkage="auto",
        priors=[0.5, 0.5],
    )


def evaluate(model, X, y, indices):
    X_split = X[indices]
    y_split = y[indices]
    pred = model.predict(X_split)
    prob = model.predict_proba(X_split)

    return (
        log_loss(y_split, prob, labels=[0, 1]),
        balanced_accuracy_score(y_split, pred),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--input-file", default="X_eeg_raw.npy")
    parser.add_argument("--checkpoint-root", type=Path, default=BASE_DIR / "checkpoints_lda_2s")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_test_ratio", type=float, default=0.2)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, default=None)
    args = parser.parse_args()

    run_dir = args.checkpoint_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)

    print(f"Loading {args.input_file}...")
    X = np.load(args.data_dir / args.input_file, mmap_mode="r")
    with (args.data_dir / "preprocess_meta.json").open() as f:
        meta = json.load(f)
    sfreq = float(meta["epoch"]["sfreq"])
    data_start_s = float(meta["epoch"]["crop_tmin"])
    data_end_s = data_start_s + (X.shape[-1] - 1) / sfreq
    end_s = data_end_s if args.end_s is None else args.end_s

    if not data_start_s <= args.start_s < end_s <= data_end_s:
        raise ValueError(f"Time range must satisfy {data_start_s} <= start-s < end-s <= {data_end_s}.")

    start_idx = int(round((args.start_s - data_start_s) * sfreq))
    end_idx = int(round((end_s - data_start_s) * sfreq)) + 1
    X = X[:, :, start_idx:end_idx]

    print(f"Model: LDA | X: {X.shape} | time: {args.start_s:g}-{end_s:g}s | run: {run_dir}")

    if not 0 < args.val_test_ratio < 0.5:
        raise ValueError("--val_test_ratio must be between 0 and 0.5.")

    n_trials = X.shape[0]
    original_y = np.load(args.data_dir / "y.npy").astype(np.int64)
    if len(original_y) != n_trials:
        raise ValueError(f"X and y.npy trial counts do not match: {n_trials} != {len(original_y)}")
    all_idx = np.arange(n_trials)

    train_idx, remaining_idx = train_test_split(
        all_idx,
        test_size=2 * args.val_test_ratio,
        stratify=original_y,
        random_state=args.seed,
    )

    val_idx, test_idx = train_test_split(
        remaining_idx,
        test_size=0.5,
        stratify=original_y[remaining_idx],
        random_state=args.seed,
    )
    split_sizes = [len(train_idx), len(val_idx), len(test_idx)]
    print(f"Data split: train={split_sizes[0]} val={split_sizes[1]} test={split_sizes[2]}")

    config_args = vars(args) | {
        "data_dir": str(args.data_dir),
        "checkpoint_root": str(args.checkpoint_root),
    }
    run_config = {
        "args": config_args,
        "model": "lda",
        "input_shape": list(X.shape),
        "sfreq": sfreq,
        "time_range_s": [args.start_s, end_s],
        "split_sizes": split_sizes,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    print("Calculating mean and std from the training set...")
    train_mean = X[train_idx].mean(axis=(0, 2))[:, None].astype(np.float32)
    train_std = X[train_idx].std(axis=(0, 2))[:, None].astype(np.float32)
    train_std = np.maximum(train_std, 1e-6)

    print("Normalizing and flattening epochs for LDA...")
    X = ((X - train_mean) / train_std).astype(np.float32).reshape(n_trials, -1)

    results = {}
    for label_name in LABEL_NAMES:
        class_id = int(label_name)
        print(f"\n=== one-vs-rest class {label_name} ===")
        model_dir = run_dir / f"y_{label_name}_ovr"
        model_dir.mkdir()
        checkpoint = model_dir / "model.pkl"

        y = (original_y == class_id).astype(np.int64)

        model = make_lda_model()
        model.fit(X[train_idx], y[train_idx])

        val_loss, val_acc = evaluate(model, X, y, val_idx)
        test_loss, test_acc = evaluate(model, X, y, test_idx)
        with checkpoint.open("wb") as f:
            pickle.dump(model, f)

        results[label_name] = {
            "best_epoch": None,
            "best_val_loss": val_loss,
            "best_val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        print(f"val={val_loss:.4f}/{val_acc:.2f} | test={test_loss:.4f}/{test_acc:.2f}")

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("\nTest accuracy")
    for name, result in results.items():
        print(f"y_{name}_ovr: {result['test_acc']:.3f}")
    print(f"\nSaved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
