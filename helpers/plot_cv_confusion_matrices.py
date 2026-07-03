import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from trainers.train_v5_braindecode_eeg_all_sessions import (  # noqa: E402
    EEGDataset,
    build_model,
    day_label_stratify_key,
    load_all_patient_days,
    one_vs_rest_labels,
    pair_indices,
    pair_labels,
    task_split_indices,
)

DEFAULT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
    / "sub2_yjkim"
    / "v5_sub2_eegnet_days_combined_lr5e-5"
)


def log(message):
    print(message, flush=True)


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ".":
        path = Path(*path.parts[1:])
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        return PROJECT_ROOT.parent / path
    return PROJECT_ROOT / path


def iter_run_dirs(root):
    if (root / "cross_validation.csv").exists() and (root / "run_config.json").exists():
        yield root
        return

    for csv_path in sorted(root.rglob("cross_validation.csv")):
        run_dir = csv_path.parent
        if (run_dir / "run_config.json").exists():
            yield run_dir


def load_run_args(run_dir):
    with (run_dir / "run_config.json").open(encoding="utf-8") as f:
        run_config = json.load(f)

    args = argparse.Namespace(**run_config["args"])
    args.data_dir = resolve_project_path(args.data_dir)
    args.input_file = getattr(args, "input_file", "X_eeg.npy")
    args.model = getattr(args, "model", "eegnet")
    return args


def predict_confusion(model, X, y, indices, batch_size, device):
    loader = DataLoader(EEGDataset(X, y, indices), batch_size=batch_size, shuffle=False)
    confusion = np.zeros((2, 2), dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch.to(device))
            pred = logits.argmax(1).cpu().numpy()
            true = y_batch.numpy()
            for true_label, pred_label in zip(true, pred):
                confusion[int(true_label), int(pred_label)] += 1

    return confusion


def load_model(checkpoint, args, n_channels, n_times, device):
    log(f"    loading checkpoint: {checkpoint.relative_to(checkpoint.parents[3])}")
    model = build_model(args.model, n_channels, n_times).to(device)
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    return model


def collect_confusions(run_dir):
    log(f"\nRun: {run_dir}")
    args = load_run_args(run_dir)
    log(f"  loading data: {args.data_dir} / {args.input_file}")
    X, original_y, day_ids, day_names = load_all_patient_days(args)
    n_trials, n_channels, n_times = X.shape
    all_idx = np.arange(n_trials)
    cv_stratify_key = day_label_stratify_key(day_ids, original_y)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    inner_val_ratio = args.val_test_ratio / (1 - 1 / 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  device={device} | X={X.shape} | days={', '.join(day_names)}")

    confusions = {"0": [], "2": [], "1_vs_3": []}
    for fold, (remaining_idx, fold_test_idx) in enumerate(
        splitter.split(all_idx, cv_stratify_key),
        start=1,
    ):
        log(f"  fold {fold}/5: building train-normalized test data")
        fold_train_idx, _ = train_test_split(
            remaining_idx,
            test_size=inner_val_ratio,
            stratify=cv_stratify_key[remaining_idx],
            random_state=args.seed + fold,
        )
        fold_mean = X[fold_train_idx].mean(axis=(0, 2))[:, None].astype(np.float32)
        fold_std = X[fold_train_idx].std(axis=(0, 2))[:, None].astype(np.float32)
        fold_std = np.maximum(fold_std, 1e-6)
        fold_X = ((X - fold_mean) / fold_std).astype(np.float32)

        for label in (0, 2):
            label_name = str(label)
            log(f"  fold {fold}/5 task {label_name}: predicting test split")
            y = one_vs_rest_labels(original_y, label)
            task_seed = args.seed + fold * 100 + label * 10
            task_test_idx = task_split_indices(
                original_y,
                fold_test_idx,
                label,
                False,
                task_seed + 2,
            )
            checkpoint = run_dir / "cross_validation" / f"fold_{fold}" / f"y_{label_name}" / "best_model.pt"
            model = load_model(checkpoint, args, n_channels, n_times, device)
            confusions[label_name].append(
                predict_confusion(model, fold_X, y, task_test_idx, args.batch_size, device)
            )
            del model

        pair_y = pair_labels(original_y, target_label=1, comparison_label=3)
        pair_test_idx = pair_indices(original_y, fold_test_idx, 1, 3)
        checkpoint = run_dir / "cross_validation" / f"fold_{fold}" / "y_1_vs_3" / "best_model.pt"
        log(f"  fold {fold}/5 task 1_vs_3: predicting test split")
        model = load_model(checkpoint, args, n_channels, n_times, device)
        confusions["1_vs_3"].append(
            predict_confusion(model, fold_X, pair_y, pair_test_idx, args.batch_size, device)
        )
        del model, fold_X

    return confusions


def plot_confusions(confusions, output_path):
    log(f"  plotting PNG: {output_path}")
    task_titles = {
        "0": "0 vs rest",
        "2": "2 vs rest",
        "1_vs_3": "1 vs 3",
    }
    class_labels = {
        "0": ("label 0", "rest"),
        "2": ("label 2", "rest"),
        "1_vs_3": ("label 1", "label 3"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)
    for ax, task in zip(axes, ["0", "2", "1_vs_3"]):
        matrix = np.mean(confusions[task], axis=0)
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"{task_titles[task]}\nmean of 5 folds")
        ax.set_xticks([0, 1], class_labels[task], rotation=30, ha="right")
        ax.set_yticks([0, 1], class_labels[task])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        threshold = matrix.max() / 2 if matrix.max() else 0
        for y_pos in range(2):
            for x_pos in range(2):
                value = matrix[y_pos, x_pos]
                color = "white" if value > threshold else "black"
                ax.text(
                    x_pos,
                    y_pos,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color=color,
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create test-set confusion matrix PNGs for v5 cross-validation runs."
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=DEFAULT_ROOT,
        help="Run directory, or a folder containing run directories.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    run_dirs = list(iter_run_dirs(root))
    if not run_dirs:
        raise FileNotFoundError(f"No cross-validation runs found under {root}")

    for run_dir in run_dirs:
        output_path = run_dir / "results" / "confusion_matrices.png"
        confusions = collect_confusions(run_dir)
        plot_confusions(confusions, output_path)
        log(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
