"""Train and test one-vs-rest EEGNet models across all days for one patient."""

import argparse
import csv
from html import parser
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNet as BraindecodeEEGNet
from braindecode.models import FBCNet
from braindecode.models import ShallowFBCSPNet

from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
OVR_CHECKPOINT_NAMES = {
    0: "stop_vs_rest.pt",
    2: "help_vs_rest.pt",
    4: "toilet_vs_rest.pt",
}
PAIR_CHECKPOINT_NAME = "yes_no.pt"

# python trainers/train_v5_braindecode_eeg.py   --patient sub2_yjkim   --model eegnet
# python trainers/train_v5_braindecode_eeg.py   --patient sub2_yjkim   --model shallownet

def add_patient_day_args(parser):
    parser.add_argument("--patient", required=True)
    parser.add_argument("--day", default="all_days")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--checkpoint-root", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--run-name", default=None)


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def apply_patient_day_paths(args, cli_args):
    args.patient = cli_args.patient
    args.day = cli_args.day
    args.data_dir = resolve_project_path(cli_args.data_root) / cli_args.patient
    args.input_file = cli_args.input_file or getattr(args, "input_file", "X_eeg.npy")
    args.checkpoint_root = (
        resolve_project_path(cli_args.checkpoint_root)
        / f"{cli_args.patient}_nogit"
    )
    if cli_args.run_name:
        args.checkpoint_root = args.checkpoint_root / cli_args.run_name


class EEGDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X, self.y, self.indices = X, y, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return torch.from_numpy(self.X[idx]), int(self.y[idx])


def build_model(model_name, n_channels, n_times, sfreq=None):
    if model_name == "eegnet":
        return BraindecodeEEGNet(
            n_chans=n_channels,
            n_outputs=2,
            n_times=n_times,
            final_conv_length="auto",
            sfreq=sfreq
        )
    if model_name == "shallownet":
        return ShallowFBCSPNet(
            n_chans=n_channels,
            n_outputs=2,
            n_times=n_times,
            final_conv_length="auto",
            sfreq=sfreq
        )
    if model_name == "fbcnet":
        return FBCNet(
            n_chans=n_channels,
            n_outputs=2,
            n_times=n_times,
            sfreq=sfreq
        )
    raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    class_loss = torch.zeros(2, device=device)
    class_correct, class_total = torch.zeros(2, device=device), torch.zeros(2, device=device)
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        pred = logits.argmax(1)
        losses = nn.functional.cross_entropy(logits, y, reduction="none")
        total_loss += losses.sum().item()
        correct += (pred == y).sum().item()
        total += len(y)
        class_total += torch.bincount(y, minlength=2)
        class_correct += torch.bincount(y[pred == y], minlength=2)
        for class_id in range(2):
            class_loss[class_id] += losses[y == class_id].sum()
    balanced_loss = (class_loss / class_total).mean().item()
    balanced_acc = (class_correct / class_total).mean().item()
    accuracy = correct / total
    return total_loss / total, balanced_loss, balanced_acc, accuracy


def format_class_counts(y, indices):
    counts = np.bincount(y[indices], minlength=2)
    return f"class0={counts[0]} class1={counts[1]} total={counts.sum()}"


def format_original_label_counts(y, indices):
    counts = np.bincount(y[indices], minlength=5)
    labels = " ".join(f"label{i}={count}" for i, count in enumerate(counts))
    return f"{labels} total={counts.sum()}"


def format_day_counts(day_ids, day_names, indices):
    counts = np.bincount(day_ids[indices], minlength=len(day_names))
    days = " ".join(f"{day}={counts[i]}" for i, day in enumerate(day_names))
    return f"{days} total={counts.sum()}"


def day_label_stratify_key(day_ids, original_y):
    return np.array([f"{day_id}_{label}" for day_id, label in zip(day_ids, original_y)])


def load_all_patient_days(args):
    patient_dir = args.data_dir
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient data directory not found: {patient_dir}")

    day_dirs = [
        day_dir
        for day_dir in sorted(patient_dir.iterdir())
        if day_dir.is_dir()
        and (day_dir / args.input_file).exists()
        and (day_dir / "y.npy").exists()
    ]
    if not day_dirs:
        raise FileNotFoundError(
            f"No day folders with {args.input_file} and y.npy found under {patient_dir}"
        )

    X_parts, y_parts, day_id_parts, day_names = [], [], [], []
    expected_shape = None
    for day_id, day_dir in enumerate(day_dirs):
        X_day = np.load(day_dir / args.input_file).astype(np.float32)
        y_day = np.load(day_dir / "y.npy").astype(np.int64)
        if len(y_day) != X_day.shape[0]:
            raise ValueError(
                f"{day_dir.name}: X and y.npy trial counts do not match: "
                f"{X_day.shape[0]} != {len(y_day)}"
            )
        if expected_shape is None:
            expected_shape = X_day.shape[1:]
        elif X_day.shape[1:] != expected_shape:
            raise ValueError(
                f"{day_dir.name}: expected input shape (*, {expected_shape}), "
                f"got {X_day.shape}"
            )
        X_parts.append(X_day)
        y_parts.append(y_day)
        day_id_parts.append(np.full(len(y_day), day_id, dtype=np.int64))
        day_names.append(day_dir.name)

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(day_id_parts, axis=0),
        day_names,
    )


def split_train_val_test_by_day(original_y, day_ids, day_names, val_test_ratio, seed):
    train_parts, val_parts, test_parts = [], [], []
    for day_id, day_name in enumerate(day_names):
        day_idx = np.flatnonzero(day_ids == day_id)
        day_train_idx, day_remaining_idx = train_test_split(
            day_idx,
            test_size=2 * val_test_ratio,
            stratify=original_y[day_idx],
            random_state=seed,
        )
        day_val_idx, day_test_idx = train_test_split(
            day_remaining_idx,
            test_size=0.5,
            stratify=original_y[day_remaining_idx],
            random_state=seed,
        )
        train_parts.append(day_train_idx)
        val_parts.append(day_val_idx)
        test_parts.append(day_test_idx)

    rng = np.random.default_rng(seed)
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def one_vs_rest_labels(original_y, label):
    return (original_y != label).astype(np.int64)


def pair_labels(original_y, target_label, comparison_label):
    return np.where(
        original_y == target_label,
        0,
        np.where(original_y == comparison_label, 1, -1),
    ).astype(np.int64)


def pair_indices(original_y, indices, target_label, comparison_label):
    keep = np.isin(original_y[indices], [target_label, comparison_label])
    return indices[keep]


def balance_one_vs_rest_indices(original_y, indices, target_label, seed):
    target_indices = indices[original_y[indices] == target_label]
    rest_labels = np.unique(original_y[indices][original_y[indices] != target_label])
    if len(target_indices) == 0 or len(rest_labels) == 0:
        raise ValueError(f"Cannot balance one-vs-rest task for label {target_label}.")

    per_label, remainder = divmod(len(target_indices), len(rest_labels))
    rng = np.random.default_rng(seed)
    selected = [target_indices]
    for position, rest_label in enumerate(rest_labels):
        count = per_label + (position < remainder)
        candidates = indices[original_y[indices] == rest_label]
        if len(candidates) < count:
            raise ValueError(
                f"Label {rest_label} has {len(candidates)} trials, but {count} are needed "
                f"to balance target label {target_label}."
            )
        selected.append(rng.choice(candidates, size=count, replace=False))

    balanced_indices = np.concatenate(selected)
    rng.shuffle(balanced_indices)
    return balanced_indices


def task_split_indices(original_y, indices, target_label, balance_rest, seed):
    if not balance_rest:
        return indices
    return balance_one_vs_rest_indices(original_y, indices, target_label, seed)


def write_log(log_path, message=""):
    print(message)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{message}\n")


def display_path(path):
    path = Path(path)
    try:
        return f"./{path.relative_to(Path.cwd())}"
    except ValueError:
        return str(path)


def load_config(config_path):
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    config.setdefault("window_size", None)
    config.setdefault("window_stride", None)
    config.setdefault("run_window_diagnostic", False)
    config.setdefault("balance_rest_to_target", False)
    config.setdefault("run_5fold_cross_validation", False)
    config.setdefault("selection_metric", "balanced_accuracy")

    if config["selection_metric"] not in {"balanced_accuracy", "balanced_loss"}:
        raise ValueError("selection_metric must be 'balanced_accuracy' or 'balanced_loss'.")

    for key in ["data_dir", "checkpoint_root"]:
        if key in config:
            path = Path(config[key])
            config[key] = path if path.is_absolute() else BASE_DIR / path

    return argparse.Namespace(**config)


def train_task(
    X,
    y,
    train_idx,
    val_idx,
    test_idx,
    n_channels,
    args,
    device,
    checkpoint,
    log_path,
    log_prefix="",
):
    train_counts = np.bincount(y[train_idx], minlength=2)
    if np.any(train_counts == 0):
        raise ValueError("Training split is missing one binary class.")
    sample_weights = 1.0 / train_counts[y[train_idx]]

    if args.weighted_sampler:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_idx),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = DataLoader(
            EEGDataset(X, y, train_idx),
            batch_size=args.batch_size,
            sampler=sampler,
        )
    else:
        sampler = None
        train_loader = DataLoader(
            EEGDataset(X, y, train_idx),
            batch_size=args.batch_size,
            shuffle=True,
        )

    train_eval_loader = DataLoader(
        EEGDataset(X, y, train_idx),
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_loader = DataLoader(
        EEGDataset(X, y, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        EEGDataset(X, y, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
    )

    torch.manual_seed(args.seed)
    model = build_model(args.model, n_channels, X.shape[-1], args.sfreq).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_loss = best_val_bal_loss = float("inf")
    best_val_bal_acc = best_val_acc = best_epoch = waiting = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        train_class_correct = torch.zeros(2, device=device)
        train_class_total = torch.zeros(2, device=device)
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            pred = logits.argmax(1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_class_total += torch.bincount(y_batch, minlength=2)
            train_class_correct += torch.bincount(y_batch[pred == y_batch], minlength=2)

        train_loss /= len(train_idx)
        train_bal_acc = (train_class_correct / train_class_total).mean().item()
        train_acc = (train_class_correct.sum() / train_class_total.sum()).item()
        val_loss, val_bal_loss, val_bal_acc, val_acc = evaluate(model, val_loader, criterion, device)
        if epoch == 1 or epoch % 10 == 0:
            write_log(
                log_path,
                f"{log_prefix}epoch {epoch:03d} | "
                f"train loss: {train_loss:.4f} bal_acc: {train_bal_acc:.2f} acc: {train_acc:.2f} | "
                f"val loss: {val_loss:.4f} bal_loss: {val_bal_loss:.4f} "
                f"bal_acc: {val_bal_acc:.2f} acc: {val_acc:.2f}",
            )

        improved = (
            val_bal_acc > best_val_bal_acc
            if args.selection_metric == "balanced_accuracy"
            else val_bal_loss < best_val_bal_loss
        )
        if epoch == 1 or improved:
            best_val_loss = val_loss
            best_val_bal_loss = val_bal_loss
            best_val_bal_acc = val_bal_acc
            best_val_acc = val_acc
            best_epoch, waiting = epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            waiting += 1
            if waiting >= args.patience:
                break

    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    train_loss, train_bal_loss, train_bal_acc, train_acc = evaluate(
        model, train_eval_loader, criterion, device
    )
    test_loss, test_bal_loss, test_bal_acc, test_acc = evaluate(
        model, test_loader, criterion, device
    )
    result = {
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "train_loss": train_loss,
        "train_bal_loss": train_bal_loss,
        "train_bal_acc": train_bal_acc,
        "train_acc": train_acc,
        "best_val_loss": best_val_loss,
        "best_val_bal_loss": best_val_bal_loss,
        "best_val_bal_acc": best_val_bal_acc,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_bal_loss": test_bal_loss,
        "test_bal_acc": test_bal_acc,
        "test_acc": test_acc,
    }

    del model, optimizer, train_loader, train_eval_loader, val_loader, test_loader, sampler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    
    ########################
    ###### 1. Setting ######
    ########################

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "train_config_v3-6.json")
    parser.add_argument("--model", choices=["eegnet", "shallownet", "fbcnet"], default="eegnet")
    add_patient_day_args(parser)
    cli_args = parser.parse_args()
    args = load_config(cli_args.config)
    args.config = cli_args.config
    args.model = cli_args.model
    apply_patient_day_paths(args, cli_args)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.checkpoint_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    log_path = run_dir / "train.log"

    write_log(log_path, "Training configuration")
    write_log(log_path, f"  config={display_path(args.config)}")
    write_log(log_path, f"  model={args.model}")
    write_log(log_path, f"  patient={args.patient}")
    write_log(log_path, f"  day={args.day}")
    write_log(log_path, f"  patient_dir={display_path(args.data_dir)}")
    write_log(log_path, f"  input_file={args.input_file}")
    write_log(log_path, f"  checkpoint_root={display_path(args.checkpoint_root)}")
    write_log(log_path, f"  epochs={args.epochs}")
    write_log(log_path, f"  batch_size={args.batch_size}")
    write_log(log_path, f"  lr={args.lr}")
    write_log(log_path, f"  weight_decay={args.weight_decay}")
    write_log(log_path, f"  patience={args.patience}")
    write_log(log_path, f"  seed={args.seed}")
    write_log(log_path, f"  val_test_ratio={args.val_test_ratio}")
    write_log(log_path, f"  weighted_sampler={args.weighted_sampler}")
    write_log(log_path, f"  run_window_diagnostic={args.run_window_diagnostic}")
    write_log(log_path, f"  window_size={args.window_size}")
    write_log(log_path, f"  window_stride={args.window_stride}")
    write_log(log_path, f"  balance_rest_to_target={args.balance_rest_to_target}")
    write_log(log_path, f"  run_5fold_cross_validation={args.run_5fold_cross_validation}")
    write_log(log_path, f"  selection_metric={args.selection_metric}")

    #############################
    ###### 2. Data Loading ######
    #############################

    write_log(log_path, f"\nLoading all days from {display_path(args.data_dir)}...")
    X, original_y, day_ids, day_names = load_all_patient_days(args)
    if X.shape[-1] < 32:
        raise ValueError("Input time dimension is too short for EEGNet pooling.")

    write_log(log_path, f"Loaded days: {', '.join(day_names)}")
    write_log(log_path, f"Device: {device} | X: {X.shape} | run: {display_path(run_dir)}")

    if not 0 < args.val_test_ratio < 0.5:
        raise ValueError("--val_test_ratio must be between 0 and 0.5.")

    n_trials, n_channels, _ = X.shape
    if len(original_y) != n_trials:
        raise ValueError(f"X and y.npy trial counts do not match: {n_trials} != {len(original_y)}")
    task_labels = np.unique(original_y)
    required_labels = {0, 1, 2, 3, 4}
    missing_labels = required_labels - set(task_labels.tolist())
    if missing_labels:
        raise ValueError(f"Required labels are missing from y.npy: {sorted(missing_labels)}")
    ovr_labels = np.array([0, 2, 4], dtype=np.int64)
    print("type of label: ", task_labels)

    ############################################
    ###### 3. Train/validation/test Split ######
    ############################################

    all_idx = np.arange(n_trials)
    train_idx, val_idx, test_idx = split_train_val_test_by_day(
        original_y,
        day_ids,
        day_names,
        args.val_test_ratio,
        args.seed,
    )
    split_sizes = [len(train_idx), len(val_idx), len(test_idx)]
    write_log(log_path, f"Data split: train={split_sizes[0]} val={split_sizes[1]} test={split_sizes[2]}")
    write_log(log_path, "Day distribution:")
    write_log(log_path, f"  train: {format_day_counts(day_ids, day_names, train_idx)}")
    write_log(log_path, f"  val:   {format_day_counts(day_ids, day_names, val_idx)}")
    write_log(log_path, f"  test:  {format_day_counts(day_ids, day_names, test_idx)}")
    write_log(log_path, "Original label distribution:")
    write_log(log_path, f"  train: {format_original_label_counts(original_y, train_idx)}")
    write_log(log_path, f"  val:   {format_original_label_counts(original_y, val_idx)}")
    write_log(log_path, f"  test:  {format_original_label_counts(original_y, test_idx)}")

    config_args = vars(args) | {
        "config": display_path(args.config),
        "data_dir": display_path(args.data_dir),
        "checkpoint_root": display_path(args.checkpoint_root),
    }
    run_config = {
        "args": config_args,
        "input_shape": list(X.shape),
        "days": day_names,
        "split_sizes": split_sizes,
        "split_day_counts": {
            "train": np.bincount(day_ids[train_idx], minlength=len(day_names)).tolist(),
            "val": np.bincount(day_ids[val_idx], minlength=len(day_names)).tolist(),
            "test": np.bincount(day_ids[test_idx], minlength=len(day_names)).tolist(),
        },
        "tasks": ["0_vs_rest", "1_vs_3", "2_vs_rest", "4_vs_rest"],
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    ##############################
    ###### 4. Normalization ######
    ##############################

    if not args.run_5fold_cross_validation:
        write_log(log_path, "Normalizing X using training-set mean and std...")
        train_mean = X[train_idx].mean(axis=(0, 2))[:, None].astype(np.float32)
        train_std = X[train_idx].std(axis=(0, 2))[:, None].astype(np.float32)
        train_std = np.maximum(train_std, 1e-6)
        X = ((X - train_mean) / train_std).astype(np.float32)

    results = {}
    regular_task_labels = [] if args.run_5fold_cross_validation else ovr_labels
    for label in regular_task_labels:
        label_name = str(int(label))
        write_log(log_path, f"\n=== y_{label_name} ===")
        model_dir = run_dir / f"y_{label_name}"
        model_dir.mkdir()
        checkpoint = model_dir / OVR_CHECKPOINT_NAMES[int(label)]

        y = one_vs_rest_labels(original_y, label)
        task_seed = args.seed + int(label) * 10
        task_train_idx = task_split_indices(
            original_y,
            train_idx,
            label,
            args.balance_rest_to_target,
            task_seed,
        )
        task_val_idx = task_split_indices(
            original_y,
            val_idx,
            label,
            False,
            task_seed + 1,
        )
        task_test_idx = task_split_indices(
            original_y,
            test_idx,
            label,
            False,
            task_seed + 2,
        )

        train_counts = np.bincount(y[task_train_idx], minlength=2)
        if np.any(train_counts == 0):
            raise ValueError(f"Task y_{label_name} is missing one class in the training split.")
        write_log(log_path, "Binary label distribution:")
        write_log(log_path, f"  train: {format_class_counts(y, task_train_idx)}")
        write_log(log_path, f"  val:   {format_class_counts(y, task_val_idx)}")
        write_log(log_path, f"  test:  {format_class_counts(y, task_test_idx)}")
        write_log(log_path, "Original labels in this split:")
        write_log(log_path, f"  train: {format_original_label_counts(original_y, task_train_idx)}")
        write_log(log_path, f"  val:   {format_original_label_counts(original_y, task_val_idx)}")
        write_log(log_path, f"  test:  {format_original_label_counts(original_y, task_test_idx)}")

        results[label_name] = train_task(
            X,
            y,
            task_train_idx,
            task_val_idx,
            task_test_idx,
            n_channels,
            args,
            device,
            checkpoint,
            log_path,
        )
        result = results[label_name]
        write_log(
            log_path,
            f"best epoch={result['best_epoch']} | "
            f"train loss: {result['train_loss']:.4f} bal_acc: {result['train_bal_acc']:.2f} "
            f"acc: {result['train_acc']:.2f} | "
            f"val loss: {result['best_val_loss']:.4f} "
            f"bal_loss: {result['best_val_bal_loss']:.4f} "
            f"bal_acc: {result['best_val_bal_acc']:.2f} "
            f"acc: {result['best_val_acc']:.2f} | "
            f"test loss: {result['test_loss']:.4f} bal_acc: {result['test_bal_acc']:.2f} "
            f"acc: {result['test_acc']:.2f}",
        )

    if not args.run_5fold_cross_validation:
        write_log(log_path, "\n=== y_1_vs_3 ===")
        pair_y = pair_labels(original_y, target_label=1, comparison_label=3)
        pair_train_idx = pair_indices(original_y, train_idx, 1, 3)
        pair_val_idx = pair_indices(original_y, val_idx, 1, 3)
        pair_test_idx = pair_indices(original_y, test_idx, 1, 3)
        pair_model_dir = run_dir / "y_1_vs_3"
        pair_model_dir.mkdir()

        write_log(log_path, "Binary labels: class0=label1 class1=label3")
        write_log(log_path, f"  train: {format_class_counts(pair_y, pair_train_idx)}")
        write_log(log_path, f"  val:   {format_class_counts(pair_y, pair_val_idx)}")
        write_log(log_path, f"  test:  {format_class_counts(pair_y, pair_test_idx)}")
        results["1_vs_3"] = train_task(
            X,
            pair_y,
            pair_train_idx,
            pair_val_idx,
            pair_test_idx,
            n_channels,
            args,
            device,
            pair_model_dir / PAIR_CHECKPOINT_NAME,
            log_path,
            log_prefix="1_vs_3 | ",
        )
        pair_result = results["1_vs_3"]
        write_log(
            log_path,
            f"1_vs_3 best epoch={pair_result['best_epoch']} | "
            f"train loss: {pair_result['train_loss']:.4f} "
            f"bal_acc: {pair_result['train_bal_acc']:.2f} acc: {pair_result['train_acc']:.2f} | "
            f"val loss: {pair_result['best_val_loss']:.4f} "
            f"bal_loss: {pair_result['best_val_bal_loss']:.4f} "
            f"bal_acc: {pair_result['best_val_bal_acc']:.2f} acc: {pair_result['best_val_acc']:.2f} | "
            f"test loss: {pair_result['test_loss']:.4f} "
            f"bal_acc: {pair_result['test_bal_acc']:.2f} acc: {pair_result['test_acc']:.2f}",
        )

    if not args.run_5fold_cross_validation:
        (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    if args.run_window_diagnostic and not args.run_5fold_cross_validation:
        if args.window_size is None or args.window_stride is None:
            raise ValueError("window_size and window_stride are required when run_window_diagnostic is true.")
        if args.window_size < 32:
            raise ValueError("window_size must be at least 32 for EEGNet pooling.")
        if args.window_size > X.shape[-1]:
            raise ValueError("window_size cannot exceed the input time dimension.")
        if args.window_stride <= 0:
            raise ValueError("window_stride must be positive.")

        window_starts = range(0, X.shape[-1] - args.window_size + 1, args.window_stride)
        diagnostic_dir = run_dir / "window_diagnostic"
        diagnostic_dir.mkdir()
        csv_path = run_dir / "window_diagnostic.csv"
        csv_columns = [
            "label",
            "window_start",
            "window_end",
            "best_epoch",
            "best_val_bal_loss",
            "best_val_bal_acc",
            "test_bal_acc",
            "test_acc",
        ]

        write_log(log_path, "\n=== Sliding-window diagnostic ===")
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            writer.writeheader()

            for label in ovr_labels:
                label_name = str(int(label))
                y = one_vs_rest_labels(original_y, label)
                task_seed = args.seed + int(label) * 10
                task_train_idx = task_split_indices(
                    original_y,
                    train_idx,
                    label,
                    args.balance_rest_to_target,
                    task_seed,
                )
                task_val_idx = task_split_indices(
                    original_y,
                    val_idx,
                    label,
                    False,
                    task_seed + 1,
                )
                task_test_idx = task_split_indices(
                    original_y,
                    test_idx,
                    label,
                    False,
                    task_seed + 2,
                )
                for window_start in window_starts:
                    window_end = window_start + args.window_size
                    window_name = f"{window_start}_{window_end}"
                    window_dir = diagnostic_dir / f"y_{label_name}" / window_name
                    window_dir.mkdir(parents=True)
                    checkpoint = window_dir / OVR_CHECKPOINT_NAMES[int(label)]
                    X_window = X[:, :, window_start:window_end]

                    write_log(log_path, f"\nlabel={label_name} window=[{window_start}:{window_end}]")
                    result = train_task(
                        X_window,
                        y,
                        task_train_idx,
                        task_val_idx,
                        task_test_idx,
                        n_channels,
                        args,
                        device,
                        checkpoint,
                        log_path,
                        log_prefix=f"label={label_name} window=[{window_start}:{window_end}] | ",
                    )
                    write_log(
                        log_path,
                        f"label={label_name} window=[{window_start}:{window_end}] | "
                        f"best epoch={result['best_epoch']} | "
                        f"train loss: {result['train_loss']:.4f} "
                        f"bal_acc: {result['train_bal_acc']:.2f} acc: {result['train_acc']:.2f} | "
                        f"val loss: {result['best_val_loss']:.4f} "
                        f"bal_loss: {result['best_val_bal_loss']:.4f} "
                        f"bal_acc: {result['best_val_bal_acc']:.2f} acc: {result['best_val_acc']:.2f} | "
                        f"test loss: {result['test_loss']:.4f} "
                        f"bal_acc: {result['test_bal_acc']:.2f} acc: {result['test_acc']:.2f}",
                    )
                    writer.writerow(
                        {
                            "label": label_name,
                            "window_start": window_start,
                            "window_end": window_end,
                            "best_epoch": result["best_epoch"],
                            "best_val_bal_loss": result["best_val_bal_loss"],
                            "best_val_bal_acc": result["best_val_bal_acc"],
                            "test_bal_acc": result["test_bal_acc"],
                            "test_acc": result["test_acc"],
                        }
                    )
                    csv_file.flush()

        write_log(log_path, f"Window diagnostic CSV: {display_path(csv_path)}")

    if args.run_5fold_cross_validation:
        n_folds = 5
        inner_val_ratio = args.val_test_ratio / (1 - 1 / n_folds)
        if not 0 < inner_val_ratio < 1:
            raise ValueError("val_test_ratio is incompatible with 5-fold cross-validation.")

        write_log(log_path, "\n=== 5-fold cross-validation ===")
        cv_dir = run_dir / "cross_validation"
        cv_dir.mkdir()
        cv_csv_path = run_dir / "cross_validation.csv"
        cv_columns = [
            "fold",
            "label",
            "best_epoch",
            "best_val_bal_loss",
            "best_val_bal_acc",
            "test_loss",
            "test_bal_loss",
            "test_bal_acc",
            "test_acc",
        ]
        cv_results = {}
        cv_source = X
        cv_stratify_key = day_label_stratify_key(day_ids, original_y)
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

        with cv_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=cv_columns)
            writer.writeheader()

            for fold, (remaining_idx, fold_test_idx) in enumerate(
                splitter.split(all_idx, cv_stratify_key),
                start=1,
            ):
                fold_train_idx, fold_val_idx = train_test_split(
                    remaining_idx,
                    test_size=inner_val_ratio,
                    stratify=cv_stratify_key[remaining_idx],
                    random_state=args.seed + fold,
                )
                fold_dir = cv_dir / f"fold_{fold}"
                fold_dir.mkdir()
                fold_results = {}

                write_log(
                    log_path,
                    f"\n--- fold {fold}/{n_folds}: "
                    f"train={len(fold_train_idx)} val={len(fold_val_idx)} "
                    f"test={len(fold_test_idx)} ---",
                )
                write_log(log_path, "Day distribution:")
                write_log(log_path, f"  train: {format_day_counts(day_ids, day_names, fold_train_idx)}")
                write_log(log_path, f"  val:   {format_day_counts(day_ids, day_names, fold_val_idx)}")
                write_log(log_path, f"  test:  {format_day_counts(day_ids, day_names, fold_test_idx)}")
                fold_mean = cv_source[fold_train_idx].mean(axis=(0, 2))[:, None].astype(np.float32)
                fold_std = cv_source[fold_train_idx].std(axis=(0, 2))[:, None].astype(np.float32)
                fold_std = np.maximum(fold_std, 1e-6)
                fold_X = ((cv_source - fold_mean) / fold_std).astype(np.float32)

                for label in ovr_labels:
                    label_name = str(int(label))
                    y = one_vs_rest_labels(original_y, label)
                    task_seed = args.seed + fold * 100 + int(label) * 10
                    task_train_idx = task_split_indices(
                        original_y,
                        fold_train_idx,
                        label,
                        args.balance_rest_to_target,
                        task_seed,
                    )
                    task_val_idx = task_split_indices(
                        original_y,
                        fold_val_idx,
                        label,
                        False,
                        task_seed + 1,
                    )
                    task_test_idx = task_split_indices(
                        original_y,
                        fold_test_idx,
                        label,
                        False,
                        task_seed + 2,
                    )
                    label_dir = fold_dir / f"y_{label_name}"
                    label_dir.mkdir()

                    write_log(log_path, f"\nfold={fold} label={label_name}")
                    write_log(log_path, "Binary label distribution:")
                    write_log(log_path, f"  train: {format_class_counts(y, task_train_idx)}")
                    write_log(log_path, f"  val:   {format_class_counts(y, task_val_idx)}")
                    write_log(log_path, f"  test:  {format_class_counts(y, task_test_idx)}")
                    write_log(log_path, "Original labels in this split:")
                    write_log(
                        log_path,
                        f"  train: {format_original_label_counts(original_y, task_train_idx)}",
                    )
                    write_log(
                        log_path,
                        f"  val:   {format_original_label_counts(original_y, task_val_idx)}",
                    )
                    write_log(
                        log_path,
                        f"  test:  {format_original_label_counts(original_y, task_test_idx)}",
                    )
                    result = train_task(
                        fold_X,
                        y,
                        task_train_idx,
                        task_val_idx,
                        task_test_idx,
                        n_channels,
                        args,
                        device,
                        label_dir / OVR_CHECKPOINT_NAMES[int(label)],
                        log_path,
                        log_prefix=f"fold={fold} label={label_name} | ",
                    )
                    fold_results[label_name] = result
                    writer.writerow(
                        {
                            "fold": fold,
                            "label": label_name,
                            "best_epoch": result["best_epoch"],
                            "best_val_bal_loss": result["best_val_bal_loss"],
                            "best_val_bal_acc": result["best_val_bal_acc"],
                            "test_loss": result["test_loss"],
                            "test_bal_loss": result["test_bal_loss"],
                            "test_bal_acc": result["test_bal_acc"],
                            "test_acc": result["test_acc"],
                        }
                    )

                pair_y = pair_labels(original_y, target_label=1, comparison_label=3)
                pair_train_idx = pair_indices(original_y, fold_train_idx, 1, 3)
                pair_val_idx = pair_indices(original_y, fold_val_idx, 1, 3)
                pair_test_idx = pair_indices(original_y, fold_test_idx, 1, 3)
                pair_dir = fold_dir / "y_1_vs_3"
                pair_dir.mkdir()
                write_log(log_path, f"\nfold={fold} label=1_vs_3")
                write_log(log_path, "Binary labels: class0=label1 class1=label3")
                write_log(log_path, "Binary label distribution:")
                write_log(log_path, f"  train: {format_class_counts(pair_y, pair_train_idx)}")
                write_log(log_path, f"  val:   {format_class_counts(pair_y, pair_val_idx)}")
                write_log(log_path, f"  test:  {format_class_counts(pair_y, pair_test_idx)}")
                write_log(log_path, "Original labels in this split:")
                write_log(
                    log_path,
                    f"  train: {format_original_label_counts(original_y, pair_train_idx)}",
                )
                write_log(
                    log_path,
                    f"  val:   {format_original_label_counts(original_y, pair_val_idx)}",
                )
                write_log(
                    log_path,
                    f"  test:  {format_original_label_counts(original_y, pair_test_idx)}",
                )
                pair_result = train_task(
                    fold_X,
                    pair_y,
                    pair_train_idx,
                    pair_val_idx,
                    pair_test_idx,
                    n_channels,
                    args,
                    device,
                    pair_dir / PAIR_CHECKPOINT_NAME,
                    log_path,
                    log_prefix=f"fold={fold} label=1_vs_3 | ",
                )
                fold_results["1_vs_3"] = pair_result
                writer.writerow(
                    {
                        "fold": fold,
                        "label": "1_vs_3",
                        "best_epoch": pair_result["best_epoch"],
                        "best_val_bal_loss": pair_result["best_val_bal_loss"],
                        "best_val_bal_acc": pair_result["best_val_bal_acc"],
                        "test_loss": pair_result["test_loss"],
                        "test_bal_loss": pair_result["test_bal_loss"],
                        "test_bal_acc": pair_result["test_bal_acc"],
                        "test_acc": pair_result["test_acc"],
                    }
                )
                csv_file.flush()

                cv_results[str(fold)] = fold_results
                (fold_dir / "results.json").write_text(json.dumps(fold_results, indent=2))
                del fold_X

        cv_task_names = ["0", "2", "4", "1_vs_3"]
        cv_summary = {}
        for label_name in cv_task_names:
            label_results = [cv_results[str(fold)][label_name] for fold in range(1, n_folds + 1)]
            cv_summary[label_name] = {
                "mean_test_bal_loss": float(np.mean([result["test_bal_loss"] for result in label_results])),
                "std_test_bal_loss": float(np.std([result["test_bal_loss"] for result in label_results])),
                "mean_test_bal_acc": float(np.mean([result["test_bal_acc"] for result in label_results])),
                "std_test_bal_acc": float(np.std([result["test_bal_acc"] for result in label_results])),
                "mean_test_acc": float(np.mean([result["test_acc"] for result in label_results])),
                "std_test_acc": float(np.std([result["test_acc"] for result in label_results])),
            }

        (run_dir / "cross_validation_results.json").write_text(
            json.dumps({"folds": cv_results, "summary": cv_summary}, indent=2)
        )
        write_log(log_path, "\n5-fold cross-validation summary")
        for label_name, summary in cv_summary.items():
            write_log(
                log_path,
                f"y_{label_name}: test bal_loss={summary['mean_test_bal_loss']:.3f} "
                f"+/- {summary['std_test_bal_loss']:.3f} | "
                f"bal_acc={summary['mean_test_bal_acc']:.3f} "
                f"+/- {summary['std_test_bal_acc']:.3f} | "
                f"test acc={summary['mean_test_acc']:.3f} +/- {summary['std_test_acc']:.3f}",
            )
        write_log(log_path, f"Cross-validation CSV: {display_path(cv_csv_path)}")

    if not args.run_5fold_cross_validation:
        write_log(log_path, "\nTest accuracy")
        for name, result in results.items():
            write_log(log_path, f"y_{name}: {result['test_acc']:.3f}")
    write_log(log_path, f"\nSaved to: {display_path(run_dir)}")


if __name__ == "__main__":
    main()
