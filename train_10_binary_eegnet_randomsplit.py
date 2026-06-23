"""Train and test ten binary EEGNet models with one shared stratified split."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Binary task names. "01" means original labels 0/1 vs labels 2/3/4.
LABEL_NAMES = ["01", "02", "03", "04", "12", "13", "14", "23", "24", "34"]

# Folder that contains this script.
BASE_DIR = Path(__file__).resolve().parent


class EEGDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X, self.y, self.indices = X, y, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return torch.from_numpy(self.X[idx]), int(self.y[idx])


class EEGNet(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, (1, 65), padding="same", bias=False),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25),
            nn.Conv2d(16, 16, (1, 17), padding="same", groups=16, bias=False),
            nn.Conv2d(16, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, x):
        return self.classifier(self.features(x.unsqueeze(1)).flatten(1))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    class_correct, class_total = torch.zeros(2, device=device), torch.zeros(2, device=device)
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        pred = logits.argmax(1)
        total_loss += criterion(logits, y).item() * len(y)
        correct += (pred == y).sum().item()
        total += len(y)
        class_total += torch.bincount(y, minlength=2)
        class_correct += torch.bincount(y[pred == y], minlength=2)
    balanced_acc = (class_correct / class_total).mean().item()
    accuracy = correct / total
    return total_loss / total, balanced_acc, accuracy


def format_class_counts(y, indices):
    counts = np.bincount(y[indices], minlength=2)
    return f"class0={counts[0]} class1={counts[1]} total={counts.sum()}"


def format_original_label_counts(y, indices):
    counts = np.bincount(y[indices], minlength=5)
    labels = " ".join(f"label{i}={count}" for i, count in enumerate(counts))
    return f"{labels} total={counts.sum()}"

def balanced_task_indices(original_y, base_idx, pair_labels, pair_n, other_n, seed):
    rng = np.random.default_rng(seed)
    selected = []

    for label in range(5):
        # pair labels are the two labels mapped to binary class 0.
        n = pair_n if label in pair_labels else other_n
        candidates = base_idx[original_y[base_idx] == label]
        selected.append(rng.choice(candidates, size=n, replace=False))

    return np.concatenate(selected)

def write_log(log_path, message=""):
    print(message)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{message}\n")


def display_path(path):
    path = Path(path)
    try:
        # Keep logs readable by hiding the full /home/... prefix.
        return f"./{path.relative_to(Path.cwd())}"
    except ValueError:
        return str(path)


def load_config(config_path):
    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    for key in ["data_dir", "checkpoint_root"]:
        path = Path(config[key])
        config[key] = path if path.is_absolute() else BASE_DIR / path

    return argparse.Namespace(**config)


def main():
    # 학습 설정은 train_config.json에서 수정
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=BASE_DIR / "train_config.json")
    cli_args = parser.parse_args()
    args = load_config(cli_args.config)
    args.config = cli_args.config

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.checkpoint_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    log_path = run_dir / "train.log"

    write_log(log_path, "Training configuration")
    write_log(log_path, f"  config={display_path(args.config)}")
    write_log(log_path, f"  data_dir={display_path(args.data_dir)}")
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

    write_log(log_path, f"\nLoading {args.input_file}...")
    X = np.load(args.data_dir / args.input_file, mmap_mode="r")
    if X.shape[-1] < 32:
        raise ValueError("Input time dimension is too short for EEGNet pooling.")

    write_log(log_path, f"Device: {device} | X: {X.shape} | run: {display_path(run_dir)}")

    if not 0 < args.val_test_ratio < 0.5:
        raise ValueError("--val_test_ratio must be between 0 and 0.5.")

    n_trials, n_channels, _ = X.shape
    original_y = np.load(args.data_dir / "y.npy").astype(np.int64)
    if len(original_y) != n_trials:
        raise ValueError(f"X and y.npy trial counts do not match: {n_trials} != {len(original_y)}")
    all_idx = np.arange(n_trials)

    # original label 기준으로 train/val/test split
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
    write_log(log_path, f"Data split: train={split_sizes[0]} val={split_sizes[1]} test={split_sizes[2]}")
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
        "split_sizes": split_sizes,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    # train set 통계로 한 번만 정규화
    write_log(log_path, "Normalizing X using training-set mean and std...")
    train_mean = X[train_idx].mean(axis=(0, 2))[:, None].astype(np.float32)
    train_std = X[train_idx].std(axis=(0, 2))[:, None].astype(np.float32)
    train_std = np.maximum(train_std, 1e-6)
    X = ((X - train_mean) / train_std).astype(np.float32)

    results = {}
    for label_name in LABEL_NAMES:
        write_log(log_path, f"\n=== y_{label_name} ===")
        model_dir = run_dir / f"y_{label_name}"
        model_dir.mkdir()
        checkpoint = model_dir / "best_model.pt"

        y = np.load(args.data_dir / f"y_{label_name}.npy").astype(np.int64)

        # Example: label_name "01" -> pair_labels {0, 1}.
        pair_labels = {int(label_name[0]), int(label_name[1])}

        # weighted_sampler=False이면 original label 수를 맞춰서 task dataset 구성
        if args.weighted_sampler:
            task_train_idx, task_val_idx, task_test_idx = train_idx, val_idx, test_idx
        else:
            task_train_idx = balanced_task_indices(original_y, train_idx, pair_labels, 60, 40, args.seed)
            task_val_idx = balanced_task_indices(original_y, val_idx, pair_labels, 20, 13, args.seed)
            task_test_idx = balanced_task_indices(original_y, test_idx, pair_labels, 20, 13, args.seed)

        # Class-count based weights for WeightedRandomSampler.
        train_counts = np.bincount(y[task_train_idx], minlength=2)
        sample_weights = 1.0 / train_counts[y[task_train_idx]]
        write_log(log_path, "Binary label distribution:")
        write_log(log_path, f"  train: {format_class_counts(y, task_train_idx)}")
        write_log(log_path, f"  val:   {format_class_counts(y, task_val_idx)}")
        write_log(log_path, f"  test:  {format_class_counts(y, task_test_idx)}")
        write_log(log_path, "Original labels in this split:")
        write_log(log_path, f"  train: {format_original_label_counts(original_y, task_train_idx)}")
        write_log(log_path, f"  val:   {format_original_label_counts(original_y, task_val_idx)}")
        write_log(log_path, f"  test:  {format_original_label_counts(original_y, task_test_idx)}")

        # weighted_sampler=True이면 train loader만 weighted sampling 적용
        if args.weighted_sampler:
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(task_train_idx),
                replacement=True,
                generator=torch.Generator().manual_seed(args.seed),
            )
            train_loader = DataLoader(
                EEGDataset(X, y, task_train_idx),
                batch_size=args.batch_size,
                sampler=sampler,
            )
        else:
            sampler = None
            train_loader = DataLoader(
                EEGDataset(X, y, task_train_idx),
                batch_size=args.batch_size,
                shuffle=True,
            )

        val_loader = DataLoader(
            EEGDataset(X, y, task_val_idx),
            batch_size=args.batch_size,
            shuffle=False,
        )

        test_loader = DataLoader(
            EEGDataset(X, y, task_test_idx),
            batch_size=args.batch_size,
            shuffle=False,
        )

        torch.manual_seed(args.seed)
        model = EEGNet(n_channels).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_val_loss = float("inf")
        best_val_bal_acc = best_val_acc = best_epoch = waiting = 0

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0
            train_class_correct, train_class_total = torch.zeros(2, device=device), torch.zeros(2, device=device)
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

            train_loss /= len(task_train_idx)
            train_bal_acc = (train_class_correct / train_class_total).mean().item()
            train_acc = (train_class_correct.sum() / train_class_total.sum()).item()
            val_loss, val_bal_acc, val_acc = evaluate(model, val_loader, criterion, device)
            if epoch == 1 or epoch % 10 == 0:
                write_log(log_path, f"epoch {epoch:03d} | "
                    f"train loss: {train_loss:.4f} bal_acc: {train_bal_acc:.2f} acc: {train_acc:.2f} | "
                    f"val loss: {val_loss:.4f} bal_acc: {val_bal_acc:.2f} acc: {val_acc:.2f}")

            if val_loss < best_val_loss:
                best_val_loss, best_val_bal_acc, best_val_acc = val_loss, val_bal_acc, val_acc
                best_epoch, waiting = epoch, 0
                # best model 기준은 validation loss
                torch.save(model.state_dict(), checkpoint)
            else:
                waiting += 1
                if waiting >= args.patience:
                    break

        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        test_loss, test_bal_acc, test_acc = evaluate(model, test_loader, criterion, device)
        results[label_name] = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_bal_acc": best_val_bal_acc,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_bal_acc": test_bal_acc,
            "test_acc": test_acc,
        }
        write_log(
            log_path,
            f"best epoch={best_epoch} | "
            f"val loss: {best_val_loss:.4f} bal_acc: {best_val_bal_acc:.2f} acc: {best_val_acc:.2f} | "
            f"test loss: {test_loss:.4f} bal_acc: {test_bal_acc:.2f} acc: {test_acc:.2f}"
        )

        del model, optimizer, train_loader, val_loader, test_loader, sampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    write_log(log_path, "\nTest accuracy")
    for name, result in results.items():
        write_log(log_path, f"y_{name}: {result['test_acc']:.3f}")
    write_log(log_path, f"\nSaved to: {display_path(run_dir)}")


if __name__ == "__main__":
    main()
