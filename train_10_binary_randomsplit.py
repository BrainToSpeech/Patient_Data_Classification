"""Train and test ten binary EEGNet models with one shared stratified split."""

import argparse
import json
from datetime import datetime
from pathlib import Path

from mne.decoding import CSP
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import mne

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

mne.set_log_level("WARNING")

LABEL_NAMES = ["01", "02", "03", "04", "12", "13", "14", "23", "24", "34"]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/processed/260602_sub1_hjlee_raw"


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
    return total_loss / total, (class_correct / class_total).mean().item()


def format_class_counts(y, indices):
    counts = np.bincount(y[indices], minlength=2)
    return f"class0={counts[0]} class1={counts[1]} total={counts.sum()}"


def format_original_label_counts(y, indices):
    counts = np.bincount(y[indices], minlength=5)
    labels = " ".join(f"label{i}={count}" for i, count in enumerate(counts))
    return f"{labels} total={counts.sum()}"

def make_svm_csp_model(args):
    return Pipeline([
        ("csp", CSP(n_components=4, reg="ledoit_wolf", log=True, cov_est="epoch", norm_trace=False)),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True))
    ])


def evaluate_svm_csp(model, X, y, indices):
    X_split = X[indices]
    y_split = y[indices]
    pred = model.predict(X_split)
    prob = model.predict_proba(X_split)

    return {
        "loss": log_loss(y_split, prob, labels=[0, 1]),
        "acc": balanced_accuracy_score(y_split, pred),
    }

def write_log(log_path, message=""):
    print(message)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{message}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--input-file", default="X_eeg_raw.npy")
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_test_ratio", type=float, default=0.2)
    parser.add_argument("--model", choices=["eegnet", "svm_csp"], default="eegnet")
    parser.add_argument("--csp-components", type=int, default=6)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-kernel", default="rbf", choices=["linear", "rbf"])
    parser.add_argument("--svm-gamma", default="scale")
    args = parser.parse_args()
    if args.checkpoint_root is None:
        args.checkpoint_root = BASE_DIR / f"checkpoints_{args.model}_randomsplit"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.checkpoint_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    log_path = run_dir / "train.log"

    write_log(log_path, "Training configuration")
    write_log(log_path, f"  model={args.model}")
    write_log(log_path, f"  data_dir={args.data_dir}")
    write_log(log_path, f"  input_file={args.input_file}")
    write_log(log_path, f"  epochs={args.epochs}")
    write_log(log_path, f"  batch_size={args.batch_size}")
    write_log(log_path, f"  lr={args.lr}")
    write_log(log_path, f"  weight_decay={args.weight_decay}")
    write_log(log_path, f"  patience={args.patience}")
    write_log(log_path, f"  seed={args.seed}")
    write_log(log_path, f"  val_test_ratio={args.val_test_ratio}")

    write_log(log_path, f"\nLoading {args.input_file}...")
    X = np.load(args.data_dir / args.input_file, mmap_mode="r")

    write_log(log_path, f"Device: {device} | X: {X.shape} | run: {run_dir}")

    n_trials, n_channels, _ = X.shape
    original_y = np.load(args.data_dir / "y.npy").astype(np.int64)

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
    write_log(log_path, f"Data split: train={split_sizes[0]} val={split_sizes[1]} test={split_sizes[2]}")
    write_log(log_path, "Original label distribution:")
    write_log(log_path, f"  train: {format_original_label_counts(original_y, train_idx)}")
    write_log(log_path, f"  val:   {format_original_label_counts(original_y, val_idx)}")
    write_log(log_path, f"  test:  {format_original_label_counts(original_y, test_idx)}")

    config_args = vars(args) | {
        "data_dir": str(args.data_dir),
        "checkpoint_root": str(args.checkpoint_root),
    }
    run_config = {
        "args": config_args,
        "input_shape": list(X.shape),
        "split_sizes": split_sizes,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    if args.model == "eegnet":
        write_log(log_path, "Normalizing X using training-set mean and std...")
        train_mean = X[train_idx].mean(axis=(0, 2))[:, None].astype(np.float32)
        train_std = X[train_idx].std(axis=(0, 2))[:, None].astype(np.float32)
        train_std = np.maximum(train_std, 1e-6)
        X = ((X - train_mean) / train_std).astype(np.float32)
    else:
        write_log(log_path, "Using raw X for SVM+CSP.")
        X = np.asarray(X, dtype=np.float64)

    results = {}
    for label_name in LABEL_NAMES:
        write_log(log_path, f"\n=== y_{label_name} ===")
        model_dir = run_dir / f"y_{label_name}"
        model_dir.mkdir()
        checkpoint = model_dir / "best_model.pt"

        y = np.load(args.data_dir / f"y_{label_name}.npy").astype(np.int64)

        train_counts = np.bincount(y[train_idx], minlength=2)
        sample_weights = 1.0 / train_counts[y[train_idx]]
        write_log(log_path, "Binary label distribution:")
        write_log(log_path, f"  train: {format_class_counts(y, train_idx)}")
        write_log(log_path, f"  val:   {format_class_counts(y, val_idx)}")
        write_log(log_path, f"  test:  {format_class_counts(y, test_idx)}")
        write_log(log_path, "Original labels in this split:")
        write_log(log_path, f"  train: {format_original_label_counts(original_y, train_idx)}")
        write_log(log_path, f"  val:   {format_original_label_counts(original_y, val_idx)}")
        write_log(log_path, f"  test:  {format_original_label_counts(original_y, test_idx)}")

        if args.model == "svm_csp":
            model = make_svm_csp_model(args)
            model.fit(X[train_idx], y[train_idx])

            val_result = evaluate_svm_csp(model, X, y, val_idx)
            test_result = evaluate_svm_csp(model, X, y, test_idx)

            results[label_name] = {
                "best_epoch": None,
                "best_val_loss": val_result["loss"],
                "best_val_acc": val_result["acc"],
                "test_loss": test_result["loss"],
                "test_acc": test_result["acc"],
            }

            write_log(
                log_path,
                f"svm_csp val={val_result['loss']:.4f}/{val_result['acc']:.2f} | "
                f"test={test_result['loss']:.4f}/{test_result['acc']:.2f}"
            )
            continue

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
        model = EEGNet(n_channels).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=20,
            min_lr=1e-7,
        )

        best_val_loss = float("inf")
        best_val_acc = best_epoch = waiting = 0

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

            train_loss /= len(train_idx)
            train_acc = (train_class_correct / train_class_total).mean().item()
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step(val_loss)
            if epoch == 1 or epoch % 10 == 0:
                write_log(log_path, f"epoch {epoch:03d} | "
                    f"train(loss/acc)={train_loss:.4f}/{train_acc:.2f} | "
                    f"val(loss/acc)={val_loss:.4f}/{val_acc:.2f} | "
                    f"lr={optimizer.param_groups[0]['lr']:.1e}")

            if val_loss < best_val_loss:
                best_val_loss, best_val_acc = val_loss, val_acc
                best_epoch, waiting = epoch, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                waiting += 1
                if waiting >= args.patience:
                    break

        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        results[label_name] = {
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        write_log(
            log_path,
            f"best epoch={best_epoch} val={best_val_loss:.4f}/{best_val_acc:.2f} | "
            f"test={test_loss:.4f}/{test_acc:.2f}"
        )

        del model, optimizer, scheduler, train_loader, val_loader, test_loader, sampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    write_log(log_path, "\nTest accuracy")
    for name, result in results.items():
        write_log(log_path, f"y_{name}: {result['test_acc']:.3f}")
    write_log(log_path, f"\nSaved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
