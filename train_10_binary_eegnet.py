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

LABEL_NAMES = ["01", "02", "03", "04", "12", "13", "14", "23", "24", "34"]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/processed/260602_sub1_hjlee_raw"


class EEGDataset(Dataset):
    def __init__(self, X, y, indices, mean, std):
        self.X, self.y, self.indices = X, y, indices
        self.mean, self.std = mean, std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x = (self.X[self.indices[i]] - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32)), int(self.y[self.indices[i]])


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--input-file", default="X_eeg_raw.npy")
    parser.add_argument("--checkpoint-root", type=Path, default=BASE_DIR / "checkpoints_eegnet_2s")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_test_ratio", type=float, default=0.2)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--end-s", type=float, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if X.shape[-1] < 32:
        raise ValueError("Selected time range is too short for EEGNet pooling.")

    print(f"Device: {device} | X: {X.shape} | "
        f"time: {args.start_s:g}-{end_s:g}s | run: {run_dir}")

    if not 0 < args.val_test_ratio < 0.5:
        raise ValueError("--val_test_ratio must be between 0 and 0.5.")

    n_trials, n_channels, _ = X.shape
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

    results = {}
    for label_name in LABEL_NAMES:
        print(f"\n=== y_{label_name} ===")
        model_dir = run_dir / f"y_{label_name}"
        model_dir.mkdir()
        checkpoint = model_dir / "best_model.pt"

        y = np.load(args.data_dir / f"y_{label_name}.npy").astype(np.int64)

        train_counts = np.bincount(y[train_idx], minlength=2)
        sample_weights = 1.0 / train_counts[y[train_idx]]

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_idx),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )

        train_loader = DataLoader(
            EEGDataset(X, y, train_idx, train_mean, train_std),
            batch_size=args.batch_size,
            sampler=sampler,
        )

        val_loader = DataLoader(
            EEGDataset(X, y, val_idx, train_mean, train_std),
            batch_size=args.batch_size,
            shuffle=False,
        )

        test_loader = DataLoader(
            EEGDataset(X, y, test_idx, train_mean, train_std),
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
                print(f"epoch {epoch:03d} | "
                    f"train={train_loss:.4f}/{train_acc:.2f} | "
                    f"val={val_loss:.4f}/{val_acc:.2f} | "
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
        print(
            f"best epoch={best_epoch} val={best_val_loss:.4f}/{best_val_acc:.2f} | "
            f"test={test_loss:.4f}/{test_acc:.2f}"
        )

        del model, optimizer, scheduler, train_loader, val_loader, test_loader, sampler
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("\nTest accuracy")
    for name, result in results.items():
        print(f"y_{name}: {result['test_acc']:.3f}")
    print(f"\nSaved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
