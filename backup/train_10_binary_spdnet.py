"""Train and test ten binary SPDNet models with the same time-ordered split."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


LABEL_NAMES = ["01", "02", "03", "04", "12", "13", "14", "23", "24", "34"]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/processed/260602_sub1_hjlee_raw"


class CovarianceDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X, self.y, self.indices = X, y, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x = np.array(self.X[self.indices[i]], dtype=np.float32, copy=True)
        return torch.from_numpy(x), int(self.y[self.indices[i]])


def spd_logm_batch(covariances, eps=1e-6):
    covariances = 0.5 * (covariances + covariances.transpose(-1, -2))
    eye = torch.eye(covariances.shape[-1], device=covariances.device)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariances + eps * eye)
    return eigenvectors @ torch.diag_embed(eigenvalues.clamp_min(eps).log()) @ eigenvectors.transpose(-1, -2)


class SPDNetLite(nn.Module):
    def __init__(self, n_windows, n_channels, proj_dim=8, hidden=24, dropout=0.25):
        super().__init__()
        self.proj = nn.Parameter(torch.randn(n_windows, n_channels, proj_dim) * 0.05)
        self.register_buffer("tri_idx", torch.triu_indices(proj_dim, proj_dim), persistent=False)
        feature_dim = n_windows * (proj_dim * (proj_dim + 1) // 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        projection, _ = torch.linalg.qr(self.proj, mode="reduced")
        projected = projection.transpose(-1, -2).unsqueeze(0) @ x @ projection.unsqueeze(0)
        projected = 0.5 * (projected + projected.transpose(-1, -2))
        log_covariances = spd_logm_batch(projected)
        features = log_covariances[:, :, self.tri_idx[0], self.tri_idx[1]].reshape(x.shape[0], -1)
        return self.classifier(features)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total = 0
    class_correct, class_total = torch.zeros(2, device=device), torch.zeros(2, device=device)
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        pred = logits.argmax(1)
        total_loss += criterion(logits, y).item() * len(y)
        total += len(y)
        class_total += torch.bincount(y, minlength=2)
        class_correct += torch.bincount(y[pred == y], minlength=2)
    return total_loss / total, (class_correct / class_total).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint-root", type=Path, default=BASE_DIR / "checkpoints_spdnet")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-test-ratio", type=float, default=0.2)
    parser.add_argument("--proj-dim", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=24)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = args.checkpoint_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)

    print("Loading covariance data...")
    X = np.load(args.data_dir / "X_eeg_cov.npy", mmap_mode="r")
    end = X.shape[1] if args.end is None else args.end
    X = X[:, args.start:end]

    n_trials, n_windows, n_channels, _ = X.shape
    train_end = int(n_trials * (1 - 2 * args.val_test_ratio))
    val_end = int(n_trials * (1 - args.val_test_ratio))
    train_idx, val_idx, test_idx = range(train_end), range(train_end, val_end), range(val_end, n_trials)
    split_sizes = [len(train_idx), len(val_idx), len(test_idx)]

    print(f"Device: {device} | X: {X.shape} | subwindows: {args.start}:{end} | run: {run_dir}")
    print(f"Data split: train={split_sizes[0]} val={split_sizes[1]} test={split_sizes[2]}")

    config_args = vars(args) | {
        "data_dir": str(args.data_dir),
        "checkpoint_root": str(args.checkpoint_root),
    }
    run_config = {
        "args": config_args,
        "input_shape": list(X.shape),
        "subwindow_range": [args.start, end],
        "split_sizes": split_sizes,
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    results = {}
    for label_name in LABEL_NAMES:
        print(f"\n=== y_{label_name} ===")
        model_dir = run_dir / f"y_{label_name}"
        model_dir.mkdir()
        checkpoint = model_dir / "best_model.pt"

        y = np.load(args.data_dir / f"y_{label_name}.npy").astype(np.int64)
        loaders = [
            DataLoader(
                CovarianceDataset(X, y, indices),
                batch_size=args.batch_size,
                shuffle=(i == 0),
            )
            for i, indices in enumerate((train_idx, val_idx, test_idx))
        ]
        train_loader, val_loader, test_loader = loaders

        torch.manual_seed(args.seed)
        model = SPDNetLite(n_windows, n_channels, args.proj_dim, args.hidden, args.dropout).to(device)
        counts = np.bincount(y[:train_end], minlength=2)
        weights = torch.tensor(len(train_idx) / (2 * counts), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        best_val_loss, best_val_acc = float("inf"), 0
        best_epoch = waiting = 0

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
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * len(y_batch)
                train_class_total += torch.bincount(y_batch, minlength=2)
                train_class_correct += torch.bincount(y_batch[pred == y_batch], minlength=2)

            train_loss /= len(train_idx)
            train_acc = (train_class_correct / train_class_total).mean().item()
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"epoch {epoch:03d} | "
                    f"train={train_loss:.4f}/{train_acc:.2f}% | "
                    f"val={val_loss:.4f}/{val_acc:.2f}%"
                )

            if (val_acc, -val_loss) > (best_val_acc, -best_val_loss):
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
            f"best epoch={best_epoch} val={best_val_loss:.4f}/{best_val_acc:.2f}% | "
            f"test={test_loss:.4f}/{test_acc:.2f}%"
        )

        del model, optimizer, loaders
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("\nTest balanced accuracy")
    for name, result in results.items():
        print(f"y_{name}: {result['test_acc']:.3f}")
    print(f"\nSaved to: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
