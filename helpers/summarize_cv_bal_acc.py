import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def find_balanced_accuracy_column(fieldnames):
    if "test_bal_acc" in fieldnames:
        return "test_bal_acc"
    if len(fieldnames) < 2:
        raise ValueError("CSV must have at least two columns to use the penultimate column")
    return fieldnames[-2]


def read_run_summary(csv_path):
    by_label = defaultdict(list)

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("empty CSV")
        if "label" not in reader.fieldnames:
            raise ValueError("missing required 'label' column")

        bal_acc_col = find_balanced_accuracy_column(reader.fieldnames)
        for row in reader:
            label = row["label"]
            if not label:
                continue
            by_label[label].append(float(row[bal_acc_col]))

    return {
        label: {
            "n": len(values),
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
        }
        for label, values in sorted(by_label.items())
    }


def iter_run_csvs(root):
    for csv_path in sorted(root.rglob("cross_validation.csv")):
        yield csv_path.parent, csv_path


def read_run_config(run_dir):
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return {"session": "", "patient": "", "model": ""}

    with config_path.open() as f:
        config = json.load(f)

    args = config.get("args", {})
    return {
        "session": args.get("day", ""),
        "patient": args.get("patient", ""),
        "model": args.get("model", ""),
    }


def print_summary(root, run_summaries):
    print(f"Root: {root}")
    if not run_summaries:
        print("No cross_validation.csv files found.")
        return

    for run_dir, summary in run_summaries:
        print(f"\n[{run_dir.relative_to(root)}]")
        print(f"{'label':<16} {'folds':>5} {'mean_bal_acc':>13} {'std':>10}")
        print("-" * 48)
        for label, stats in summary.items():
            print(
                f"{label:<16} {stats['n']:>5d} "
                f"{stats['mean']:>13.4f} {stats['std']:>10.4f}"
            )


def write_summary_csv(root, run_summaries, output_csv):
    fieldnames = [
        "run_index",
        "session",
        "patient",
        "model",
        "label",
        "folds",
        "mean_bal_acc",
        "std_bal_acc",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for run_dir, summary in run_summaries:
            run_config = read_run_config(run_dir)
            for label, stats in summary.items():
                writer.writerow(
                    {
                        "run_index": run_dir.name,
                        "session": run_config["session"],
                        "patient": run_config["patient"],
                        "model": run_config["model"],
                        "label": label,
                        "folds": stats["n"],
                        "mean_bal_acc": stats["mean"],
                        "std_bal_acc": stats["std"],
                    }
                )


def main():
    parser = argparse.ArgumentParser(
        description="Summarize test balanced accuracy across folds for each run."
    )
    parser.add_argument(
        "run_root",
        type=Path,
        help="Folder containing run subdirectories with cross_validation.csv files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Where to save the summary CSV. Defaults to "
            "RUN_ROOT/<run_root_name>_cv_bal_acc_summary.csv."
        ),
    )
    args = parser.parse_args()

    root = args.run_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    output_csv = (
        args.output_csv.expanduser().resolve()
        if args.output_csv
        else root / f"{root.name}_cv_bal_acc_summary.csv"
    )

    run_summaries = []
    for run_dir, csv_path in iter_run_csvs(root):
        try:
            run_summaries.append((run_dir, read_run_summary(csv_path)))
        except Exception as exc:
            print(f"Skipping {csv_path}: {exc}")

    print_summary(root, run_summaries)
    write_summary_csv(root, run_summaries, output_csv)
    print(f"\nSaved CSV: {output_csv}")


if __name__ == "__main__":
    main()
