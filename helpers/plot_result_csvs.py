import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


LABEL_ORDER = ["0", "1 vs. 3", "2"]
MODELS = ["EEGNet", "ShallowNet", "FBCNet"]
COLORS = {"EEGNet": "#222222", "ShallowNet": "#2ca25f", "FBCNet": "#984ea3"}
MARKERS = {"EEGNet": "o", "ShallowNet": "s", "FBCNet": "^"}


def parse_mean_std(value):
    mean, std = re.split(r"\s*(?:±|\+/-)\s*", str(value).strip())
    return float(mean), float(std)


def load_results(csv_path):
    rows = []
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        for model in MODELS:
            mean, std = parse_mean_std(row[model])
            rows.append(
                {
                    "session": str(row["Session"]),
                    "label": str(row["Label"]),
                    "model": model,
                    "mean": mean,
                    "std": std,
                }
            )

    return pd.DataFrame(rows)


def plot_csv(csv_path):
    data = load_results(csv_path)
    sessions = sorted(data["session"].unique())
    x_positions = range(len(sessions))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True, constrained_layout=True)

    for ax, label in zip(axes, LABEL_ORDER):
        label_data = data[data["label"] == label]

        for model in MODELS:
            model_data = (
                label_data[label_data["model"] == model]
                .set_index("session")
                .reindex(sessions)
            )
            offsets = {
                "EEGNet": -0.08,
                "ShallowNet": 0.0,
                "FBCNet": 0.08,
            }
            offset = offsets[model]
            xs = [x + offset for x in x_positions]

            ax.errorbar(
                xs,
                model_data["mean"],
                yerr=model_data["std"],
                label=model,
                color=COLORS[model],
                marker=MARKERS[model],
                linewidth=1.8,
                markersize=5,
                capsize=3,
            )

        ax.axhline(0.5, color="0.35", linestyle="--", linewidth=1)
        ax.set_title(f"Label {label}")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(sessions, rotation=45, ha="right")
        ax.set_xlabel("Session")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0.3, 1.0)

    axes[0].set_ylabel("Mean balanced accuracy")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle(csv_path.stem)

    output_path = csv_path.with_suffix(".png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def iter_csv_paths(paths):
    for path in paths:
        if path.is_dir():
            yield from sorted(path.glob("*.csv"))
        else:
            yield path


def main():
    parser = argparse.ArgumentParser(
        description="Plot result CSVs as faceted point plots with error bars."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="CSV files or folders")
    args = parser.parse_args()

    for csv_path in iter_csv_paths(args.paths):
        output_path = plot_csv(csv_path)
        print(f"{csv_path} -> {output_path}")


if __name__ == "__main__":
    main()
