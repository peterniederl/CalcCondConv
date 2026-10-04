import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_histories(results_dir, run_names, metric):
    results_path = Path(results_dir)
    histories = {}
    for run_name in run_names:
        history_path = results_path / run_name / "history.json"
        if history_path.is_file():
            history = json.loads(history_path.read_text())
            values = history.get(metric) or []
        else:
            log_path = results_path / run_name / "training_log.csv"
            if not log_path.is_file():
                raise FileNotFoundError(
                    f"Neither history.json nor training_log.csv found in "
                    f"{results_path / run_name}"
                )
            with log_path.open(newline="") as log_file:
                rows = csv.DictReader(log_file)
                if metric not in (rows.fieldnames or []):
                    raise KeyError(f"Metric {metric!r} not found in {log_path}")
                values = [row[metric] for row in rows if row.get(metric, "") != ""]
        if not values:
            raise KeyError(f"Metric {metric!r} has no values in {results_path / run_name}")
        histories[run_name] = [float(value) for value in values]
    return histories


def compare_histories(histories, metric, output_path=None):
    figure, axis = plt.subplots(figsize=(11, 6))
    for run_name, values in histories.items():
        epochs = range(1, len(values) + 1)
        axis.plot(epochs, values, linewidth=2, label=run_name)

    axis.set_title(f"{metric} comparison")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(metric)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", frameon=False)
    figure.tight_layout()

    if output_path is None:
        plt.show()
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
        print(f"History comparison written to {output_path}")
    plt.close(figure)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare one history metric across explicitly selected experiment runs."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run directory names, for example se_run_01 axial_avg_pool_dual_norm_run_01",
    )
    parser.add_argument(
        "--metric",
        default="val_accuracy",
        help="History metric to plot (default: val_accuracy).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing the run folders (default: results).",
    )
    parser.add_argument(
        "--output",
        help="Optional image path. If omitted, display the plot interactively.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    histories = load_histories(args.results_dir, args.runs, args.metric)
    compare_histories(histories, args.metric, args.output)


if __name__ == "__main__":
    main()
