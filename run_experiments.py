import argparse
import csv
import gc
import json
import subprocess
from pathlib import Path

import tensorflow as tf

from experiment_config import ExperimentConfig
from experiment_runner import ResNetExperiment


MODEL_CONVOLUTION = {
    "standard": "standard",
    "grouped": "grouped",
    "depthwise_separable": "depthwise_separable",
    "bottleneck_conv": "bottleneck_conv",
    "condconv": "condconv",
    "basis_conv": "basis_conv",
    "dynamic_basis_conv": "dynamic_basis_conv",
    "maxmean_dynamic": "maxmean_dynamic",
    "dynamic_mixer": "dynamic_mixer",
    "dynamic_depthwise": "dynamic_depthwise",
    "dynamic_basis_depthwise": "dynamic_basis_depthwise",
    "dynamic_basis_depthwise_low_rank": "dynamic_basis_depthwise_low_rank",
}


class ConfiguredExperiment(ResNetExperiment):
    def __init__(
        self,
        run_name,
        conv_type,
        seed,
        results_dir,
        groups,
        num_bases,
        reduction,
        coefficient_rank,
        kernel_size,
        num_experts,
        basis_factor,
        gate_reduction,
        mixer_reduction,
        mixer_rank,
        maxmean_temperature,
        maxmean_alpha,
        axial_position,
        axial_reduction,
        axial_use_pointwise,
        axial_use_tanh,
    ):
        self.run_name = run_name
        self.conv_type = conv_type
        self.seed = seed
        self.results_dir = results_dir
        self.groups = groups
        self.num_bases = num_bases
        self.reduction = reduction
        self.coefficient_rank = coefficient_rank
        self.kernel_size = kernel_size
        self.num_experts = num_experts
        self.basis_factor = basis_factor
        self.gate_reduction = gate_reduction
        self.mixer_reduction = mixer_reduction
        self.mixer_rank = mixer_rank
        self.maxmean_temperature = maxmean_temperature
        self.maxmean_alpha = maxmean_alpha
        self.axial_position = axial_position
        self.axial_reduction = axial_reduction
        self.axial_use_pointwise = axial_use_pointwise
        self.axial_use_tanh = axial_use_tanh

    def config(self):
        config = ExperimentConfig(
            run_name=self.run_name,
            seed=self.seed,
            results_dir=self.results_dir,
        )
        config.model["conv_type"] = self.conv_type
        config.model["groups"] = self.groups
        config.model["num_bases"] = self.num_bases
        config.model["reduction"] = self.reduction
        config.model["coefficient_rank"] = self.coefficient_rank
        config.model["kernel_size"] = self.kernel_size
        config.model["num_experts"] = self.num_experts
        config.model["basis_factor"] = self.basis_factor
        config.model["gate_reduction"] = self.gate_reduction
        config.model["mixer_reduction"] = self.mixer_reduction
        config.model["mixer_rank"] = self.mixer_rank
        config.model["maxmean_temperature"] = self.maxmean_temperature
        config.model["maxmean_alpha"] = self.maxmean_alpha
        config.model["axial_position"] = self.axial_position
        config.model["axial_reduction"] = self.axial_reduction
        config.model["axial_use_pointwise"] = self.axial_use_pointwise
        config.model["axial_use_tanh"] = self.axial_use_tanh
        return config


def _best_validation_metrics(history):
    val_accuracy = history.get("val_accuracy", [])
    val_loss = history.get("val_loss", [])
    if not val_accuracy:
        return None, None
    best_index = max(range(len(val_accuracy)), key=val_accuracy.__getitem__)
    best_loss = val_loss[best_index] if val_loss else None
    return val_accuracy[best_index], best_loss


def _read_result(results_dir, model_name, run_number, seed):
    run_name = f"{model_name}_run_{run_number:02d}"
    run_dir = Path(results_dir) / run_name
    history = json.loads((run_dir / "history.json").read_text())
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    best_val_accuracy, best_val_loss = _best_validation_metrics(history)
    return {
        "model": model_name,
        "run": run_number,
        "seed": seed,
        "run_name": run_name,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "final_val_accuracy": history["val_accuracy"][-1],
        "final_val_loss": history["val_loss"][-1],
        "evaluation_accuracy": evaluation["accuracy"],
        "evaluation_loss": evaluation["loss"],
    }


def _average_best_validation_accuracy(results_dir):
    scores = []
    results_path = Path(results_dir)
    for history_path in results_path.glob("*_run_*/history.json"):
        if not _run_is_complete(history_path.parent):
            continue
        history = json.loads(history_path.read_text())
        val_accuracy = history.get("val_accuracy") or []
        if val_accuracy:
            scores.append(max(val_accuracy))
    return sum(scores) / len(scores) if scores else None


def _run_is_complete(run_dir):
    return all(
        (run_dir / artifact).is_file()
        for artifact in ("history.json", "evaluation.json", "final_model.keras")
    )


def _next_run_number(results_dir, model_name):
    run_number = 1
    while True:
        run_dir = Path(results_dir) / f"{model_name}_run_{run_number:02d}"
        if not run_dir.exists() or not _run_is_complete(run_dir):
            return run_number
        run_number += 1


def write_summary(results_dir, records):
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    (results_path / "comparison.json").write_text(json.dumps(records, indent=2))

    fields = list(records[0]) if records else []
    with (results_path / "comparison.csv").open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def run_experiments(
    run_counts,
    model_names,
    base_seed,
    results_dir,
    groups=2,
    num_bases=32,
    reduction=4,
    coefficient_rank=8,
    kernel_size=3,
    num_experts=4,
    basis_factor=1,
    gate_reduction=4,
    mixer_reduction=4,
    mixer_rank=8,
    maxmean_temperature=0.1,
    maxmean_alpha=0.1,
    axial_position="none",
    axial_reduction=1,
    axial_use_pointwise=False,
    axial_use_tanh=False,
):
    records = []
    for model_name in model_names:
        for _ in range(run_counts[model_name]):
            run_number = _next_run_number(results_dir, model_name)
            seed = base_seed + run_number - 1
            run_name = f"{model_name}_run_{run_number:02d}"
            print(f"\nStarting {run_name} with seed {seed}")
            experiment = ConfiguredExperiment(
                run_name=run_name,
                conv_type=MODEL_CONVOLUTION[model_name],
                seed=seed,
                results_dir=results_dir,
                groups=groups,
                num_bases=num_bases,
                reduction=reduction,
                coefficient_rank=coefficient_rank,
                kernel_size=kernel_size,
                num_experts=num_experts,
                basis_factor=basis_factor,
                gate_reduction=gate_reduction,
                mixer_reduction=mixer_reduction,
                mixer_rank=mixer_rank,
                maxmean_temperature=maxmean_temperature,
                maxmean_alpha=maxmean_alpha,
                axial_position=axial_position,
                axial_reduction=axial_reduction,
                axial_use_pointwise=axial_use_pointwise,
                axial_use_tanh=axial_use_tanh,
            )
            experiment.run()
            records.append(_read_result(results_dir, model_name, run_number, seed))
            tf.keras.backend.clear_session()
            gc.collect()
            write_summary(results_dir, records)
            experiment.publish_results()

            average_best_accuracy = _average_best_validation_accuracy(results_dir)
            current_best_accuracy = records[-1]["best_val_accuracy"]
            if (
                average_best_accuracy is not None
                and current_best_accuracy < average_best_accuracy
            ):
                print(
                    f"Skipping remaining {model_name} runs: "
                    f"best validation accuracy {current_best_accuracy:.4f} is below "
                    f"the global average {average_best_accuracy:.4f}."
                )
                break

            if len(records) % 3 == 0:
                subprocess.run(["clear"], check=False)

    return records


def _parse_run_counts(run_specs, model_names):
    run_counts = dict.fromkeys(model_names, 3)
    for run_spec in run_specs or []:
        try:
            model_name, count_text = run_spec.split("=", 1)
            count = int(count_text)
        except ValueError as error:
            raise ValueError(
                f"Invalid run specification {run_spec!r}; use MODEL=COUNT"
            ) from error
        if model_name not in MODEL_CONVOLUTION:
            raise ValueError(f"Unknown model in --runs: {model_name}")
        if model_name not in model_names:
            raise ValueError(
                f"Model {model_name!r} must also be selected with --models"
            )
        if count < 1:
            raise ValueError(f"Run count for {model_name} must be at least 1")
        run_counts[model_name] = count
    return run_counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare ResNets with configurable residual convolutions."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        metavar="MODEL=COUNT",
        help="Runs per model, for example standard=2 grouped=5 (default: 3 each).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_CONVOLUTION),
        default=list(MODEL_CONVOLUTION),
        help="Convolution types to compare (default: all registered types).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the first run.")
    parser.add_argument(
        "--groups", type=int, default=2,
        help="Number of groups for grouped convolutions (default: 2).",
    )
    parser.add_argument(
        "--num-bases", type=int, default=32,
        help="Number of learned kernels for dynamic depthwise variants (default: 32).",
    )
    parser.add_argument(
        "--reduction", type=int, default=4,
        help="Hidden-layer reduction factor for dynamic variants (default: 4).",
    )
    parser.add_argument(
        "--coefficient-rank", type=int, default=8,
        help="Low-rank factorization rank for dynamic_basis_depthwise_low_rank (default: 8).",
    )
    parser.add_argument(
        "--kernel-size", type=int, default=3,
        help="Square kernel size for dynamic depthwise variants (default: 3).",
    )
    parser.add_argument(
        "--num-experts", type=int, default=4,
        help="Expert kernel count for CondConv2D (default: 4).",
    )
    parser.add_argument(
        "--basis-factor", type=int, default=1,
        help="Channel reduction factor for basis_conv; must divide stage channels (default: 1).",
    )
    parser.add_argument(
        "--gate-reduction", type=int, default=4,
        help="Hidden-width reduction for dynamic_basis_conv's sigmoid gate (default: 4).",
    )
    parser.add_argument(
        "--mixer-reduction", type=int, default=4,
        help="Reduction factor for the dynamic_mixer router hidden width (default: 4).",
    )
    parser.add_argument(
        "--mixer-rank", type=int, default=8,
        help="Low-rank channel mixer rank for dynamic_mixer (default: 8).",
    )
    parser.add_argument(
        "--maxmean-temperature", type=float, default=0.1,
        help="Initial positive temperature for maxmean_dynamic (default: 0.1).",
    )
    parser.add_argument(
        "--maxmean-alpha", type=float, default=0.1,
        help="Initial strength of the MaxMean dynamic branch (default: 0.1).",
    )
    parser.add_argument(
        "--axial-position", choices=["none", "first", "second", "both"], default="none",
        help="Insert AxialContext before the first and/or second convolution of each residual block (default: none).",
    )
    parser.add_argument(
        "--axial-reduction", type=int, default=1,
        help="Channel reduction factor inside AxialContext; needs --axial-pointwise when > 1 (default: 1).",
    )
    parser.add_argument(
        "--axial-pointwise", action="store_true",
        help="Apply a 1x1 projection before the axial convolutions.",
    )
    parser.add_argument(
        "--axial-tanh", action="store_true",
        help="Apply tanh to the concatenated axial context.",
    )
    parser.add_argument("--results-dir", default="results", help="Directory for run outputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        run_counts = _parse_run_counts(args.runs, args.models)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    if (
        args.groups < 1
        or args.num_bases < 1
        or args.reduction < 1
        or args.coefficient_rank < 1
        or args.kernel_size < 1
        or args.num_experts < 1
        or args.basis_factor < 1
        or args.gate_reduction < 1
        or args.mixer_reduction < 1
        or args.mixer_rank < 1
        or args.maxmean_temperature <= 0
        or args.axial_reduction < 1
    ):
        raise SystemExit(
            "error: --groups, --num-bases, --reduction, --coefficient-rank, --kernel-size, --num-experts, --basis-factor, --gate-reduction, --mixer-reduction, --mixer-rank, --maxmean-temperature, and --axial-reduction must be positive"
        )
    if args.axial_reduction > 1 and not args.axial_pointwise:
        raise SystemExit("error: --axial-reduction > 1 requires --axial-pointwise")
    if "grouped" in args.models and args.groups < 2:
        raise SystemExit("error: --groups must be at least 2 when selecting grouped")
    records = run_experiments(
        run_counts,
        args.models,
        args.seed,
        args.results_dir,
        groups=args.groups,
        num_bases=args.num_bases,
        reduction=args.reduction,
        coefficient_rank=args.coefficient_rank,
        kernel_size=args.kernel_size,
        num_experts=args.num_experts,
        basis_factor=args.basis_factor,
        gate_reduction=args.gate_reduction,
        mixer_reduction=args.mixer_reduction,
        mixer_rank=args.mixer_rank,
        maxmean_temperature=args.maxmean_temperature,
        maxmean_alpha=args.maxmean_alpha,
        axial_position=args.axial_position,
        axial_reduction=args.axial_reduction,
        axial_use_pointwise=args.axial_pointwise,
        axial_use_tanh=args.axial_tanh,
    )
    print(f"\nCompleted {len(records)} runs.")
    print(f"Comparison written to {Path(args.results_dir) / 'comparison.csv'}")


if __name__ == "__main__":
    main()