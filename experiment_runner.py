import json
import platform
import random
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from callbacks import CosineAnnealingScheduler
from data_pipeline import TinyImageNetData
from model_factory import build_resnet


class ResNetExperiment:
    """Shared runner. A new experiment only needs to override config()."""

    def config(self):
        raise NotImplementedError

    def build_model(self, config):
        return build_resnet(config)

    def _generate_html_report(self, results_dir):
        repo_root = Path(__file__).resolve().parent
        report_path = Path(results_dir) / "comparison.html"
        script_path = repo_root / "visualize_results.py"
        subprocess.run(
            ["python3", str(script_path), "--results-dir", str(results_dir), "--output", str(report_path)],
            cwd=str(repo_root),
            check=True,
        )
        return report_path

    def _publish_results_to_github(self, results_dir):
        repo_root = Path(__file__).resolve().parent
        git_dir = repo_root / ".git"
        if not git_dir.exists():
            print("[publish] No git repository detected, skipping Git push.")
            return False

        files_to_stage = [str(Path(results_dir))]

        try:
            subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=str(repo_root), check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError:
            print("[publish] Git repository is not valid, skipping Git push.")
            return False

        try:
            subprocess.run(["git", "add", *files_to_stage], cwd=str(repo_root), check=True)
            diff_status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=str(repo_root), capture_output=True)
            if diff_status.returncode == 0:
                print("[publish] No new comparison artifacts to publish.")
                return False
            subprocess.run(["git", "commit", "-m", "Update experiment results"], cwd=str(repo_root), check=True)
            subprocess.run(["git", "push", "origin", "HEAD"], cwd=str(repo_root), check=True)
            print("[publish] Results and HTML report were pushed to GitHub.")
            return True
        except subprocess.CalledProcessError as exc:
            print(f"[publish] Git publish failed: {exc}")
            return False

    def publish_results(self):
        results_dir = Path(self.config().results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        self._generate_html_report(results_dir)
        return self._publish_results_to_github(results_dir)

    def run(self):
        config = self.config()
        random.seed(config.seed)
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)
        use_mixed_precision = not (
            sys.platform == "darwin" and platform.machine() == "arm64"
        )
        tf.keras.mixed_precision.set_global_policy(
            "mixed_float16" if use_mixed_precision else "float32"
        )
        run_dir = Path(config.results_dir) / config.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(run_dir / "settings.json", config.to_dict())
        self._write_json(run_dir / "environment.json", {
            "python": sys.version,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
        })

        train_ds, val_ds = TinyImageNetData(config).datasets()
        model = self.build_model(config)
        if sys.platform == "darwin" and platform.machine() == "arm64":
            optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=config.base_lr,
                beta_1=config.beta_1,
                beta_2=config.beta_2,
            )
        else:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=config.base_lr,
                weight_decay=config.weight_decay,
                beta_1=config.beta_1,
                beta_2=config.beta_2,
            )
        model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=config.label_smoothing), metrics=["accuracy"])
        self._write_json(run_dir / "model_config.json", json.loads(model.to_json()))
        with (run_dir / "model_summary.txt").open("w") as summary_file:
            with redirect_stdout(summary_file):
                model.summary()

        callbacks = [
            CosineAnnealingScheduler(config.base_lr, config.min_lr, config.epochs, config.warmup_epochs),
            keras.callbacks.CSVLogger(run_dir / "training_log.csv"),
            keras.callbacks.ModelCheckpoint(
                str(run_dir / "best_model.weights.h5"),
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                save_weights_only=True,
            ),
        ]
        history = model.fit(
            train_ds,
            epochs=config.epochs,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=1,
        )
        evaluation = model.evaluate(val_ds, return_dict=True, verbose=1)
        self._write_json(run_dir / "history.json", history.history)
        self._write_json(run_dir / "evaluation.json", {key: float(value) for key, value in evaluation.items()})
        model.save(run_dir / "final_model.keras")
        return run_dir

    @staticmethod
    def _write_json(path, value):
        with path.open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
