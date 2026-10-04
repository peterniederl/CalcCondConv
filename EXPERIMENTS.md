# ResNet experiments

The experiment scripts use one shared ResNet pipeline so augmentation, data ordering, optimizer, loss, and training settings stay identical across runs.

Run a baseline, SE, or CBAM experiment from the repository root:

```bash
python3 run_baseline.py
python3 run_se.py
python3 run_cbam.py
```

Each run creates `results/<run-name>/` containing:

- `settings.json`: all experiment and augmentation settings
- `environment.json`: Python, platform, and TensorFlow versions
- `model_config.json`: serialized Keras model setup
- `model_summary.txt`: readable architecture and parameter count
- `training_log.csv`: epoch-by-epoch Keras log
- `history.json`: training and validation metrics
- `evaluation.json`: final evaluation metrics
- `best_model.keras` and `final_model.keras`: saved models

Create a new SE or CBAM test by copying a run file and changing only its class and config, for example:

```python
from experiment_config import ExperimentConfig
from experiment_runner import ResNetExperiment


class MyExperiment(ResNetExperiment):
    def config(self):
        return ExperimentConfig(run_name="my_test", attention="cbam")


if __name__ == "__main__":
    MyExperiment().run()
```

For a completely new model architecture, add a builder to `model_factory.py` and override `build_model()` in the new run file. The inherited runner still uses the shared data pipeline and saves the same artifacts:

```python
from experiment_config import ExperimentConfig
from experiment_runner import ResNetExperiment
from model_factory import build_my_model


class MyModelExperiment(ResNetExperiment):
    def config(self):
        return ExperimentConfig(run_name="my_model", attention="none")

    def build_model(self, config):
        return build_my_model(config)
```

Visualize one, several, or all completed runs:

```bash
python3 visualize_results.py baseline
python3 visualize_results.py baseline se cbam --output results/comparison.png
python3 visualize_results.py
```
