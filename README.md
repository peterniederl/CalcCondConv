# ResNet Convolution Experiments on Tiny ImageNet

This repository provides a reproducible training pipeline for comparing ten ResNet residual-block convolution types:

- **Standard**: regular 3 x 3 convolutions in each residual block
- **Grouped**: independent grouped 3 x 3 convolutions in each residual block (two groups by default)
- **Depthwise separable**: a 3 x 3 depthwise convolution followed by a 1 x 1 pointwise convolution
- **Conditional convolution**: per-image mixtures of learned full-convolution expert kernels
- **Basis convolution**: a learned spatial basis per input channel combined with a learned channel-mixing matrix
- **Dynamic basis convolution**: per-example tanh gates weight the reduced basis channels before channel mixing
- **Dynamic mixer**: a per-image diagonal-plus-low-rank channel mixer applied after a shared depthwise spatial basis
- **Dynamic depthwise**: per-image mixtures of learned per-channel spatial kernels
- **Dynamic basis depthwise**: shared spatial basis kernels with per-image, per-channel mixing
- **Low-rank dynamic basis depthwise**: the same per-channel mixing, factorized through a configurable low rank

The dynamic depthwise operators preserve the input channel count, so each is followed by a 1 x 1 convolution to match the residual block's output channels. The ResNet stem and shortcut projections continue to use standard `Conv2D`; the selected type replaces the two 3 x 3 convolution operations in each residual block.

The experiments use the Tiny ImageNet dataset with a shared data pipeline, augmentation settings, optimizer, loss, learning-rate schedule, and evaluation procedure. This keeps comparisons between model variants consistent.

## Dataset

Download and extract Tiny ImageNet so the repository contains:

```text
tiny-imagenet-200/
├── train/
├── val/
├── wnids.txt
└── words.txt
```

The dataset contains 200 classes with 64 x 64 RGB images. The local dataset directory is ignored by Git because it is input data rather than source code.

## Running experiments

The recommended entry point runs three repetitions for each model by default:

```bash
python3 run_experiments.py
```

This runs all ten variants for three runs each by default. To choose models and set run counts independently:

```bash
python3 run_experiments.py --models standard grouped depthwise_separable condconv basis_conv dynamic_basis_conv dynamic_mixer dynamic_depthwise dynamic_basis_depthwise dynamic_basis_depthwise_low_rank \
    --runs standard=5 grouped=3 depthwise_separable=2 condconv=2 basis_conv=2 dynamic_depthwise=2 dynamic_basis_depthwise=2 dynamic_basis_depthwise_low_rank=2
    --runs standard=5 grouped=3 depthwise_separable=2 condconv=2 basis_conv=2 dynamic_mixer=2 dynamic_depthwise=2 dynamic_basis_depthwise=2 dynamic_basis_depthwise_low_rank=2
```

Tune convolution-specific parameters from the command line:

```bash
python3 run_experiments.py --models grouped --groups 4
python3 run_experiments.py --models dynamic_depthwise --num-bases 16 --reduction 8
python3 run_experiments.py --models dynamic_basis_depthwise_low_rank --num-bases 16 --coefficient-rank 8
python3 run_experiments.py --models dynamic_basis_depthwise --num-bases 16 --kernel-size 7
python3 run_experiments.py --models depthwise_separable --runs depthwise_separable=1
python3 run_experiments.py --models condconv --runs condconv=1 --num-experts 4
python3 run_experiments.py --models basis_conv --runs basis_conv=1
python3 run_experiments.py --models basis_conv --runs basis_conv=1 --basis-factor 2
python3 run_experiments.py --models dynamic_basis_conv --runs dynamic_basis_conv=1 --basis-factor 2 --gate-reduction 4
python3 run_experiments.py --models dynamic_mixer --runs dynamic_mixer=1 --mixer-rank 8 --mixer-reduction 4
```

Grouped convolutions require `--groups` to divide every input and output channel count in the residual stages. For `basis_conv`, `--basis-factor` is a channel reduction factor: the intermediate basis width is `input_channels // basis_factor`, so the factor must divide each stage's channel count. For example, 64 channels with factor 2 gives basis shape `[3, 3, 64, 32]` and mixer shape `[32, 64]`. `dynamic_basis_conv` uses the same reduced basis but predicts per-image tanh weights over those basis channels from global-average-pooled features; `--gate-reduction` controls the gate MLP width. CondConv uses 4 experts by default; set `--num-experts` to change this. Dynamic mixer derives its router hidden width from the input channel count as `max(channels // mixer_reduction, 1)`; `mixer_reduction` defaults to 4 and rank defaults to 8. Configure these with `--mixer-reduction` and `--mixer-rank`. Dynamic depthwise variants use 32 basis kernels, a reduction factor of 4, and a 3 x 3 kernel by default; the low-rank variant uses coefficient rank 8. Set `--kernel-size 7` to use 7 x 7 dynamic kernels. When building an `ExperimentConfig` directly, set `config.model["conv_type"]` to `standard`, `grouped`, `depthwise_separable`, `condconv`, `basis_conv`, `dynamic_basis_conv`, `dynamic_mixer`, `dynamic_depthwise`, `dynamic_basis_depthwise`, or `dynamic_basis_depthwise_low_rank`. Optionally set `config.model["groups"]`, `config.model["basis_factor"]`, `config.model["gate_reduction"]`, `config.model["num_experts"]`, `config.model["mixer_reduction"]`, `config.model["mixer_rank"]`, `config.model["num_bases"]`, `config.model["reduction"]`, `config.model["coefficient_rank"]`, or `config.model["kernel_size"]`.

Each model is trained for 100 epochs by default. A new model is created for each run, and the previous TensorFlow session is cleared before the next model is built.

## Results

Each completed run is saved under `results/<run-name>/`, including:

- experiment and environment settings
- model configuration and summary
- epoch-by-epoch training logs
- training history and final evaluation metrics
- best weights checkpoint and final Keras model file

The multi-run script also creates:

```text
results/comparison.json
results/comparison.csv
```

These files contain the metrics for every model and repetition, including best validation accuracy, final validation accuracy, and evaluation accuracy.

To plot completed runs:

```bash
python3 visualize_results.py
python3 visualize_results.py standard_run_01 grouped_run_01 \
    --output results/comparison.png
```

## Project structure

```text
experiment_config.py   Shared training configuration
data_pipeline.py       Tiny ImageNet loading and augmentation
model_factory.py       ResNet with selectable residual convolution operators
experiment_runner.py   Shared training and evaluation logic
run_experiments.py     Repeated multi-model experiment runner
visualize_results.py   Result loading and plotting
callbacks.py           Learning-rate scheduling callback
```

The older ConvMixer notebook and its supporting utilities are retained separately as historical research material and are not part of the current ResNet experiment pipeline.
