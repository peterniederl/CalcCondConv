from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from augmentation_config import AUG, CUTMIX_ALPHA, CUTMIX_PROB, MIXUP_ALPHA, MIX_PROB


@dataclass
class ExperimentConfig:
    run_name: str
    attention: str = "none"
    data_dir: str = "tiny-imagenet-200"
    results_dir: str = "results"
    image_size: int = 64
    batch_size: int = 128
    num_classes: int = 200
    epochs: int = 100
    seed: int = 42
    base_lr: float = 3e-3
    min_lr: float = 1e-5
    warmup_epochs: int = 5
    weight_decay: float = 1e-5
    beta_1: float = 0.9
    beta_2: float = 0.999
    label_smoothing: float = 0.0
    augmentation: Dict[str, Any] = field(default_factory=lambda: dict(AUG))
    mixup_alpha: float = MIXUP_ALPHA
    cutmix_alpha: float = CUTMIX_ALPHA
    mixup_probability: float = MIX_PROB
    cutmix_probability: float = CUTMIX_PROB
    model: Dict[str, Any] = field(default_factory=lambda: {
        "stem_channels": 64,
        "stage_channels": [64, 128, 256],
        "stage_depths": [2, 3, 4],
        "dropout_rate": 0.3,
    })

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
