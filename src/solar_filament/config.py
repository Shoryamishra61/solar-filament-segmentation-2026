from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

from .artifacts import read_json


@dataclass(frozen=True)
class TrainingConfig:
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    crop_size: int = 512
    batch_size: int = 2
    learning_rate: float = 0.0002
    weight_decay: float = 0.0001
    max_steps: int = 256
    seed: int = 2026
    validation_fold: int = 1
    embargo_hours: float = 72.0
    boundary_weight: float = 0.0
    device: str = "cuda"
    fixed_crops: int = 0
    augment: bool = True
    positive_crop_probability: float = 0.7

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Seed must be an integer in [0, 2**32)")
        if type(self.augment) is not bool:
            raise ValueError("augment must be a boolean")
        if self.encoder not in {"resnet18", "resnet34", "tu-convnext_tiny.fb_in1k"}:
            raise ValueError("Only audited ResNet18/34 encoder recipes are supported")
        if self.encoder_weights not in {None, "imagenet"}:
            raise ValueError("Only random initialization or generic ImageNet weights are permitted")
        if type(self.crop_size) is not int or not 128 <= self.crop_size <= 1024:
            raise ValueError("Crop size must be an integer between 128 and 1024")
        if self.crop_size % 32:
            raise ValueError("Crop size must be divisible by 32")
        for name in ["batch_size", "max_steps"]:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size > 16 or self.max_steps > 100_000:
            raise ValueError("Training exceeds the bounded local recipe")
        for name in ["learning_rate", "weight_decay", "embargo_hours", "boundary_weight"]:
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.learning_rate <= 0.1:
            raise ValueError("Learning rate must be in (0, 0.1]")
        if not 0 <= self.positive_crop_probability <= 1:
            raise ValueError("Positive crop probability must be in [0, 1]")
        if type(self.fixed_crops) is not int or not 0 <= self.fixed_crops <= 1024:
            raise ValueError("fixed_crops must be an integer in [0, 1024]")
        if self.fixed_crops and self.augment:
            raise ValueError("Tiny-overfit runs must disable augmentation")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("Device must be explicitly cpu or cuda")

    @property
    def output_channels(self) -> int:
        return 2 if self.boundary_weight > 0 else 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PostprocessConfig:
    threshold: float = 0.5
    min_area_pixels: int = 32

    def __post_init__(self):
        if not isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("Foreground threshold must be in (0, 1)")
        if type(self.min_area_pixels) is not int or self.min_area_pixels < 1:
            raise ValueError("Minimum area must be a positive integer")


def load_training_config(path: Path) -> TrainingConfig:
    return TrainingConfig(**read_json(path))
