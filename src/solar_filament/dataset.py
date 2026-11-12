from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from scipy import ndimage
from torch.utils.data import Dataset

from .catalog import AnnotationCatalog
from .config import TrainingConfig


def normalize_grayscale(pixels: np.ndarray) -> np.ndarray:
    if pixels.ndim != 2 or pixels.dtype != np.uint8:
        raise ValueError("Input must be a two-dimensional uint8 grayscale image")
    return (pixels.astype(np.float32) / 255.0 - 0.5) / 0.25


def save_training_preview(crops: "TrainingCrops", path: Path) -> None:
    size = crops.config.crop_size
    count = min(4, len(crops))
    canvas = Image.new("RGB", (size * 3, size * count))
    for index in range(count):
        image, target = crops[index]
        pixels = np.clip((image[0].numpy() * 0.25 + 0.5) * 255, 0, 255).astype(np.uint8)
        rgb = np.repeat(pixels[:, :, None], 3, axis=2)
        foreground = target[0].numpy().astype(bool)
        overlay = rgb.copy()
        overlay[foreground] = (0.4 * rgb[foreground] + 0.6 * np.array([0, 255, 100])).astype(
            np.uint8
        )
        for column, view in enumerate([rgb, overlay, foreground.astype(np.uint8) * 255]):
            canvas.paste(Image.fromarray(view).convert("RGB"), (column * size, index * size))
    canvas.save(path)


@lru_cache(maxsize=12)
def read_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError(f"Expected grayscale image: {path}")
        result = np.asarray(image).copy()
    result.setflags(write=False)
    return result


class TrainingCrops(Dataset):
    def __init__(
        self,
        data_root: Path,
        catalog: AnnotationCatalog,
        training_names: list[str],
        config: TrainingConfig,
    ):
        self.image_root = data_root / "train" / "train_images"
        self.catalog = catalog
        self.names = tuple(sorted(training_names))
        if not self.names or not set(self.names) <= set(catalog.by_filename):
            raise ValueError("Training names must refer only to the training annotation catalog")
        self.config = config
        self._target_cache = OrderedDict()

    def __len__(self):
        return self.config.max_steps * self.config.batch_size

    def targets(self, entry: str) -> tuple[np.ndarray, np.ndarray]:
        if entry in self._target_cache:
            self._target_cache.move_to_end(entry)
            return self._target_cache[entry]
        foreground = np.zeros((2048, 2048), dtype=bool)
        boundaries = np.zeros_like(foreground)
        for encoded in self.catalog.rles(entry):
            binary = coco_mask.decode(encoded).astype(bool)
            foreground |= binary
            if self.config.output_channels == 2:
                boundaries |= ndimage.binary_dilation(binary) ^ ndimage.binary_erosion(binary)
        foreground.setflags(write=False)
        boundaries.setflags(write=False)
        self._target_cache[entry] = foreground, boundaries
        if len(self._target_cache) > 16:
            self._target_cache.popitem(last=False)
        return foreground, boundaries

    def __getitem__(self, index: int):
        if self.config.fixed_crops:
            index %= self.config.fixed_crops
        rng = np.random.default_rng(self.config.seed + index)
        name = self.names[int(rng.integers(len(self.names)))]
        entries = self.catalog.by_filename[name]
        entry = entries[int(rng.integers(len(entries)))]
        pixels = read_grayscale(self.image_root / name)
        foreground, boundaries = self.targets(entry)
        if rng.random() < self.config.positive_crop_probability and foreground.any():
            # Sample an instance before its vertices so large masks do not monopolize crops.
            instance = self.catalog.polygons[entry][
                int(rng.integers(len(self.catalog.polygons[entry])))
            ]
            polygon = instance[int(rng.integers(len(instance)))]
            vertex = int(rng.integers(len(polygon) // 2)) * 2
            center_x, center_y = polygon[vertex : vertex + 2]
        else:
            center_x, center_y = rng.uniform(128, 1920, size=2)
        size = self.config.crop_size
        left = int(np.clip(center_x - size / 2 + rng.uniform(-size / 4, size / 4), 0, 2048 - size))
        top = int(np.clip(center_y - size / 2 + rng.uniform(-size / 4, size / 4), 0, 2048 - size))
        image = pixels[top : top + size, left : left + size].copy()
        target = np.stack(
            [
                foreground[top : top + size, left : left + size],
                boundaries[top : top + size, left : left + size],
            ]
        )
        target = target[: self.config.output_channels]
        if self.config.augment:
            rotations = int(rng.integers(4))
            image, target = np.rot90(image, rotations), np.rot90(target, rotations, axes=(-2, -1))
            if rng.random() < 0.5:
                image, target = image[:, ::-1], target[:, :, ::-1]
        return (
            torch.from_numpy(normalize_grayscale(image)[None].copy()),
            torch.from_numpy(target.astype(np.float32).copy()),
        )
