from collections.abc import Iterator

import numpy as np
import torch
from scipy import ndimage
from torch import nn

from .config import PostprocessConfig
from .dataset import normalize_grayscale
from .masks import encode_mask


def tile_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length < 1 or tile_size < 1 or not 1 <= stride <= tile_size:
        raise ValueError("Invalid tiling geometry")
    positions = list(range(0, max(length - tile_size, 0) + 1, stride))
    last = max(length - tile_size, 0)
    if positions[-1] != last:
        positions.append(last)
    return positions


def batches(values: list, size: int) -> Iterator[list]:
    if size < 1:
        raise ValueError("Batch size must be positive")
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


@torch.inference_mode()
def tiled_probabilities(
    network: nn.Module,
    pixels: np.ndarray,
    *,
    tile_size: int,
    batch_size: int,
    device: torch.device,
    stride: int | None = None,
) -> np.ndarray:
    stride = tile_size // 2 if stride is None else stride
    normalized = normalize_grayscale(pixels)
    height, width = pixels.shape
    padded = np.pad(
        normalized,
        ((0, max(0, tile_size - height)), (0, max(0, tile_size - width))),
        constant_values=-2.0,
    )
    positions = [
        (y, x)
        for y in tile_positions(padded.shape[0], tile_size, stride)
        for x in tile_positions(padded.shape[1], tile_size, stride)
    ]
    window = np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype(np.float32)
    window = np.maximum(window, 0.01)
    weighted = None
    weights = np.zeros(padded.shape, dtype=np.float32)
    network.eval()
    for batch in batches(positions, batch_size):
        tiles = np.stack([padded[y : y + tile_size, x : x + tile_size] for y, x in batch])[:, None]
        tensor = torch.from_numpy(tiles).to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = network(tensor).float().sigmoid()
        probabilities = output.cpu().numpy()
        if probabilities.ndim != 4 or probabilities.shape[0] != len(batch):
            raise ValueError("Network output must have NCHW dimensions")
        if probabilities.shape[-2:] != (tile_size, tile_size):
            raise ValueError("Network changed tile dimensions")
        if not np.isfinite(probabilities).all():
            raise FloatingPointError("Inference produced non-finite probabilities")
        if weighted is None:
            weighted = np.zeros((probabilities.shape[1], *padded.shape), dtype=np.float32)
        for (y, x), probability in zip(batch, probabilities, strict=True):
            weighted[:, y : y + tile_size, x : x + tile_size] += probability * window
            weights[y : y + tile_size, x : x + tile_size] += window
    if weighted is None or (weights <= 0).any():
        raise ValueError("Inference did not cover every image pixel")
    result = weighted[:, :height, :width] / weights[None, :height, :width]
    return np.clip(result, 0, 1)


def extract_instances(probabilities: np.ndarray, config: PostprocessConfig) -> list[dict]:
    if probabilities.ndim != 2 or not np.isfinite(probabilities).all():
        raise ValueError("Expected a finite, two-dimensional foreground probability map")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Foreground probabilities must be in [0, 1]")
    labels, count = ndimage.label(probabilities >= config.threshold, structure=np.ones((3, 3)))
    if not count:
        return []
    areas = np.bincount(labels.ravel())
    retained = np.flatnonzero(areas[1:] >= config.min_area_pixels) + 1
    instances = []
    for label in retained:
        binary = labels == label
        encoded = encode_mask(binary)
        instances.append({**encoded, "confidence": float(probabilities[binary].mean())})
    return sorted(instances, key=lambda instance: -instance["confidence"])
