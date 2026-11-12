import numpy as np
from pycocotools import mask as coco_mask


def validate_rle_counts(counts: str, shape: tuple[int, int]) -> None:
    """Check compressed COCO runs before passing an untrusted string to native code."""
    if not isinstance(counts, str) or not counts:
        raise ValueError("RLE counts must be a non-empty string")
    height, width = shape
    if height <= 0 or width <= 0 or height * width > 2048 * 2048:
        raise ValueError("Invalid or unsupported mask dimensions")
    if len(counts) > 2 * height * width + 32:
        raise ValueError("RLE counts exceed the supported mask size")
    runs: list[int] = []
    cursor = 0
    total = 0
    while cursor < len(counts):
        value = 0
        shift = 0
        while True:
            if cursor >= len(counts) or shift > 30:
                raise ValueError("Truncated or oversized RLE integer")
            code = ord(counts[cursor]) - 48
            cursor += 1
            if not 0 <= code <= 63:
                raise ValueError("Invalid compressed RLE character")
            value |= (code & 31) << shift
            shift += 5
            if not code & 32:
                if code & 16:
                    value |= -1 << shift
                break
        if len(runs) > 2:
            value += runs[-2]
        if value < 0:
            raise ValueError("RLE contains a negative run")
        runs.append(value)
        total += value
        if total > height * width:
            raise ValueError("RLE exceeds image bounds")
    if total != height * width:
        raise ValueError("RLE does not cover the complete image")


def encode_mask(binary: np.ndarray) -> dict:
    if binary.ndim != 2 or (binary.dtype != np.bool_ and not np.isin(binary, [0, 1]).all()):
        raise ValueError("Expected a two-dimensional binary mask")
    encoded = coco_mask.encode(np.asfortranarray(binary, dtype=np.uint8))
    return {"size": list(binary.shape), "counts": encoded["counts"].decode("ascii")}


def decode_counts(counts: str, shape: tuple[int, int] = (2048, 2048)) -> np.ndarray:
    validate_rle_counts(counts, shape)
    return coco_mask.decode({"size": list(shape), "counts": counts.encode("ascii")})


def polygon_rle(polygons: list[list[float]], shape: tuple[int, int]) -> dict:
    if not isinstance(polygons, list) or not polygons:
        raise ValueError("Expected at least one COCO polygon")
    for polygon in polygons:
        coordinates = np.asarray(polygon, dtype=np.float64)
        if coordinates.ndim != 1 or len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError("A polygon requires at least three coordinate pairs")
        if not np.isfinite(coordinates).all():
            raise ValueError("Polygon coordinates must be finite")
        if (coordinates[0::2] < 0).any() or (coordinates[0::2] > shape[1]).any():
            raise ValueError("Polygon x coordinate is outside the image")
        if (coordinates[1::2] < 0).any() or (coordinates[1::2] > shape[0]).any():
            raise ValueError("Polygon y coordinate is outside the image")
    return coco_mask.merge(coco_mask.frPyObjects(polygons, *shape))


def native_rle(rle: dict) -> dict:
    counts = rle["counts"]
    return {
        "size": rle["size"],
        "counts": counts.encode("ascii") if isinstance(counts, str) else counts,
    }
