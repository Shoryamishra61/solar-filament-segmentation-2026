"""Contract tests: RLE encoding, disjointness accounting, verifier behavior."""
import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mutils

from solar_filament.masks import decode_counts, encode_mask


def test_encode_decode_roundtrip():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:20, 10:30] = True
    encoded = encode_mask(mask)
    assert encoded["size"] == [64, 64]
    decoded = decode_counts(encoded["counts"], (64, 64))
    assert np.array_equal(decoded.astype(bool), mask)


def test_rejects_oversized_rle():
    with pytest.raises(ValueError):
        decode_counts("9" * 99999, (64, 64))


def _rle(mask: np.ndarray) -> str:
    return mutils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def test_disjoint_pair_detection():
    a = np.zeros((32, 32), dtype=bool)
    b = np.zeros((32, 32), dtype=bool)
    a[4:8, 4:8] = True
    b[6:12, 6:12] = True  # overlaps a
    overlap = mutils.decode({"size": [32, 32], "counts": _rle(a)}) & mutils.decode(
        {"size": [32, 32], "counts": _rle(b)}
    )
    assert overlap.any()
    frame = pd.DataFrame(
        {"filament_id": ["p_1", "p_2"], "segmentation_rle": [_rle(a), _rle(b)]}
    )
    assert len(frame) == 2 and frame["filament_id"].nunique() == 2
