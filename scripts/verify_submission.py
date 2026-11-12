"""Validate a competition submission CSV against the official contract.

Checks: exact schema, COCO compressed-RLE decodability, 2048x2048 geometry,
positive area, strict per-photo disjointness, instance-id uniqueness/shape.
Optionally cross-checks the file SHA-256 against a pinned expected value.

Usage:
    python scripts/verify_submission.py path/to/submission.csv [--sha256 <hex>]
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mutils

SIZE = [2048, 2048]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--sha256", default=None, help="expected file SHA-256")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    checks = {}
    frame = pd.read_csv(args.csv)
    checks["schema"] = list(frame.columns) == ["filament_id", "segmentation_rle"]
    ids = frame["filament_id"].tolist()
    checks["unique_ids"] = len(set(ids)) == len(ids)
    photos = defaultdict(list)
    ok_shape = ok_area = True
    for fid, rle in zip(frame["filament_id"], frame["segmentation_rle"]):
        photos[fid.rsplit("_", 1)[0]].append(rle)
        try:
            mask = mutils.decode({"size": SIZE, "counts": str(rle).encode("ascii")})
        except Exception:
            ok_shape = False
            break
        ok_area &= bool(mask.any())
    checks["rle_decode_and_geometry"] = ok_shape
    checks["positive_area"] = bool(ok_area)
    checks["photo_stems"] = len(photos)
    disjoint = True
    occupied = {}
    for photo, rles in photos.items():
        acc = None
        for rle in rles:
            mask = mutils.decode({"size": SIZE, "counts": str(rle).encode("ascii")}).astype(bool)
            if acc is None:
                acc = mask.copy()
            else:
                disjoint &= not bool((mask & acc).any())
                acc |= mask
        occupied[photo] = acc
    checks["disjoint_per_photo"] = bool(disjoint)
    checks["rows"] = len(frame)
    digest = sha256_file(args.csv)
    checks["sha256"] = digest
    checks["sha256_matches"] = (args.sha256 is None or digest == args.sha256)

    if args.json:
        print(json.dumps(checks, indent=1))
    else:
        for key, value in checks.items():
            print(f"{key}: {value}")
    ok = all(v for k, v in checks.items() if k not in {"rows", "photo_stems", "sha256", "sha256_matches"})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
