import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .artifacts import file_sha256
from .masks import decode_counts


def write_submission(path: Path, predictions: dict[str, list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filament_id", "segmentation_rle"])
        for filename, instances in sorted(predictions.items()):
            for index, instance in enumerate(instances, start=1):
                writer.writerow([f"{Path(filename).stem}_{index}", instance["counts"]])


def audit_submission(
    path: Path, expected_names: set[str], shape: tuple[int, int] = (2048, 2048)
) -> dict:
    expected_ids = {Path(name).stem for name in expected_names}
    masks_by_image = defaultdict(list)
    seen_ids = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Submission columns must be exactly filament_id,segmentation_rle")
        for row in reader:
            identifier, counts = row["filament_id"], row["segmentation_rle"]
            if set(row) != {"filament_id", "segmentation_rle"} or counts is None:
                raise ValueError("Malformed CSV row")
            if identifier in seen_ids:
                raise ValueError(f"Duplicate filament ID: {identifier}")
            seen_ids.add(identifier)
            image_id, separator, suffix = identifier.rpartition("_")
            if (
                not separator
                or re.fullmatch(r"[1-9]\d*", suffix) is None
                or image_id not in expected_ids
            ):
                raise ValueError(f"Unknown or malformed filament ID: {identifier}")
            masks_by_image[image_id].append(counts)
    for image_id, counts_list in masks_by_image.items():
        occupied = np.zeros(shape, dtype=bool)
        for counts in counts_list:
            mask = decode_counts(counts, shape).astype(bool)
            if not mask.any():
                raise ValueError(f"Empty mask in image: {image_id}")
            if (occupied & mask).any():
                raise ValueError(f"Overlapping instance masks in image: {image_id}")
            occupied |= mask
    missing = sorted(expected_ids - set(masks_by_image))
    return {
        "schema_and_masks": "PASS",
        "csv_sha256": file_sha256(path),
        "filament_rows": len(seen_ids),
        "images_with_predictions": len(masks_by_image),
        "expected_images": len(expected_ids),
        "zero_detection_images": missing,
        "coverage_status": "PASS"
        if not missing
        else "BLOCK: confirm zero-detection image handling",
        "kaggle_acceptance": "UNVERIFIED",
    }
