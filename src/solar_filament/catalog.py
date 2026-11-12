import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from .artifacts import file_sha256, read_json
from .masks import polygon_rle

COMPETITION = "filament-segmentation-2026"
ANNOTATION_FILE = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_NAME = re.compile(r"\d{14}(?:Bh|Ch|Lh|Mh|Th|Uh)\.jpeg")


class AnnotationCatalog:
    """Index only the competition training polygons; extra ground-truth metadata is unused."""

    def __init__(self, annotation_path: Path):
        document = read_json(annotation_path)
        self.annotation_sha256 = file_sha256(annotation_path)
        self.entries: dict[str, dict] = {}
        self.by_filename: dict[str, list[str]] = {}
        self.polygons: dict[str, list[list[list[float]]]] = {}
        for image in document["images"]:
            identifier, filename = image["id"], image["file_name"]
            if not isinstance(identifier, str) or identifier in self.entries:
                raise ValueError("Annotation image IDs must be unique strings")
            if not isinstance(filename, str) or IMAGE_NAME.fullmatch(filename) is None:
                raise ValueError(f"Invalid observation filename: {filename!r}")
            if "-" not in identifier or identifier.split("-", maxsplit=1)[1] != Path(filename).stem:
                raise ValueError("Annotator-image ID does not match its photograph filename")
            if (image["height"], image["width"]) != (2048, 2048):
                raise ValueError(f"Unexpected image dimensions: {filename}")
            self.entries[identifier] = {"file_name": filename}
            self.by_filename.setdefault(filename, []).append(identifier)
            self.polygons[identifier] = []
        seen = set()
        for annotation in document["annotations"]:
            identifier = annotation["id"]
            if not isinstance(identifier, str) or identifier in seen:
                raise ValueError("Instance annotation IDs must be unique strings")
            seen.add(identifier)
            entry = annotation["image_id"]
            if entry not in self.entries:
                raise ValueError(f"Instance refers to an unknown image ID: {entry}")
            polygons = annotation["segmentation"]
            encoded = polygon_rle(polygons, (2048, 2048))
            # Rasterized zero-area polygons cannot supply a usable target.
            if coco_mask.area(encoded) <= 0:
                raise ValueError(f"Empty instance polygon: {identifier}")
            self.polygons[entry].append(polygons)

    def rles(self, entry: str) -> list[dict]:
        return [polygon_rle(polygon, (2048, 2048)) for polygon in self.polygons[entry]]


def inspect_image(path: Path) -> dict:
    if IMAGE_NAME.fullmatch(path.name) is None:
        raise ValueError(f"Unexpected image filename: {path.name}")
    captured = datetime.strptime(path.name[:14], "%Y%m%d%H%M%S")
    with Image.open(path) as image:
        if image.size != (2048, 2048) or image.mode != "L":
            raise ValueError(f"Expected a 2048x2048 grayscale JPEG: {path}")
        pixels = np.asarray(image)
        pixel_hash = hashlib.sha256(pixels.tobytes()).hexdigest()
    return {
        "file_name": path.name,
        "sha256": file_sha256(path),
        "pixel_sha256": pixel_hash,
        "captured": captured.isoformat(),
        "site": path.stem[14:],
        "bytes": path.stat().st_size,
    }


def build_manifest(data_root: Path, catalog: AnnotationCatalog, workers: int = 4) -> dict:
    if not 1 <= workers <= 16:
        raise ValueError("Image inspection workers must be between 1 and 16")
    result = {
        "schema_version": 1,
        "competition": COMPETITION,
        "annotation_sha256": catalog.annotation_sha256,
    }
    for split in ["train", "test"]:
        paths = sorted((data_root / split / f"{split}_images").glob("*.jpeg"))
        if not paths:
            raise ValueError(f"No {split} JPEGs found")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            result[split] = list(executor.map(inspect_image, paths))
    train_names = {row["file_name"] for row in result["train"]}
    if train_names != set(catalog.by_filename):
        raise ValueError("Training JPEGs and annotation filenames disagree")
    test_names = {row["file_name"] for row in result["test"]}
    if train_names & test_names:
        raise ValueError("A filename occurs in training and test data")
    train_hashes = {row["pixel_sha256"] for row in result["train"]}
    if train_hashes & {row["pixel_sha256"] for row in result["test"]}:
        raise ValueError("Decoded pixels overlap between training and test data")
    result["annotation_sets"] = len(catalog.entries)
    result["instances"] = sum(len(x) for x in catalog.polygons.values())
    return result


def verify_manifest(data_root: Path, manifest: dict, catalog: AnnotationCatalog) -> None:
    if manifest["annotation_sha256"] != catalog.annotation_sha256:
        raise ValueError("Annotation SHA-256 differs from the frozen manifest")
    for split in ["train", "test"]:
        directory = data_root / split / f"{split}_images"
        expected = {row["file_name"] for row in manifest[split]}
        if {p.name for p in directory.glob("*.jpeg")} != expected:
            raise ValueError(f"{split} file inventory changed")
        for row in manifest[split]:
            if file_sha256(directory / row["file_name"]) != row["sha256"]:
                raise ValueError(f"Image SHA-256 changed: {row['file_name']}")
