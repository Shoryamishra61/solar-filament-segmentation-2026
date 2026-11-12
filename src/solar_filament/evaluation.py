import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .artifacts import create_run_directory, file_sha256, object_sha256, source_identity, write_json
from .catalog import AnnotationCatalog
from .config import PostprocessConfig
from .experiment import load_checkpoint
from .inference import extract_instances, tiled_probabilities
from .metric import PanopticTotals, score_instances
from .submission import audit_submission, write_submission


def evaluate_run(
    project: Path,
    data_root: Path,
    catalog: AnnotationCatalog,
    manifest: dict,
    folds: dict,
    checkpoint: Path,
    name: str,
    *,
    device_name: str = "cuda",
    max_images: int = 0,
    postprocess: PostprocessConfig | None = None,
    test: bool = False,
    tile_size: int | None = None,
) -> dict:
    if type(max_images) is not int or max_images < 0:
        raise ValueError("max_images must be a nonnegative integer")
    postprocess = postprocess or PostprocessConfig()
    network, metadata, device = load_checkpoint(checkpoint, manifest, folds, device_name)
    names = (
        [row["file_name"] for row in manifest["test"]] if test else metadata["validation_images"]
    )
    if not test and set(names) & set(metadata["training_images"]):
        raise ValueError("In-sample evaluation is prohibited")
    if not test and any(folds["assignments"][name] == folds["holdout_fold"] for name in names):
        raise ValueError("Development evaluation attempted to access the reserved holdout")
    all_names = sorted(names)
    names = all_names[:max_images] if max_images else all_names
    run = create_run_directory(project / "runs", name)
    probabilities_root = run / "probabilities"
    probabilities_root.mkdir()
    predicted_by_image = {}
    image_totals = []
    totals = PanopticTotals()
    started = time.perf_counter()
    for index, filename in enumerate(names, start=1):
        split = "test" if test else "train"
        with Image.open(data_root / split / f"{split}_images" / filename) as image:
            pixels = np.asarray(image)
        probabilities = tiled_probabilities(
            network,
            pixels,
            tile_size=tile_size or metadata["config"]["crop_size"],
            batch_size=metadata["config"]["batch_size"],
            device=device,
        )
        np.savez_compressed(
            probabilities_root / f"{Path(filename).stem}.npz", probabilities=probabilities
        )
        predicted = extract_instances(probabilities[0], postprocess)
        predicted_by_image[filename] = predicted
        per_image = PanopticTotals()
        if not test:
            for entry in catalog.by_filename[filename]:
                per_image += score_instances(catalog.rles(entry), predicted)
            totals += per_image
        image_totals.append(
            {"file_name": filename, **per_image.report(), "predicted_instances": len(predicted)}
        )
        if index == 1 or index % 10 == 0 or index == len(names):
            print(
                f"{'test' if test else 'validation'} {index}/{len(names)} "
                f"instances={len(predicted)} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    report = {
        "schema_version": 1,
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_run": metadata["run_name"],
        "training_source_sha256": metadata["source_sha256"],
        "evaluation_source_sha256": source_identity(),
        "manifest_sha256": object_sha256(manifest),
        "folds_sha256": object_sha256(folds),
        "postprocess": asdict(postprocess),
        "tile_size": tile_size or metadata["config"]["crop_size"],
        "split": "test" if test else "validation",
        "processed_images": len(names),
        "expected_images": len(all_names),
        "is_complete_split": len(names) == len(all_names),
        "evidence_class": "test-inference-only"
        if test
        else (
            "full-development-fold" if len(names) == len(all_names) else "partial-fold-diagnostic"
        ),
        "metrics": None if test else totals.report(),
        "per_image": image_totals,
        "elapsed_seconds": time.perf_counter() - started,
        "promotion_status": "BLOCK: no complete multi-fold and untouched-holdout comparison",
        "threshold_selection": "caller-supplied; calibration provenance not established",
    }
    if test:
        csv_path = run / "submission.csv"
        write_submission(csv_path, predicted_by_image)
        report["submission_audit"] = audit_submission(csv_path, set(all_names))
    write_json(run / "evaluation.json", report)
    write_json(run / "predictions.json", predicted_by_image)
    return report
