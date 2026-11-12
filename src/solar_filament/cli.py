import argparse
import json
from pathlib import Path

from .artifacts import file_sha256, object_sha256, read_json, write_json
from .catalog import ANNOTATION_FILE, AnnotationCatalog, build_manifest, verify_manifest
from .config import PostprocessConfig, load_training_config
from .folds import make_folds, validate_folds


def load_context(project: Path, data_root: Path) -> tuple[AnnotationCatalog, dict, dict]:
    manifest = read_json(project / "state" / "dataset_manifest.json")
    folds = read_json(project / "state" / "folds.json")
    validate_folds(manifest, folds)
    catalog = AnnotationCatalog(data_root / "train" / ANNOTATION_FILE)
    verify_manifest(data_root, manifest, catalog)
    reference = read_json(project / "references" / "official_scorer.json")
    if (
        file_sha256(project / "references" / "official_self_evaluation.ipynb")
        != reference["sha256"]
    ):
        raise ValueError("The unmodified official notebook identity has changed")
    return catalog, manifest, folds


def main():
    parser = argparse.ArgumentParser(description="Audited solar filament experiments")
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--workers", type=int, default=4)
    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--name", required=True)
    for command in ["evaluate", "predict-test"]:
        evaluate = subparsers.add_parser(command)
        evaluate.add_argument("--checkpoint", type=Path, required=True)
        evaluate.add_argument("--name", required=True)
        evaluate.add_argument("--max-images", type=int, default=0)
        evaluate.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        evaluate.add_argument("--threshold", type=float, default=0.5)
        evaluate.add_argument("--min-area", type=int, default=32)
    subparsers.add_parser("verify")
    arguments = parser.parse_args()
    project = arguments.project.resolve()
    data_root = (
        arguments.data_root or project / "data" / "raw" / "MAGFiLO_1.0_Kaggle_2026"
    ).resolve()
    if arguments.command == "prepare":
        catalog = AnnotationCatalog(data_root / "train" / ANNOTATION_FILE)
        manifest = build_manifest(data_root, catalog, arguments.workers)
        folds = make_folds(manifest)
        validate_folds(manifest, folds)
        write_json(project / "state" / "dataset_manifest.json", manifest)
        write_json(project / "state" / "folds.json", folds)
        result = {
            "training_images": len(manifest["train"]),
            "test_images": len(manifest["test"]),
            "annotation_sets": manifest["annotation_sets"],
            "instances": manifest["instances"],
            "fold_sizes": folds["fold_sizes"],
            "manifest_sha256": object_sha256(manifest),
        }
        write_json(project / "state" / "data_audit.json", result)
    else:
        catalog, manifest, folds = load_context(project, data_root)
        if arguments.command == "verify":
            result = {"data_and_fold_identity": "PASS", "manifest_sha256": object_sha256(manifest)}
        elif arguments.command == "train":
            from .experiment import train_run

            result = train_run(
                project,
                data_root,
                catalog,
                manifest,
                folds,
                load_training_config(arguments.config),
                arguments.name,
            )
        else:
            from .evaluation import evaluate_run

            result = evaluate_run(
                project,
                data_root,
                catalog,
                manifest,
                folds,
                arguments.checkpoint,
                arguments.name,
                device_name=arguments.device,
                max_images=arguments.max_images,
                postprocess=PostprocessConfig(arguments.threshold, arguments.min_area),
                test=arguments.command == "predict-test",
            )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
