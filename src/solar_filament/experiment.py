import ast
import importlib.metadata
import json
import os
import random
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .artifacts import (
    create_run_directory,
    file_sha256,
    object_sha256,
    read_json,
    source_identity,
    write_json,
)
from .catalog import AnnotationCatalog
from .config import TrainingConfig
from .dataset import TrainingCrops, save_training_preview
from .folds import partition
from .network import build_network, segmentation_loss


def require_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use an audited CUDA environment")
    return torch.device(name)


def checkpoint_write(path: Path, payload: dict) -> None:
    pending = path.with_suffix(".pending")
    torch.save(payload, pending)
    os.replace(pending, path)


def optimizer_update(
    loss: torch.Tensor,
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict:
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite training loss")
    scale_before = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [parameter.grad for parameter in network.parameters() if parameter.grad is not None]
    if not gradients:
        raise RuntimeError("Training produced no gradients")
    finite = bool(torch.stack([torch.isfinite(gradient).all() for gradient in gradients]).all())
    if not finite:
        if not scaler.is_enabled():
            raise FloatingPointError("Non-finite gradients without an enabled loss scaler")
        # unscale_ recorded the overflow. GradScaler skips the optimizer and lowers its scale.
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            raise FloatingPointError("AMP overflow did not reduce the loss scale")
        return {
            "gradient_norm": None,
            "optimizer_stepped": False,
            "amp_overflow": True,
            "loss_scale": scaler.get_scale(),
        }
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        network.parameters(), max_norm=5.0, error_if_nonfinite=True
    )
    scaler.step(optimizer)
    scaler.update()
    return {
        "gradient_norm": float(gradient_norm),
        "optimizer_stepped": True,
        "amp_overflow": False,
        "loss_scale": scaler.get_scale(),
    }


def verify_training_snapshot(
    snapshot: Path, expected_sha256: str, current: Path | None = None
) -> None:
    current = current or Path(__file__).parent
    if source_identity(snapshot) != expected_sha256:
        raise ValueError("Training source snapshot identity differs from the checkpoint")
    # Post-processing may evolve; the trained architecture and input/target contract may not.
    for name in ["config.py", "dataset.py", "network.py", "catalog.py", "folds.py"]:
        if file_sha256(snapshot / name) != file_sha256(current / name):
            raise ValueError(f"Model or preprocessing source is incompatible: {name}")

    def polygon_contract(directory: Path) -> str:
        tree = ast.parse((directory / "masks.py").read_text(encoding="utf-8"))
        matches = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "polygon_rle"
        ]
        if len(matches) != 1:
            raise ValueError("The target polygon contract is missing or ambiguous")
        return ast.dump(matches[0], include_attributes=False)

    if polygon_contract(snapshot) != polygon_contract(current):
        raise ValueError("Target polygon rasterization differs from the training contract")


@torch.inference_mode()
def fixed_crop_loss(network, crops: TrainingCrops, device: torch.device) -> float:
    network.eval()
    losses = []
    for offset in range(0, crops.config.fixed_crops, crops.config.batch_size):
        samples = [
            crops[i]
            for i in range(offset, min(offset + crops.config.batch_size, crops.config.fixed_crops))
        ]
        images = torch.stack([x for x, _ in samples]).to(device)
        targets = torch.stack([y for _, y in samples]).to(device)
        losses.append(
            float(segmentation_loss(network(images), targets, crops.config.boundary_weight))
        )
    network.train()
    return float(np.mean(losses))


def train_run(
    project: Path,
    data_root: Path,
    catalog: AnnotationCatalog,
    manifest: dict,
    folds: dict,
    config: TrainingConfig,
    name: str,
) -> dict:
    if config.validation_fold == folds["holdout_fold"]:
        raise ValueError("The reserved holdout cannot be used for development training runs")
    training_names, validation_names = partition(
        manifest, folds, config.validation_fold, config.embargo_hours
    )
    device = require_device(config.device)
    torch.set_num_threads(min(os.cpu_count() or 1, 4))
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    metadata = {
        "schema_version": 1,
        "run_name": name,
        "status": "RUNNING",
        "config": config.to_dict(),
        "config_sha256": object_sha256(config.to_dict()),
        "manifest_sha256": object_sha256(manifest),
        "folds_sha256": object_sha256(folds),
        "source_sha256": source_identity(),
        "official_scorer_sha256": read_json(project / "references" / "official_scorer.json")[
            "sha256"
        ],
        "annotation_sha256": catalog.annotation_sha256,
        "training_images": training_names,
        "validation_images": validation_names,
        "evidence_class": "tiny-overfit-only" if config.fixed_crops else "bounded-training-run",
        "versions": {
            package: importlib.metadata.version(package)
            for package in [
                "torch",
                "torchvision",
                "numpy",
                "pycocotools",
                "segmentation-models-pytorch",
            ]
        },
        "pretraining": "ImageNet-only encoder"
        if config.encoder_weights
        else "random initialization",
    }
    for previous in sorted((project / "runs").glob("*/run.json")):
        prior = read_json(previous)
        identity_keys = ["config_sha256", "manifest_sha256", "folds_sha256", "source_sha256"]
        if prior.get("status") == "COMPLETE" and all(
            prior.get(key) == metadata[key] for key in identity_keys
        ):
            raise FileExistsError(f"An identical completed experiment already exists: {previous}")
    run = create_run_directory(project / "runs", name)
    write_json(run / "run.json", metadata)
    shutil.copytree(
        Path(__file__).parent, run / "source_snapshot", ignore=shutil.ignore_patterns("__pycache__")
    )
    started = time.perf_counter()
    try:
        network = build_network(config).to(device)
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_steps, eta_min=config.learning_rate * 0.05
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        crops = TrainingCrops(data_root, catalog, training_names, config)
        save_training_preview(crops, run / "training_preview.png")
        loader = DataLoader(crops, batch_size=config.batch_size, shuffle=False, num_workers=0)
        initial_fixed_loss = fixed_crop_loss(network, crops, device) if config.fixed_crops else None
        network.train()
        losses = []
        overflow_batches = 0
        consecutive_overflows = 0
        maximum_overflows = max(4, config.max_steps // 100)
        with (run / "training.jsonl").open("x", encoding="utf-8") as stream:
            for step, (images, targets) in enumerate(loader, start=1):
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = network(images)
                    loss = segmentation_loss(logits, targets, config.boundary_weight)
                learning_rate = optimizer.param_groups[0]["lr"]
                update = optimizer_update(loss, network, optimizer, scaler)
                if update["optimizer_stepped"]:
                    scheduler.step()
                overflow_batches += int(update["amp_overflow"])
                consecutive_overflows = consecutive_overflows + 1 if update["amp_overflow"] else 0
                loss_value = float(loss.detach())
                losses.append(loss_value)
                record = {
                    "step": step,
                    "loss": loss_value,
                    "learning_rate": learning_rate,
                    **update,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                if consecutive_overflows > 3 or overflow_batches > maximum_overflows:
                    raise FloatingPointError(f"AMP overflow budget exceeded at batch {step}")
                if step == 1 or step % 32 == 0 or step == config.max_steps:
                    print(json.dumps(record), flush=True)
                if step % 64 == 0 or step == config.max_steps:
                    checkpoint_write(
                        run / "checkpoint.pt",
                        {
                            "model_state": network.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "scaler_state": scaler.state_dict(),
                            "torch_rng_state": torch.get_rng_state(),
                            "step": step,
                            "metadata": metadata,
                        },
                    )
        final_fixed_loss = fixed_crop_loss(network, crops, device) if config.fixed_crops else None
        overfit_pass = (final_fixed_loss < initial_fixed_loss * 0.8) if config.fixed_crops else None
        metadata.update(
            {
                "status": "COMPLETE",
                "steps": len(losses),
                "optimizer_updates": len(losses) - overflow_batches,
                "amp_overflow_batches": overflow_batches,
                "initial_fixed_crop_loss": initial_fixed_loss,
                "final_fixed_crop_loss": final_fixed_loss,
                "tiny_overfit_pass": overfit_pass,
                "first_16_steps_mean_loss": float(np.mean(losses[:16])),
                "last_16_steps_mean_loss": float(np.mean(losses[-16:])),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()
                if device.type == "cuda"
                else 0,
                "checkpoint_sha256": file_sha256(run / "checkpoint.pt"),
                "promotion_status": "BLOCK: held-out validation and submission acceptance missing",
            }
        )
        write_json(run / "run.json", metadata, replace=True)
        if config.fixed_crops and not overfit_pass:
            raise RuntimeError(
                "Tiny-subset learning gate failed; inspect the saved losses before scaling"
            )
        return metadata
    except Exception as error:
        metadata.update(
            {
                "status": "FAILED",
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        write_json(run / "run.json", metadata, replace=True)
        (run / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def load_checkpoint(
    path: Path, manifest: dict, folds: dict, device_name: str
) -> tuple[torch.nn.Module, dict, torch.device]:
    device = require_device(device_name)
    run_document = json.loads((path.parent / "run.json").read_text(encoding="utf-8"))
    if run_document["status"] != "COMPLETE":
        raise ValueError("Only completed runs may be evaluated")
    if file_sha256(path) != run_document["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 differs from the completed run")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if metadata["source_sha256"] != run_document["source_sha256"]:
        raise ValueError("Training source identity differs between checkpoint and run record")
    verify_training_snapshot(path.parent / "source_snapshot", metadata["source_sha256"])
    active_reference = read_json(path.parents[2] / "references" / "official_scorer.json")
    if metadata["official_scorer_sha256"] != active_reference["sha256"]:
        raise ValueError("Checkpoint official scorer identity differs from the active contract")
    for key, actual in [
        ("manifest_sha256", object_sha256(manifest)),
        ("folds_sha256", object_sha256(folds)),
    ]:
        if metadata[key] != actual or run_document[key] != actual:
            raise ValueError(f"Checkpoint {key} differs from the active contract")
    if object_sha256(metadata["config"]) != metadata["config_sha256"]:
        raise ValueError("Checkpoint configuration identity is invalid")
    config = TrainingConfig(**metadata["config"])
    expected_training, expected_validation = partition(
        manifest, folds, config.validation_fold, config.embargo_hours
    )
    if (
        metadata["training_images"] != expected_training
        or metadata["validation_images"] != expected_validation
    ):
        raise ValueError("Checkpoint training/validation membership is invalid")
    network = build_network(config, initialize_pretrained=False)
    network.load_state_dict(payload["model_state"], strict=True)
    return network.to(device).eval(), metadata, device
