from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch import nn

from .artifacts import file_sha256
from .config import TrainingConfig


def build_network(config: TrainingConfig, *, initialize_pretrained: bool = True) -> nn.Module:
    extra = {}
    if (
        initialize_pretrained
        and config.encoder_weights is not None
        and config.encoder == "tu-convnext_tiny.fb_in1k"
    ):
        # Prefer the locally pinned ImageNet backbone when present (training-time
        # provenance); otherwise fall back to the standard timm/HF download so a
        # fresh clone works on Kaggle or any other machine.
        checkpoint = Path("data/pretrained/convnext-tiny-fb-in1k/pytorch_model.bin")
        if checkpoint.exists():
            if (
                file_sha256(checkpoint)
                != "4bee155ff60a2ee5415f777a381068ebb2a658fc6c7cbb6a33aca13426fb4105"
            ):
                raise ValueError("Pinned ImageNet checkpoint identity differs")
            extra["pretrained_cfg_overlay"] = {"file": str(checkpoint), "hf_hub_id": None, "url": None}
    return smp.Unet(
        encoder_name=config.encoder,
        encoder_weights=config.encoder_weights if initialize_pretrained else None,
        in_channels=1,
        classes=config.output_channels,
        decoder_channels=(128, 64, 32, 16, 16),
        **extra,
    )


def segmentation_loss(
    logits: torch.Tensor, targets: torch.Tensor, boundary_weight: float = 0.0
) -> torch.Tensor:
    if logits.shape != targets.shape or logits.ndim != 4:
        raise ValueError("Logits and targets must have identical NCHW dimensions")
    foreground_logits = logits[:, :1].float()
    truth = targets[:, :1].float()
    probabilities = foreground_logits.sigmoid()
    intersection = (probabilities * truth).sum(dim=(-2, -1))
    denominator = probabilities.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1))
    dice = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    loss = nn.functional.binary_cross_entropy_with_logits(foreground_logits, truth) + dice
    if boundary_weight:
        if logits.shape[1] != 2:
            raise ValueError("Boundary supervision requires a second output channel")
        loss += boundary_weight * nn.functional.binary_cross_entropy_with_logits(
            logits[:, 1:2].float(), targets[:, 1:2].float()
        )
    return loss
