from dataclasses import asdict, dataclass

import numpy as np
from pycocotools import mask as coco_mask

from .masks import native_rle


@dataclass(frozen=True)
class PanopticTotals:
    iou_sum: float = 0.0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def __add__(self, other: "PanopticTotals") -> "PanopticTotals":
        return PanopticTotals(
            self.iou_sum + other.iou_sum,
            self.true_positives + other.true_positives,
            self.false_positives + other.false_positives,
            self.false_negatives + other.false_negatives,
        )

    @property
    def pq(self) -> float:
        denominator = self.true_positives + 0.5 * (self.false_positives + self.false_negatives)
        return self.iou_sum / denominator if denominator else 0.0

    def report(self) -> dict:
        denominator = self.true_positives + 0.5 * (self.false_positives + self.false_negatives)
        return {
            **asdict(self),
            "pq": self.pq,
            "sq": self.iou_sum / self.true_positives if self.true_positives else 0.0,
            "rq": self.true_positives / denominator if denominator else 0.0,
        }


def score_iou_matrix(iou: np.ndarray) -> PanopticTotals:
    iou = np.asarray(iou, dtype=np.float32)
    if iou.ndim != 2 or not np.isfinite(iou).all() or ((iou < 0) | (iou > 1)).any():
        raise ValueError("Expected a finite, two-dimensional IoU matrix in [0, 1]")
    # Preserve the published helper's all-pairs rule; do not replace it with Hungarian matching.
    hit = iou > 0.5
    return PanopticTotals(
        float(iou[hit].astype(np.float64).sum()),
        int(hit.sum()),
        int((hit.sum(axis=0) == 0).sum()),
        int((hit.sum(axis=1) == 0).sum()),
    )


def score_instances(ground_truth: list[dict], predicted: list[dict]) -> PanopticTotals:
    if not ground_truth or not predicted:
        return PanopticTotals(false_positives=len(predicted), false_negatives=len(ground_truth))
    iou = coco_mask.iou(
        [native_rle(x) for x in ground_truth],
        [native_rle(x) for x in predicted],
        [0] * len(predicted),
    )
    return score_iou_matrix(iou)
