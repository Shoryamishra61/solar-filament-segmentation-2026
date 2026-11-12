"""Forensic reproduction of the historical 0.56 submission (champion-064).

AUDIT TOOL — NOT the competition submission pipeline. This script documents,
and regenerates byte-exactly, how the submission that scored public PQ 0.56
on 2026-10-21 (Kaggle ref 57468408) was originally produced:

    public notebook lamhuy8904/solar-filament-unet-segmentation-0-55
      -> embedded CHAMPION_PAYLOAD (1342 RLE masks over 178 test stems)
      -> + 3 gap-fill rows from our own Cosine95 model (one missing stem)
      -> adaptive half-maximum absorption-depth boundary refinement (alpha=0.25)
      -> 1345 rows over 179 stems; 20180507140210Th intentionally has no row.

The regenerated CSV is byte-identical to the scored artifact:
SHA-256 e17c2e26655ff6200807181d4ea69f4c3b6884fd60cfdbad01a20bb9423ad0c6
(verified 2026-11-13; see evidence/REPRODUCTION_REPORT.md).

Integrity note: the embedded payload is third-party data of unverified
provenance; against the public MAGFiLO annotations that overlap the test set
it scores PQ ~0.65, far above every honest pipeline measured in this project
(~0.37 public). It is reproduced here strictly as an audit artifact and was
never used as training signal or as the final-submission pipeline.

Usage:
    python audit/recover_champion064.py --notebook <pulled .ipynb> \
        --test-images <dir with 180 JPEGs> \
        --gapfill <gapfill_predictions.json> --out <output dir>

`kaggle kernels pull lamhuy8904/solar-filament-unet-segmentation-0-55`
retrieves the notebook. gapfill JSON: {"<stem>.jpeg": [{"size": [...],
"counts": "...", "confidence": ...}, ...]} for stem 20160513133714Th —
see audit/gapfill_regeneration.py for how those rows are re-derived from
checkpoint cosine95_long_fold1_16384.pt (SHA-256 671ebe63...).
"""
import argparse
import ast
import base64
import hashlib
import io
import json
import lzma
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

ALPHA = 0.25
TARGET_SHA256 = "e17c2e26655ff6200807181d4ea69f4c3b6884fd60cfdbad01a20bb9423ad0c6"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def extract_payload(notebook: Path) -> pd.DataFrame:
    """Cell 5 of the public notebook embeds CHAMPION_PAYLOAD (base85 + LZMA)."""
    data = json.loads(notebook.read_text(encoding="utf-8"))
    cell5 = "".join(data["cells"][5]["source"])
    start = cell5.find("CHAMPION_PAYLOAD = (\n") + len("CHAMPION_PAYLOAD = (\n")
    end = cell5.find("\n)\n", start)
    payload_str = ast.literal_eval("(" + cell5[start:end] + ")")
    raw = lzma.decompress(base64.b85decode(payload_str.encode("ascii")))
    return pd.read_csv(io.BytesIO(raw))


def assemble_champion055(payload: pd.DataFrame, gapfill: dict, test_images: Path) -> pd.DataFrame:
    """audit_055_candidate.py logic: payload rows + Cosine95 gap fill."""
    stems = sorted(p.stem for p in test_images.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    by_stem = defaultdict(list)
    for _, row in payload.iterrows():
        stem, _ = row["filament_id"].rsplit("_", 1)
        by_stem[stem].append(row["segmentation_rle"])
    cosine_by_stem = {
        Path(name).stem: [inst["counts"] for inst in instances]
        for name, instances in gapfill.items()
    }
    rows = []
    for stem in stems:
        if by_stem.get(stem):
            for i, rle in enumerate(by_stem[stem], 1):
                rows.append({"filament_id": f"{stem}_{i}", "segmentation_rle": rle})
        elif cosine_by_stem.get(stem):
            for i, rle in enumerate(cosine_by_stem[stem], 1):
                rows.append({"filament_id": f"{stem}_{i}", "segmentation_rle": rle})
    return pd.DataFrame(rows)


def adaptive_boundary_refine(champion055: pd.DataFrame, test_images: Path, alpha: float) -> pd.DataFrame:
    """build_adaptive_physics_champion.py algorithm, verbatim."""
    stems = sorted(p.stem for p in test_images.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
    by_stem = defaultdict(list)
    for _, row in champion055.iterrows():
        stem, _ = row["filament_id"].rsplit("_", 1)
        by_stem[stem].append(row["segmentation_rle"])

    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    refined = defaultdict(list)
    shaved_total = 0

    for stem in stems:
        rles = by_stem.get(stem, [])
        if not rles:
            continue
        img = cv2.imread(str(test_images / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            refined[stem].extend(rles)
            continue
        decoded, union = [], np.zeros((2048, 2048), dtype=bool)
        for rle in rles:
            m = mask_utils.decode({"size": [2048, 2048], "counts": str(rle).encode("ascii")}).astype(bool)
            decoded.append(m)
            union |= m
        occupied = np.zeros((2048, 2048), dtype=bool)
        for mask in decoded:
            orig_area = int(mask.sum())
            ys, xs = np.where(mask)
            pad = 16
            py0, py1 = max(0, ys.min() - pad), min(2048, ys.max() + pad + 1)
            px0, px1 = max(0, xs.min() - pad), min(2048, xs.max() + pad + 1)
            li, lm, lu = img[py0:py1, px0:px1], mask[py0:py1, px0:px1], union[py0:py1, px0:px1]
            inner = cv2.erode(lm.astype(np.uint8), k_erode).astype(bool)
            boundary = lm & ~inner
            dilated = cv2.dilate(lm.astype(np.uint8), k_ring).astype(bool)
            bg_ring = dilated & ~lu
            if bg_ring.sum() < 30:
                bg_ring = dilated & ~lm
            bg_mean = float(li[bg_ring].mean()) if bg_ring.any() else 128.0
            core = float(np.percentile(li[inner], 10)) if inner.any() else float(li[lm].mean())
            depth = max(6.0, bg_mean - core)
            thresh = bg_mean - max(3.0, alpha * depth)
            shaved = inner | (boundary & (li < thresh))
            if shaved.sum() < max(48, int(orig_area * 0.50)):
                shaved = lm
            full = np.zeros((2048, 2048), dtype=bool)
            full[py0:py1, px0:px1] = shaved
            clean = full & ~occupied
            area = int(clean.sum())
            if area > 0:
                occupied |= clean
                rle = mask_utils.encode(np.asfortranarray(clean.astype(np.uint8)))["counts"].decode("ascii")
                refined[stem].append(rle)
                shaved_total += orig_area - area
    rows = []
    for stem in stems:
        for idx, rle in enumerate(refined.get(stem, []), 1):
            rows.append({"filament_id": f"{stem}_{idx}", "segmentation_rle": rle})
    print(f"shaved halo pixels: {shaved_total:,} (era receipt: 255,289)")
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--gapfill", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = extract_payload(args.notebook)
    print(f"payload: {len(payload)} rows / {payload['filament_id'].str.rsplit('_', n=1).str[0].nunique()} stems")
    gapfill = json.loads(args.gapfill.read_text(encoding="utf-8"))
    champion055 = assemble_champion055(payload, gapfill, args.test_images)
    print(f"champion-055: {len(champion055)} rows")
    final = adaptive_boundary_refine(champion055, args.test_images, ALPHA)
    args.out.mkdir(parents=True, exist_ok=True)
    out_csv = args.out / "submission.csv"
    final.to_csv(out_csv, index=False)
    digest = sha256_file(out_csv)
    print(f"rows: {len(final)}  sha256: {digest}")
    if digest == TARGET_SHA256:
        print("BYTE MATCH with the scored 0.56 artifact (Kaggle ref 57468408)")
        return 0
    print("MISMATCH — investigate environment before trusting any output")
    return 1


if __name__ == "__main__":
    sys.exit(main())
