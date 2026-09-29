#!/usr/bin/env python3
"""
Run an instance and a panoptic segmentation model over the natural-scene demo images.

Intended to be run on a GPU machine (e.g. Lightning AI); the outputs are copied back and
scored locally by ``notebooks/coco_natural_scenes.py``. Only the JPGs are needed on the
GPU machine, no ground truth.

Models:
    yolo26       Ultralytics YOLO26 instance segmentation (``yolo26x-seg.pt``), COCO 80
                 "thing" classes. Masks may overlap.
    mask2former  ``facebook/mask2former-swin-large-coco-panoptic`` (HF transformers),
                 COCO panoptic 133 classes (things + stuff). Masks do not overlap.

Output, one file per model and image:
    outputs/coco_demo/<model>/<image_id>.npz
        masks   (N, H, W) bool, at the original image resolution
        labels  (N,) str, class names as reported by the model
        scores  (N,) float32

Setup (on top of the repo's `uv sync`; see pyproject.toml for the CUDA torch reinstall):
    uv pip install ultralytics

Usage:
    python scripts/run_coco_panoptic_models.py
    python scripts/run_coco_panoptic_models.py --models mask2former --device cpu
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

YOLO_WEIGHTS = "yolo26x-seg.pt"
MASK2FORMER_MODEL = "facebook/mask2former-swin-large-coco-panoptic"


def image_id(path: Path) -> str:
    """``coco_elephant_325871.jpg`` -> ``325871``."""
    return path.stem.split("_")[-1]


def save(out_dir: Path, path: Path, masks, labels, scores):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_id(path)}.npz"
    np.savez_compressed(
        out_path,
        masks=np.asarray(masks, dtype=bool),
        labels=np.asarray(labels, dtype=str),
        scores=np.asarray(scores, dtype=np.float32),
    )
    logger.info(f"  {path.name}: {len(labels)} segments -> {out_path}")


def run_yolo(image_paths, out_dir: Path, device: str, conf: float):
    from ultralytics import YOLO

    model = YOLO(YOLO_WEIGHTS)
    for path in image_paths:
        w, h = Image.open(path).size
        # retina_masks returns masks at the original image resolution rather than the
        # letterboxed network resolution.
        result = model.predict(
            str(path), conf=conf, device=device, retina_masks=True, verbose=False
        )[0]
        if result.masks is None:
            save(out_dir, path, np.zeros((0, h, w), dtype=bool), [], [])
            continue
        masks = result.masks.data.cpu().numpy() > 0.5
        assert masks.shape[1:] == (h, w), f"mask shape {masks.shape} != image {(h, w)}"
        labels = [result.names[int(c)] for c in result.boxes.cls.cpu().numpy()]
        save(out_dir, path, masks, labels, result.boxes.conf.cpu().numpy())


def run_mask2former(image_paths, out_dir: Path, device: str, conf: float):
    import torch
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    processor = AutoImageProcessor.from_pretrained(MASK2FORMER_MODEL)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MASK2FORMER_MODEL)
    model = model.to(device).eval()
    id2label = model.config.id2label

    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_panoptic_segmentation(
            outputs, threshold=conf, target_sizes=[image.size[::-1]]
        )[0]
        seg_map = result["segmentation"].cpu().numpy()
        segments = result["segments_info"]
        masks = [seg_map == s["id"] for s in segments]
        if not masks:
            masks = np.zeros((0, *seg_map.shape), dtype=bool)
        labels = [id2label[s["label_id"]] for s in segments]
        scores = [s["score"] for s in segments]
        save(out_dir, path, masks, labels, scores)


RUNNERS = {"yolo26": run_yolo, "mask2former": run_mask2former}
# Each model's own default confidence threshold.
DEFAULT_CONF = {"yolo26": 0.25, "mask2former": 0.8}


def main():
    parser = argparse.ArgumentParser(description="Run segmentation models on the COCO demo")
    parser.add_argument(
        "--images",
        type=str,
        default="data/coco_data",
        help="Folder of demo JPGs named <anything>_<image_id>.jpg (default: data/coco_data)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/coco_demo",
        help="Output root; one sub-folder per model (default: outputs/coco_demo)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=list(RUNNERS),
        choices=list(RUNNERS),
        help="Models to run (default: all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (default: cuda)",
    )
    args = parser.parse_args()

    image_paths = sorted(Path(args.images).glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found in {args.images}")
    logger.info(f"Found {len(image_paths)} images")

    for name in args.models:
        logger.info(f"Running {name} ...")
        RUNNERS[name](image_paths, Path(args.output) / name, args.device, DEFAULT_CONF[name])


if __name__ == "__main__":
    main()
