"""
COCO-style mean Average Precision (mAP) on top of pycocotools.

:class:`MAPMetric` accumulates per-image predictions and ground truth, then
runs ``pycocotools.cocoeval.COCOeval`` over the whole set. It reports the
standard COCO numbers (AP@[.50:.05:.95], AP@.50, AP@.75) plus per-class AP.

Both box (``iou_type="bbox"``) and pixel (``iou_type="segm"``) matching are
supported. In ``segm`` mode each annotation carries a boolean ``mask`` array
(or a pre-encoded COCO ``segmentation``) instead of ``x/y/width/height``.
"""

from __future__ import annotations

import contextlib
import io
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# COCO convention: an AP that could not be computed (e.g. no GT for a class).
_UNDEFINED = -1.0


class MAPMetric:
    """
    Accumulate detections image by image and compute COCO mAP.

    Args:
        iou_type: ``"bbox"`` (default) matches on axis-aligned boxes;
            ``"segm"`` matches on binary masks.
        max_dets: COCO ``maxDets`` thresholds. The last value caps the
            number of predictions scored per image; the default raises the
            COCO cap of 100 so dense pages are not silently truncated.

    Each ``update`` call is one image. Prediction dicts must contain
    ``class`` and ``confidence``; ground-truth dicts must contain ``class``.
    Geometry is ``x``, ``y``, ``width``, ``height`` (pixels, top-left origin)
    for ``bbox``, or ``mask`` (``(H, W)`` bool array) / ``segmentation``
    (COCO RLE or polygon) for ``segm``.
    """

    def __init__(self, iou_type: str = "bbox", max_dets: Sequence[int] = (1, 10, 500)):
        if iou_type not in ("bbox", "segm"):
            raise ValueError(f"iou_type must be 'bbox' or 'segm', got {iou_type!r}")
        try:
            import pycocotools  # noqa: F401
        except ImportError as e:
            raise ImportError("MAPMetric requires 'pycocotools'. Install with pip install pycocotools") from e

        self.iou_type = iou_type
        self.max_dets = list(max_dets)
        self._label_map: Dict[str, int] = {}
        self._next_id = 0
        self.reset()

    def reset(self) -> None:
        """Discard all accumulated images and annotations (labels are kept)."""
        self._images: List[Dict[str, Any]] = []
        self._gt_anns: List[Dict[str, Any]] = []
        self._pred_anns: List[Dict[str, Any]] = []
        self._next_image_id = 1
        self._next_gt_id = 1

    # ------------------------------------------------------------------ input
    def update(
        self,
        predictions: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]],
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Add one image's predictions and ground truth.

        Args:
            predictions: Dicts with geometry, ``class`` and ``confidence``.
            ground_truth: Dicts with geometry and ``class``.
            image_size: ``(height, width)``. Required for ``segm`` when
                annotations use polygon ``segmentation``; inferred from
                ``mask`` arrays otherwise.
        """
        image_id = self._next_image_id
        self._next_image_id += 1

        h, w = self._resolve_image_size(predictions, ground_truth, image_size)
        self._images.append({"id": image_id, "height": h, "width": w})

        for g in ground_truth:
            ann = self._to_coco_ann(g, image_id, h, w)
            ann["id"] = self._next_gt_id
            self._next_gt_id += 1
            self._gt_anns.append(ann)

        for p in predictions:
            ann = self._to_coco_ann(p, image_id, h, w)
            ann["score"] = float(p.get("confidence", 0.0))
            self._pred_anns.append(ann)

    def _resolve_image_size(self, predictions, ground_truth, image_size) -> Tuple[int, int]:
        if image_size is not None:
            return int(image_size[0]), int(image_size[1])
        if self.iou_type == "segm":
            for a in list(ground_truth) + list(predictions):
                if "mask" in a:
                    return tuple(int(v) for v in np.asarray(a["mask"]).shape[:2])
        # bbox mode never reads the image extent; segm with pre-encoded RLE
        # carries its own size, so a placeholder is sufficient.
        return 0, 0

    def _to_coco_ann(self, a: Dict[str, Any], image_id: int, h: int, w: int) -> Dict[str, Any]:
        ann: Dict[str, Any] = {
            "image_id": image_id,
            "category_id": self._get_label_id(a["class"]),
            "iscrowd": 0,
        }
        if self.iou_type == "bbox":
            bw, bh = float(a["width"]), float(a["height"])
            ann["bbox"] = [float(a["x"]), float(a["y"]), bw, bh]
            ann["area"] = bw * bh
        else:
            from pycocotools import mask as mask_utils

            if "mask" in a:
                m = np.asfortranarray(np.asarray(a["mask"], dtype=np.uint8))
                rle = mask_utils.encode(m)
            elif "segmentation" in a:
                seg = a["segmentation"]
                if isinstance(seg, dict) and isinstance(seg.get("counts"), bytes):
                    rle = seg
                else:  # polygon list or uncompressed RLE
                    if not (h and w):
                        raise ValueError("image_size is required for polygon / uncompressed RLE segmentation")
                    rles = mask_utils.frPyObjects(seg, h, w)
                    rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
            else:
                raise KeyError("segm annotations need a 'mask' or 'segmentation' key")
            ann["segmentation"] = rle
            ann["area"] = float(mask_utils.area(rle))
            ann["bbox"] = [float(v) for v in mask_utils.toBbox(rle)]
        return ann

    # ---------------------------------------------------------------- output
    def compute(self) -> Dict[str, Any]:
        """
        Compute COCO mAP over everything accumulated so far.

        Returns:
            ``{"map", "map_50", "map_75", "classes": {name: AP}}``.
            ``map*`` are AP@[.50:.05:.95], AP@.50 and AP@.75 over all classes.
            A value of ``-1.0`` follows the COCO convention for "undefined"
            (no ground truth to score against). With no predictions every
            AP is ``0.0``.
        """
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        classes = sorted(self._label_map.items(), key=lambda kv: kv[1])
        if not self._gt_anns:
            return {"map": _UNDEFINED, "map_50": _UNDEFINED, "map_75": _UNDEFINED,
                    "classes": {name: _UNDEFINED for name, _ in classes}}
        if not self._pred_anns:
            gt_cats = {a["category_id"] for a in self._gt_anns}
            return {"map": 0.0, "map_50": 0.0, "map_75": 0.0,
                    "classes": {name: (0.0 if cid in gt_cats else _UNDEFINED) for name, cid in classes}}

        coco_gt = COCO()
        coco_gt.dataset = {
            "images": self._images,
            "annotations": self._gt_anns,
            "categories": [{"id": cid, "name": name} for name, cid in classes],
        }
        with contextlib.redirect_stdout(io.StringIO()):
            coco_gt.createIndex()
            coco_dt = coco_gt.loadRes(self._pred_anns)
            ev = COCOeval(coco_gt, coco_dt, iouType=self.iou_type)
            ev.params.maxDets = self.max_dets
            ev.evaluate()
            ev.accumulate()

        # precision: [T iou thresholds, R recall points, K categories, A area ranges, M maxDets]
        precision = ev.eval["precision"][:, :, :, 0, -1]  # area='all', maxDets=max
        iou_thrs = ev.params.iouThrs

        def _ap(p: np.ndarray) -> float:
            valid = p[p > -1]
            return float(valid.mean()) if valid.size else _UNDEFINED

        t50 = int(np.argmin(np.abs(iou_thrs - 0.50)))
        t75 = int(np.argmin(np.abs(iou_thrs - 0.75)))
        cat_index = {cid: k for k, cid in enumerate(ev.params.catIds)}

        return {
            "map": _ap(precision),
            "map_50": _ap(precision[t50]),
            "map_75": _ap(precision[t75]),
            "classes": {
                name: (_ap(precision[:, :, cat_index[cid]]) if cid in cat_index else _UNDEFINED)
                for name, cid in classes
            },
        }

    # ---------------------------------------------------------------- labels
    def _get_label_id(self, label_str: Any) -> int:
        """Map a class label to a stable integer category id (packed from 0)."""
        label_str = str(label_str)
        if label_str not in self._label_map:
            self._label_map[label_str] = self._next_id
            self._next_id += 1
        return self._label_map[label_str]

    def get_class_name(self, label_id: int) -> str:
        """Reverse map category id to class name."""
        for name, lid in self._label_map.items():
            if lid == label_id:
                return name
        return f"Unknown_{label_id}"
