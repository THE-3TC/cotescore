"""
Document layout analysis metrics

This module provides metrics for evaluating document layout analysis predictions:
- coverage: Measures how well predictions cover ground truth regions
- overlap: Measures duplication/repetition in predictions
- iou: Intersection over Union for two boxes
- mean_iou: Average IoU across all ground truth boxes
- trespass: Measures false positive coverage on wrong ground truth regions
- excess: Measures coverage of background (non-ground truth) area
"""

from typing import List, Dict, Any, Union, Tuple, Optional, Iterable, Sequence, overload
import numpy as np
import collections

from cotescore.types import GTBoxes, MaskInstance
from cotescore.adapters import calculate_intersection_area as _calculate_intersection_area
from cotescore._core import (
    _as_pred_masks,
    _check_gt_map,
    _ms_mask,
    _area_s,
    _compose_pred_count,
    _owner_ssu_id,
)
from cotescore._core_bbox import bbox_cote_components


BBox = Dict[str, Any]
InputBox = Union[BBox, List[float], Tuple[float, ...]]


def coverage(
    gt_ssu_map: np.ndarray,
    preds: Sequence[Union[np.ndarray, MaskInstance]],
) -> float:
    """
    Calculate coverage metric between predicted and ground truth regions.

    Coverage measures how well predicted regions cover the ground truth regions.
    It is the ratio of ground truth area covered by predictions to total ground
    truth area.

    Args:
        gt_ssu_map: 2D integer array of SSU ids. Background must be 0.
        preds: Sequence of prediction masks (2D numpy arrays) or MaskInstance.
            Each mask must have the same shape as gt_ssu_map.

    Returns:
        Coverage score [0.0, 1.0].
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    pred_masks = _as_pred_masks(preds)
    ms = _ms_mask(gt_ssu_map)
    a_s = _area_s(ms)
    if a_s == 0:
        return 1.0 if not pred_masks else 0.0
    if not pred_masks:
        return 0.0
    mp = _compose_pred_count(pred_masks, gt_ssu_map.shape)
    mpb = mp > 0
    covered_area = int(np.sum(ms & mpb))
    return float(covered_area / a_s)


def overlap(
    gt_ssu_map: np.ndarray,
    preds: Sequence[Union[np.ndarray, MaskInstance]],
) -> float:
    """
    Calculate overlap metric between predicted regions.

    Overlap is the ratio of redundant prediction area (pixels covered by
    more than one prediction) within the ground truth to total ground truth area.

    Args:
        gt_ssu_map: 2D integer array of SSU ids. Background must be 0.
        preds: Sequence of prediction masks (2D numpy arrays) or MaskInstance.
            Each mask must have the same shape as gt_ssu_map.

    Returns:
        Overlap score (O_raw). 0 means no redundancy.
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    pred_masks = _as_pred_masks(preds)
    ms = _ms_mask(gt_ssu_map)
    a_s = _area_s(ms)
    if a_s == 0:
        return 0.0
    if not pred_masks:
        return 0.0
    mp = _compose_pred_count(pred_masks, gt_ssu_map.shape)
    mpb = (mp > 0).astype(np.int32)
    redundancy = mp - mpb
    overlap_area = int(np.sum(ms.astype(np.int32) * redundancy))
    return float(overlap_area / a_s)


def iou(box1: BBox, box2: BBox) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box dict with keys 'x', 'y', 'width', 'height'.
        box2: Second bounding box dict with keys 'x', 'y', 'width', 'height'.

    Returns:
        IoU score in range [0.0, 1.0].
    """
    intersection = _calculate_intersection_area(box1, box2)

    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def f1(
    predicted_regions: List[BBox], ground_truth_regions: List[BBox], threshold: float = 0.5
) -> float:
    """
    Calculate F1 score at a given IoU threshold (default 0.50) for one image.

    Matching follows COCO (``pycocotools.cocoeval``): predictions are taken in
    descending ``confidence`` (input order when absent or tied), and each claims
    the still-unmatched ground truth box with the highest IoU, provided that IoU
    is at least ``threshold``. Matched predictions are true positives (TP),
    unmatched predictions false positives (FP), and unmatched GT boxes false
    negatives (FN). Classes are ignored; for class-aware or dataset-level F1
    use :class:`cotescore.map_metric.MAPMetric`, which applies the same rule.

    Args:
        predicted_regions: List of predicted bounding boxes, optionally with a
            ``confidence`` key.
        ground_truth_regions: List of ground truth bounding boxes.
        threshold: IoU threshold for a match to count as a true positive (default 0.5).

    Returns:
        F1 score in range [0.0, 1.0]; 1.0 when both lists are empty.
    """
    if not ground_truth_regions and not predicted_regions:
        return 1.0
    if not ground_truth_regions or not predicted_regions:
        return 0.0

    order = sorted(
        range(len(predicted_regions)),
        key=lambda j: -predicted_regions[j].get("confidence", 0.0),
    )
    matched_gt = set()
    for j in order:
        best_iou, best_idx = min(threshold, 1 - 1e-10), -1
        for g, gt_box in enumerate(ground_truth_regions):
            if g in matched_gt:
                continue
            score = iou(predicted_regions[j], gt_box)
            if score >= best_iou:
                best_iou, best_idx = score, g
        if best_idx >= 0:
            matched_gt.add(best_idx)

    tp = len(matched_gt)
    fp = len(predicted_regions) - tp
    fn = len(ground_truth_regions) - tp
    return 2 * tp / (2 * tp + fp + fn)


def mean_iou(predicted_regions: List[BBox], ground_truth_regions: List[BBox]) -> float:
    """
    Calculate mean Intersection over Union (IoU) across ground truth boxes.

    For each ground truth box, finds the best matching predicted box and
    computes their IoU. Returns the average IoU across all ground truth boxes.

    Args:
        predicted_regions: List of predicted bounding boxes.
        ground_truth_regions: List of ground truth bounding boxes.

    Returns:
        Mean IoU score in range [0.0, 1.0].
    """
    if not ground_truth_regions:
        return 1.0 if not predicted_regions else 0.0

    if not predicted_regions:
        return 0.0

    total_iou = sum(
        max((iou(pred_box, gt_box) for pred_box in predicted_regions), default=0.0)
        for gt_box in ground_truth_regions
    )

    return total_iou / len(ground_truth_regions)


def trespass(
    gt_ssu_map: np.ndarray,
    preds: Sequence[Union[np.ndarray, MaskInstance]],
) -> float:
    """
    Trespass measures how much of the ground truth is covered by a prediction
    that belongs to a different SSU (Ground Truth Region).

    Args:
        gt_ssu_map: 2D integer array of SSU ids. Background must be 0.
        preds: Sequence of prediction masks (2D numpy arrays) or MaskInstance.
            Each mask must have the same shape as gt_ssu_map.

    Returns:
        Trespass score (T_raw). Fraction of GT area invaded by non-owner predictions.
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    pred_masks = _as_pred_masks(preds)
    ms = _ms_mask(gt_ssu_map)
    a_s = _area_s(ms)
    if a_s == 0:
        return 0.0
    if not pred_masks:
        return 0.0
    total_trespass_pixels = 0
    for pm in pred_masks:
        owner = _owner_ssu_id(gt_ssu_map, pm)
        if owner is None:
            continue
        trespass_pixels = int(np.sum(pm & ms & (gt_ssu_map != owner)))
        total_trespass_pixels += trespass_pixels
    return float(total_trespass_pixels / a_s)


def excess(
    gt_ssu_map: np.ndarray,
    preds: Sequence[Union[np.ndarray, MaskInstance]],
) -> float:
    """
    Excess measures the amount of area covered by predictions that is not part
    of any ground truth region (SSU). It is the ratio of predicted white-space
    area to total white-space area.

    Args:
        gt_ssu_map: 2D integer array of SSU ids. Background must be 0.
        preds: Sequence of prediction masks (2D numpy arrays) or MaskInstance.
            Each mask must have the same shape as gt_ssu_map.

    Returns:
        Excess score [0.0, 1.0].
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    pred_masks = _as_pred_masks(preds)
    if not pred_masks:
        return 0.0
    ms = _ms_mask(gt_ssu_map)
    mp = _compose_pred_count(pred_masks, gt_ssu_map.shape)
    mpb = mp > 0
    n_mask = ~ms
    white_space_area = int(np.sum(n_mask))
    if white_space_area <= 0:
        return 0.0
    excess_area = int(np.sum(n_mask & mpb))
    e_val = excess_area / white_space_area
    return min(1.0, max(0.0, float(e_val)))


def cote_score(
    gt_ssu_map: Union[np.ndarray, GTBoxes],
    preds: Union[Sequence[Union[np.ndarray, MaskInstance]], np.ndarray],
    weight_coverage: float = 1.0,
    weight_overlap: float = 1.0,
    weight_trespass: float = 1.0,
) -> Tuple[float, float, float, float, float]:
    """Compute the full COTe score decomposition for a set of predictions.

    Returns the weighted composite COTe score together with the four
    individual component values: Coverage, Overlap, Trespass, and Excess.

    Supports two mutually exclusive representations, dispatched by the type
    of `gt_ssu_map`:

    - Mask mode (default): `gt_ssu_map` is a 2D integer pixel array and
      `preds` is a sequence of prediction masks. This rasterizes-then-scores,
      exactly as before.
    - Bbox mode: `gt_ssu_map` is a :class:`~cotescore.types.GTBoxes` instance
      and `preds` is an ``(M, 4)`` array of predicted boxes in
      ``[x, y, width, height]`` format. This computes all four components
      analytically via coordinate compression, without ever allocating a
      pixel raster -- much faster when predictions are boxes rather than
      masks (the common case for doc-layout models).

    Mixing representations (e.g. `GTBoxes` with a mask sequence, or a pixel
    array with a box array) is not supported and raises `TypeError` --
    rasterize the box side first via :func:`~cotescore.adapters.boxes_to_gt_ssu_map`/
    :func:`~cotescore.adapters.boxes_to_pred_masks` instead of mixing modes.
    Note: passing a bare ``(K, 4)`` integer array as `gt_ssu_map` (forgetting
    to wrap it in `GTBoxes`) is not detected and will be silently
    misinterpreted as a tiny pixel mask, since both are valid 2D integer
    arrays.

    Args:
        gt_ssu_map: Either a 2D integer array where each pixel holds the SSU
            id of its ground-truth region (0 = background), or a
            :class:`~cotescore.types.GTBoxes` instance for the bbox fast path.
        preds: For mask mode: sequence of predictions, each either a 2D
            boolean numpy array or a :class:`~cotescore.types.MaskInstance`.
            For bbox mode: an ``(M, 4)`` numpy array of predicted boxes in
            ``[x, y, width, height]`` format, same coordinate frame as
            `gt_ssu_map.boxes`.
        weight_coverage: Weight applied to the Coverage component in the
            composite score (default 1.0).
        weight_overlap: Weight applied to the Overlap component in the
            composite score (default 1.0).
        weight_trespass: Weight applied to the Trespass component in the
            composite score (default 1.0).

    Returns:
        A 5-tuple ``(cote, coverage, overlap, trespass, excess)`` where
        ``cote = weight_coverage * C - weight_overlap * O - weight_trespass * T``
        and each component is in the range ``[0.0, 1.0]``.

    Raises:
        TypeError: If `gt_ssu_map` and `preds` use mismatched representations.
    """
    if isinstance(gt_ssu_map, GTBoxes):
        if not isinstance(preds, np.ndarray):
            raise TypeError(
                "gt_ssu_map is a GTBoxes instance, so preds must be an (M, 4) "
                "numpy array of [x, y, width, height] boxes, not a mask "
                "sequence. Rasterize preds first via boxes_to_pred_masks(...) "
                "if it is actually masks."
            )
        C, O, T, E = bbox_cote_components(gt_ssu_map, preds)
    else:
        if isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] == 4:
            raise TypeError(
                "gt_ssu_map is a pixel SSU map but preds looks like an (M, 4) "
                "box array, not a mask sequence. Rasterize preds first via "
                "boxes_to_pred_masks(...), or pass a GTBoxes gt_ssu_map instead."
            )
        gt_ssu_map = _check_gt_map(gt_ssu_map)
        pred_masks = _as_pred_masks(preds)
        n = len(pred_masks)
        C = coverage(gt_ssu_map, pred_masks)
        T = trespass(gt_ssu_map, pred_masks)
        if n <= 1:
            O = 0.0
        else:
            O = overlap(gt_ssu_map, pred_masks)
        E = excess(gt_ssu_map, pred_masks)
    cot = (weight_coverage * C) - (weight_overlap * O) - (weight_trespass * T)
    return (float(cot), float(C), float(O), float(T), float(E))




__all__ = [
    "coverage",
    "overlap",
    "iou",
    "mean_iou",
    "f1",
    "trespass",
    "excess",
    "cote_score",
]


def _standardize_box_format(box: InputBox, format_str: Optional[str] = None) -> BBox:
    """
    Standardize an input box to the standard {'x', 'y', 'width', 'height'} dictionary format.

    Args:
        box: Input bounding box. Can be:
             - Dict with keys 'x', 'y', 'width', 'height' (canonical)
             - Dict with keys 'xmin', 'ymin', 'xmax', 'ymax' (auto-detected as xyxy)
             - List/Tuple of 4 numbers: [a, b, c, d]
        format_str: Format string describing the input box structure. Required for List/Tuple inputs.
                    Supported formats:
                    - 'xywh': [x, y, width, height]
                    - 'xyxy': [xmin, ymin, xmax, ymax]
                    - 'cxcywh': [center_x, center_y, width, height]
                    If None, assumes input is already a canonical dictionary or attempts to infer from dict keys.

    Returns:
        Dict with 'x', 'y', 'width', 'height'.
    """

    if isinstance(box, dict):
        # Canonical format
        if "x" in box and "y" in box and "width" in box and "height" in box:
            return box
        # Explicit XYXY dict keys
        if "xmin" in box and "ymin" in box and "xmax" in box and "ymax" in box:
            return {
                "x": box["xmin"],
                "y": box["ymin"],
                "width": box["xmax"] - box["xmin"],
                "height": box["ymax"] - box["ymin"],
            }
        return box

    if format_str is None:
        raise ValueError("format_str must be provided for list/tuple inputs (e.g. 'xywh', 'xyxy')")

    if len(box) < 4:
        raise ValueError(f"Input box must have at least 4 elements, got {len(box)}")

    a, b, c, d = float(box[0]), float(box[1]), float(box[2]), float(box[3])

    if format_str.lower() == "xywh":
        return {"x": a, "y": b, "width": c, "height": d}

    elif format_str.lower() == "xyxy":
        # a=xmin, b=ymin, c=xmax, d=ymax
        return {"x": a, "y": b, "width": c - a, "height": d - b}

    elif format_str.lower() == "cxcywh":
        # a=cx, b=cy, c=w, d=h
        # x = cx - w/2
        # y = cy - h/2
        return {"x": a - c / 2, "y": b - d / 2, "width": c, "height": d}

    else:
        raise ValueError(f"Unknown box format: {format_str}")


def _standardize_input_format(
    regions: Iterable[InputBox], format_str: Optional[str] = None
) -> List[BBox]:
    """Standardize a sequence of regions to the internal bounding box dict format.

    Delegates to :func:`_standardize_box_format` for each element.

    Args:
        regions: Iterable of input boxes, each either a dict with ``"x"``,
            ``"y"``, ``"width"``, ``"height"`` keys, or a 4-element sequence
            interpreted according to ``format_str``.
        format_str: Box coordinate format for sequence inputs — one of
            ``"xywh"``, ``"xyxy"``, or ``"cxcywh"``. If ``None``, dict
            inputs are passed through unchanged.

    Returns:
        List of bounding box dicts with keys ``"x"``, ``"y"``, ``"width"``,
        and ``"height"``.
    """
    if not regions:
        return []

    return [_standardize_box_format(box, format_str) for box in regions]


def _create_mask(
    boxes: List[BBox], image_shape: Tuple[int, int], binary: bool = False
) -> np.ndarray:
    """
    Create a 2D mask from a list of bounding boxes.

    Args:
        boxes: List of standardized bounding boxes (dictionaries with 'x', 'y', 'width', 'height').
        image_shape: Tuple of (height, width).
        binary: If True, returns a binary mask (1 if covered, 0 otherwise).
                If False, returns a count mask (number of boxes covering each pixel).

    Returns:
        np.ndarray: 2D array of shape (height, width) with int32 dtype.
    """
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.int32)

    for box in boxes:
        # Convert to integer coordinates with clamping
        x1 = max(0, int(round(box["x"])))
        y1 = max(0, int(round(box["y"])))
        x2 = min(width, int(round(box["x"] + box["width"])))
        y2 = min(height, int(round(box["y"] + box["height"])))

        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] += 1

    if binary:
        return (mask > 0).astype(np.int32)

    return mask
