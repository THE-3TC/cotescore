"""Class-level COTe metrics.

This module implements the multi-class extension of the COTe decomposition,
producing K×K interaction matrices and K-vector class-share quantities.

The scalar (class-agnostic) COTe metrics live in ``metrics.py``.

Notation (following the formulation):
  M^S_l          — binary GT mask for class l
  M^S            — binary GT mask across all classes
  M^p,b_k        — binary union mask of all class-k predictions
  M^p_k          — count mask of class-k predictions (pixel = # preds covering it)
  A^P_k          = sum(M^p,b_k) — total pixel area of class-k predictions
  M^O            = [M^p > 1] — pixels covered by two or more predictions of any class
  A^O_k          = sum(M^O & M^S * M^p_k) — class-k prediction area on overlapping GT
  A^S            = sum(M^S)
  i(j)           — SSU that owns prediction j (majority-pixel assignment)
  M^S_{l\\i(j)} — class-l GT mask with owner SSU i(j)'s pixels removed

Interaction matrices (K×K, rows = pred class k, cols = GT class l):

  C[k,l] = sum(M^S_l & M^p,b_k) / sum(M^S & M^p,b_k)
    Coverage confusion matrix. "Of the class-k prediction area on GT, what
    fraction lands on class-l GT?" Diagonal = correct coverage; off-diagonal =
    misclassification. Background is excluded; it stays in coverage_precision.

  O[k,l] = sum_{j in k} sum(M^O & M^S_l & M^p_j) / A^O_k,
  A^O_k  = sum_{j in k} sum(M^O & M^S & M^p_j),  M^O = [M^p > 1]
    Overlap confusion matrix. "Of the overlapping GT area of the class-k
    predictions, what fraction lands on class-l GT?" M^O counts predictions
    of every class, so overlap between classes appears in both rows, but row
    k is weighted only by class k's own predictions. Asymmetric. Row is zero
    when A^O_k=0.

  T[k,l] = sum_{j in k} sum(M^S_{l\\i(j)} & M^p_j) / sum_{j in k} sum(M^S_{\\i(j)} & M^p_j)
    Trespass confusion matrix. "Of the total trespass of the class-k
    predictions, what fraction lands on class-l GT?" The owner SSU is excluded
    in every column. Diagonal = trespass against other SSUs of the same class.

  All three matrices are row-normalised: each non-empty row sums to 1.

Class shares (K-vectors summing to 1):
  C_share[k]  fraction of total coverage attributable to class k
  O_share[k]  fraction of total overlap attributable to class k
  T_share[k]  fraction of total trespass attributable to class k

  Coverage and overlap are attributed pro rata: a GT pixel under several
  predictions is split between classes by the weight M^p_k / M^p, so a pixel
  claimed by two classes is never counted twice.

  C_share[k] = sum(M^S & M^p,b * M^p_k / M^p) / (C · A^S)
  O_share[k] = sum(M^S * (M^p − M^p,b) * M^p_k / M^p) / (O · A^S)
  T_share[k] = sum_{j in k} sum(M^S_{\\i(j)} & M^p_j) / (T · A^S)

  Trespass needs no weighting: the scalar T is itself a sum over
  predictions, each of which has exactly one class.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from cotescore._core import (
    _check_gt_map,
    _ms_mask,
    _owner_ssu_id,
)
from cotescore.types import ClassCOTeCounts, ClassCOTeResult, Label, MaskInstance


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_preds_have_labels(preds: Sequence[MaskInstance]) -> None:
    """Ensure every prediction carries a class label.

    Args:
        preds: Sequence of :class:`~cot_score.types.MaskInstance` predictions.

    Raises:
        ValueError: If any prediction has ``label=None``.
    """
    for i, p in enumerate(preds):
        if p.label is None:
            raise ValueError(
                f"Prediction at index {i} has label=None; all predictions must "
                "carry a class label for class-level metrics."
            )


def _build_class_gt_masks(
    gt_ssu_map: np.ndarray,
    ssu_to_class: Dict[int, Label],
    classes: List[Label],
) -> Dict[Label, np.ndarray]:
    """Return one boolean GT mask per class (pixels belonging to that class's SSUs)."""
    result: Dict[Label, np.ndarray] = {}
    for cls in classes:
        ssu_ids = {sid for sid, c in ssu_to_class.items() if c == cls}
        mask = np.zeros(gt_ssu_map.shape, dtype=bool)
        for sid in ssu_ids:
            mask |= gt_ssu_map == sid
        result[cls] = mask
    return result


def _group_preds_by_class(
    preds: Sequence[MaskInstance],
) -> Dict[Label, List[np.ndarray]]:
    """Group prediction boolean masks by their class label."""
    groups: Dict[Label, List[np.ndarray]] = {}
    for p in preds:
        lbl = p.label  # validated non-None upstream
        m = p.mask.astype(bool, copy=False)
        if m.ndim != 2:
            raise ValueError("Prediction mask must be a 2D array")
        groups.setdefault(lbl, []).append(m)
    return groups


def _class_binary_pred_mask(
    pred_group: List[np.ndarray],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Union binary mask of all predictions in a class group."""
    mp_b = np.zeros(shape, dtype=bool)
    for m in pred_group:
        mp_b |= m
    return mp_b


def _class_count_pred_mask(
    pred_group: List[np.ndarray],
    shape: Tuple[int, int],
) -> np.ndarray:
    """Count mask of all predictions in a class group."""
    mp = np.zeros(shape, dtype=np.int32)
    for m in pred_group:
        mp += m.astype(np.int32, copy=False)
    return mp


def _prorata_share_numers(
    ms: np.ndarray,
    class_mp: Dict[Label, np.ndarray],
    mp_global: np.ndarray,
    classes: List[Label],
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-class coverage and overlap share numerators, attributed pro rata.

    Each GT pixel's coverage (1 if any prediction) and redundancy
    (M^p − M^p,b) is split between classes by the weight M^p_k / M^p — the
    fraction of the predictions on that pixel belonging to class k. The
    weights sum to 1 on every predicted pixel, so the class numerators sum
    exactly to the global coverage and overlap areas, even where predictions
    of different classes overlap.
    """
    ms_f = ms.astype(np.float64)
    mp_f = mp_global.astype(np.float64)
    inv_mp = np.divide(1.0, mp_f, out=np.zeros_like(mp_f), where=mp_f > 0)
    cov_weight = ms_f * inv_mp                                   # M^S ⊙ M^p,b / M^p
    ovl_weight = ms_f * np.maximum(mp_f - 1.0, 0.0) * inv_mp     # M^S ⊙ (M^p − M^p,b) / M^p

    cov = np.zeros(len(classes), dtype=np.float64)
    ovl = np.zeros(len(classes), dtype=np.float64)
    for k, cls in enumerate(classes):
        mp_k = class_mp[cls]
        cov[k] = float(np.sum(cov_weight * mp_k))
        ovl[k] = float(np.sum(ovl_weight * mp_k))
    return cov, ovl


def _row_normalise(numer: np.ndarray) -> np.ndarray:
    """Divide each row by its sum, so every non-empty row sums to 1."""
    totals = numer.sum(axis=1)
    out = np.zeros_like(numer, dtype=np.float64)
    nonzero = totals > 0
    out[nonzero, :] = numer[nonzero, :] / totals[nonzero, None]
    return out


def _micro_prf1(
    tp: np.ndarray, pred_area: np.ndarray, gt_area: np.ndarray
) -> Tuple[float, float, float]:
    """Aggregate per-class TP/predicted-area/GT-area into a single dataset-wide
    precision/recall/F1 (micro-averaged: sum the counts across all classes
    first, then divide once — as opposed to averaging already-normalised
    per-class ratios).
    """
    total_tp = float(np.sum(tp))
    total_pred_area = float(np.sum(pred_area))
    total_gt_area = float(np.sum(gt_area))
    micro_precision = total_tp / total_pred_area if total_pred_area > 0 else 0.0
    micro_recall = total_tp / total_gt_area if total_gt_area > 0 else 0.0
    denom = micro_precision + micro_recall
    micro_f1 = 2 * micro_precision * micro_recall / denom if denom > 0 else 0.0
    return micro_precision, micro_recall, micro_f1


# ---------------------------------------------------------------------------
# Public matrix functions
# ---------------------------------------------------------------------------


def coverage_matrix(
    gt_ssu_map: np.ndarray,
    ssu_to_class: Dict[int, Label],
    preds: Sequence[MaskInstance],
) -> Tuple[np.ndarray, List[Label]]:
    """Compute the K×K coverage interaction matrix.

    C[k, l] = sum(M^S_l & M^p,b_k) / sum(M^S & M^p,b_k)

    "Of the class-k prediction area on GT, what fraction lands on class-l GT?"
    Rows sum to 1. Diagonal entries give correct within-class coverage;
    off-diagonal entries indicate classification errors. Row k is all-zero
    when no class-k prediction touches GT. Prediction area on the background
    is excluded here; it is still in ``coverage_precision``.

    Args:
        gt_ssu_map: 2D integer array of SSU ids (0 = background).
        ssu_to_class: Mapping from SSU id to class label.
        preds: Sequence of MaskInstance, each with a non-None ``label``.

    Returns:
        (matrix, classes) where matrix is K×K float64 and classes is the
        ordered list of class labels defining row/column meaning.
    """
    result = cote_class(gt_ssu_map, ssu_to_class, preds)
    return result.coverage_matrix, result.classes


def overlap_matrix(
    gt_ssu_map: np.ndarray,
    ssu_to_class: Dict[int, Label],
    preds: Sequence[MaskInstance],
) -> Tuple[np.ndarray, List[Label]]:
    """Compute the K×K overlap interaction matrix.

    O[k, l] = sum_{j in k} sum(M^O & M^S_l & M^p_j) / A^O_k
    where A^O_k = sum_{j in k} sum(M^O & M^S & M^p_j) and M^O = [M^p > 1]
    marks pixels covered by two or more predictions of any class.

    "Of the overlapping GT area of the class-k predictions, what fraction
    lands on class-l GT?" Row k depends only on class k's predictions and
    on where overlap occurs, not on how many other-class predictions are
    involved. Rows sum to 1; the matrix is asymmetric. Row k is all-zero
    when no class-k prediction lies on overlapping GT (A^O_k = 0).

    Args:
        gt_ssu_map: 2D integer array of SSU ids (0 = background).
        ssu_to_class: Mapping from SSU id to class label.
        preds: Sequence of MaskInstance, each with a non-None ``label``.

    Returns:
        (matrix, classes) where matrix is K×K float64 and classes is the
        ordered list of class labels defining row/column meaning.
    """
    result = cote_class(gt_ssu_map, ssu_to_class, preds)
    return result.overlap_matrix, result.classes


def trespass_matrix(
    gt_ssu_map: np.ndarray,
    ssu_to_class: Dict[int, Label],
    preds: Sequence[MaskInstance],
) -> Tuple[np.ndarray, List[Label]]:
    """Compute the K×K trespass interaction matrix.

    T[k, l] = sum_{j in k} sum(M^S_{l\\i(j)} & M^p_j) / sum_{j in k} sum(M^S_{\\i(j)} & M^p_j)

    "Of the total trespass of the class-k predictions, what fraction lands on
    class-l GT?" Each prediction's owner SSU i(j) is excluded in every
    column, whatever its class. Rows sum to 1. The diagonal is trespass on
    other SSUs of the same class. Row k is all-zero when no class-k
    prediction trespasses.

    Args:
        gt_ssu_map: 2D integer array of SSU ids (0 = background).
        ssu_to_class: Mapping from SSU id to class label.
        preds: Sequence of MaskInstance, each with a non-None ``label``.

    Returns:
        (matrix, classes) where matrix is K×K float64 and classes is the
        ordered list of class labels defining row/column meaning.
    """
    result = cote_class(gt_ssu_map, ssu_to_class, preds)
    return result.trespass_matrix, result.classes


# ---------------------------------------------------------------------------
# Combined wrapper
# ---------------------------------------------------------------------------


def cote_class(
    gt_ssu_map: np.ndarray,
    ssu_to_class: Dict[int, Label],
    preds: Sequence[MaskInstance],
) -> ClassCOTeResult:
    """Compute the full class-level COTe decomposition for one image.

    Returns a :class:`~cot_score.types.ClassCOTeResult` containing all three
    K×K interaction matrices and all three K-vector share quantities.

    Share vectors each sum to 1.0 (when the corresponding global total is > 0).
    When a global total is zero, the corresponding share vector is all zeros.

    Equivalent to :func:`class_confusion_counts` followed by
    :func:`finalize_class_counts`, with the class list taken from
    ``ssu_to_class``.

    Args:
        gt_ssu_map: 2D integer array of SSU ids (0 = background).
        ssu_to_class: Mapping from SSU id to class label.
        preds: Sequence of MaskInstance, each with a non-None ``label``.

    Returns:
        ClassCOTeResult with all matrices, share vectors, and the ordered
        ``classes`` list.
    """
    classes = sorted(set(ssu_to_class.values()))
    return finalize_class_counts(
        class_confusion_counts(gt_ssu_map, ssu_to_class, preds, classes)
    )


# ---------------------------------------------------------------------------
# Multi-image accumulation
# ---------------------------------------------------------------------------


def class_confusion_counts(
    gt_ssu_map: np.ndarray,
    ssu_to_class: Dict[int, Label],
    preds: Sequence[MaskInstance],
    classes: Sequence[Label],
) -> ClassCOTeCounts:
    """Compute raw, unnormalised per-class pixel counts for one image.

    Unlike :func:`cote_class`, ``classes`` is a required, externally-fixed
    taxonomy (not derived from this image's ``ssu_to_class``), so that counts
    from many images can be summed elementwise with :func:`sum_class_counts`
    before normalising once with :func:`finalize_class_counts`. Averaging
    already-normalised per-image matrices instead would let pages with little
    or no GT for a class distort the average — accumulate raw pixel sums
    across the whole dataset and divide once.

    Predictions whose label is not in ``classes`` are ignored entirely. If
    such predictions should still register as imprecision (e.g. a model's
    "Picture" class with no equivalent in the GT taxonomy), map them onto a
    real bucket in ``classes`` (such as an "other" class) upstream.

    Args:
        gt_ssu_map: 2D integer array of SSU ids (0 = background).
        ssu_to_class: Mapping from SSU id to class label.
        preds: Sequence of MaskInstance, each with a non-None ``label``.
        classes: Fixed, ordered list of class labels defining row/column
            meaning — pass the identical list to every image being
            accumulated together.

    Returns:
        ClassCOTeCounts with raw per-class-pair pixel sums for this image.
    """
    gt_ssu_map = _check_gt_map(gt_ssu_map)
    _validate_preds_have_labels(preds)

    classes = list(classes)
    K = len(classes)
    cls_idx = {c: i for i, c in enumerate(classes)}

    ms = _ms_mask(gt_ssu_map)
    class_gt = _build_class_gt_masks(gt_ssu_map, ssu_to_class, classes)
    pred_groups = _group_preds_by_class(preds)

    class_mp_b: Dict[Label, np.ndarray] = {}
    class_mp: Dict[Label, np.ndarray] = {}
    for cls in classes:
        group = pred_groups.get(cls, [])
        class_mp_b[cls] = _class_binary_pred_mask(group, gt_ssu_map.shape)
        class_mp[cls] = _class_count_pred_mask(group, gt_ssu_map.shape)

    mp_b_global = np.zeros(gt_ssu_map.shape, dtype=bool)
    mp_global = np.zeros(gt_ssu_map.shape, dtype=np.int32)
    for cls in classes:
        mp_b_global |= class_mp_b[cls]
        mp_global += class_mp[cls]

    coverage_numer = np.zeros((K, K), dtype=np.float64)
    pred_area = np.zeros(K, dtype=np.float64)
    gt_area = np.zeros(K, dtype=np.float64)
    overlap_numer = np.zeros((K, K), dtype=np.float64)
    overlap_area = np.zeros(K, dtype=np.float64)
    trespass_numer = np.zeros((K, K), dtype=np.float64)
    trespass_share_numer = np.zeros(K, dtype=np.float64)

    m_o = mp_global > 1  # pixels covered by 2+ preds of any class

    for cls, k in cls_idx.items():
        mp_k_b = class_mp_b[cls]
        mp_k_o = class_mp[cls] * m_o
        pred_area[k] = float(np.sum(mp_k_b))
        gt_area[k] = float(np.sum(class_gt[cls]))
        overlap_area[k] = float(np.sum(mp_k_o * ms))

        for l_cls, l in cls_idx.items():
            ms_l = class_gt[l_cls]
            coverage_numer[k, l] = float(np.sum(ms_l & mp_k_b))
            overlap_numer[k, l] = float(np.sum(mp_k_o * ms_l))

        # Each prediction's owner SSU is excluded in every column, whatever
        # its class. A prediction with no owner has no GT pixels, so no trespass.
        for pm in pred_groups.get(cls, []):
            owner = _owner_ssu_id(gt_ssu_map, pm)
            if owner is None:
                continue
            trespassed = pm & (gt_ssu_map != owner)
            for l_cls, l in cls_idx.items():
                trespass_numer[k, l] += float(np.sum(trespassed & class_gt[l_cls]))
            trespass_share_numer[k] += float(np.sum(trespassed & ms))

    coverage_share_numer, overlap_share_numer = _prorata_share_numers(
        ms, class_mp, mp_global, classes
    )
    global_coverage_area = int(np.sum(ms & mp_b_global))
    mp_b_global_int = mp_b_global.astype(np.int32)
    global_overlap_area = int(np.sum(ms.astype(np.int32) * (mp_global - mp_b_global_int)))
    global_trespass_pixels = int(np.sum(trespass_share_numer))

    return ClassCOTeCounts(
        classes=classes,
        coverage_numer=coverage_numer,
        pred_area=pred_area,
        gt_area=gt_area,
        overlap_numer=overlap_numer,
        overlap_area=overlap_area,
        trespass_numer=trespass_numer,
        coverage_share_numer=coverage_share_numer,
        overlap_share_numer=overlap_share_numer,
        trespass_share_numer=trespass_share_numer,
        global_coverage_area=global_coverage_area,
        global_overlap_area=global_overlap_area,
        global_trespass_pixels=global_trespass_pixels,
    )


def sum_class_counts(a: ClassCOTeCounts, b: ClassCOTeCounts) -> ClassCOTeCounts:
    """Accumulate two :class:`~cotescore.types.ClassCOTeCounts` elementwise.

    Args:
        a: Running total (or the first image's counts).
        b: The next image's counts. Must share the same ``classes`` list, in
            the same order, as ``a``.

    Returns:
        A new ClassCOTeCounts with every field summed.

    Raises:
        ValueError: If ``a.classes != b.classes``.
    """
    if a.classes != b.classes:
        raise ValueError(
            "Cannot sum ClassCOTeCounts with different class lists; pass the "
            "same fixed `classes` to every class_confusion_counts call."
        )
    return ClassCOTeCounts(
        classes=a.classes,
        coverage_numer=a.coverage_numer + b.coverage_numer,
        pred_area=a.pred_area + b.pred_area,
        gt_area=a.gt_area + b.gt_area,
        overlap_numer=a.overlap_numer + b.overlap_numer,
        overlap_area=a.overlap_area + b.overlap_area,
        trespass_numer=a.trespass_numer + b.trespass_numer,
        coverage_share_numer=a.coverage_share_numer + b.coverage_share_numer,
        overlap_share_numer=a.overlap_share_numer + b.overlap_share_numer,
        trespass_share_numer=a.trespass_share_numer + b.trespass_share_numer,
        global_coverage_area=a.global_coverage_area + b.global_coverage_area,
        global_overlap_area=a.global_overlap_area + b.global_overlap_area,
        global_trespass_pixels=a.global_trespass_pixels + b.global_trespass_pixels,
    )


def finalize_class_counts(counts: ClassCOTeCounts) -> ClassCOTeResult:
    """Normalise accumulated :class:`~cotescore.types.ClassCOTeCounts` into
    the final matrices, shares, and precision/recall/F1.

    This is the division step shared by :func:`cote_class` (single image) and
    the multi-image accumulation path (:func:`class_confusion_counts` +
    :func:`sum_class_counts`): call it once, after summing every image's
    counts, not per image.

    Args:
        counts: Raw pixel sums, typically the result of summing
            :func:`class_confusion_counts` across every image in a dataset.

    Returns:
        ClassCOTeResult with all matrices, share vectors, and
        precision/recall/F1.
    """
    classes = list(counts.classes)
    K = len(classes)
    pred_area = counts.pred_area
    gt_area = counts.gt_area

    C = _row_normalise(counts.coverage_numer)

    O = np.zeros((K, K), dtype=np.float64)
    nonzero_o = counts.overlap_area > 0
    O[nonzero_o, :] = counts.overlap_numer[nonzero_o, :] / counts.overlap_area[nonzero_o, None]

    T = _row_normalise(counts.trespass_numer)

    C_share = np.zeros(K, dtype=np.float64)
    if counts.global_coverage_area > 0:
        C_share = counts.coverage_share_numer / counts.global_coverage_area

    O_share = np.zeros(K, dtype=np.float64)
    if counts.global_overlap_area > 0:
        O_share = counts.overlap_share_numer / counts.global_overlap_area

    T_share = np.zeros(K, dtype=np.float64)
    if counts.global_trespass_pixels > 0:
        T_share = counts.trespass_share_numer / counts.global_trespass_pixels

    tp = np.diag(counts.coverage_numer)
    coverage_precision = np.zeros(K, dtype=np.float64)
    nonzero_p = pred_area > 0
    coverage_precision[nonzero_p] = tp[nonzero_p] / pred_area[nonzero_p]
    coverage_recall = np.zeros(K, dtype=np.float64)
    nonzero_s = gt_area > 0
    coverage_recall[nonzero_s] = tp[nonzero_s] / gt_area[nonzero_s]

    coverage_f1 = np.zeros(K, dtype=np.float64)
    denom = coverage_precision + coverage_recall
    nonzero_f1 = denom > 0
    coverage_f1[nonzero_f1] = (
        2 * coverage_precision[nonzero_f1] * coverage_recall[nonzero_f1] / denom[nonzero_f1]
    )
    micro_precision, micro_recall, micro_f1 = _micro_prf1(tp, pred_area, gt_area)

    return ClassCOTeResult(
        classes=classes,
        coverage_matrix=C,
        overlap_matrix=O,
        trespass_matrix=T,
        coverage_share=C_share,
        overlap_share=O_share,
        trespass_share=T_share,
        coverage_precision=coverage_precision,
        coverage_recall=coverage_recall,
        coverage_f1=coverage_f1,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
    )


__all__ = [
    "coverage_matrix",
    "overlap_matrix",
    "trespass_matrix",
    "cote_class",
    "class_confusion_counts",
    "sum_class_counts",
    "finalize_class_counts",
]
