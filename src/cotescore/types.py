"""Shared type definitions used across the cot_score package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np


Label = Union[str, int]


@dataclass(frozen=True)
class MaskInstance:
    """A single prediction represented as a binary mask with optional metadata.

    Attributes:
        mask: A 2D boolean numpy array where ``True`` pixels indicate the
            predicted region.
        label: Optional class label for the prediction (string or integer).
        score: Optional confidence score for the prediction.
        pred_id: Optional integer identifier for the prediction.
    """

    mask: np.ndarray
    label: Optional[Label] = None
    score: Optional[float] = None
    pred_id: Optional[int] = None


@dataclass(frozen=True)
class TokenPositions:
    """GT token positions (characters or words) with pixel midpoints.

    tokens, xs, and ys are parallel arrays of length N. Tokens can be single
    characters (for CDD/SpACER) or word strings (for SpAWER) — the spatial
    logic is identical in both cases.

    Attributes:
        tokens: 1D array of token strings (characters or words).
        xs:     1D integer array of x pixel coordinates (column index).
        ys:     1D integer array of y pixel coordinates (row index).
    """

    tokens: np.ndarray  # shape (N,) dtype object
    xs: np.ndarray      # shape (N,) dtype int
    ys: np.ndarray      # shape (N,) dtype int

    def __post_init__(self) -> None:
        if not (self.tokens.ndim == self.xs.ndim == self.ys.ndim == 1):
            raise ValueError("tokens, xs, ys must be 1D arrays")
        if not (len(self.tokens) == len(self.xs) == len(self.ys)):
            raise ValueError("tokens, xs, ys must have the same length")


@dataclass(frozen=True)
class RegionChars:
    """GT characters with pixel midpoints and GT region membership.

    All arrays are parallel, length N (one entry per character).

    Attributes:
        tokens:     1D array of token strings (characters or words).
        xs:         1D integer array of x pixel coordinates.
        ys:         1D integer array of y pixel coordinates.
        region_ids: 1D integer array of GT region ids.
    """

    tokens:     np.ndarray  # shape (N,) dtype object
    xs:         np.ndarray  # shape (N,) dtype int
    ys:         np.ndarray  # shape (N,) dtype int
    region_ids: np.ndarray  # shape (N,) dtype int

    def __post_init__(self) -> None:
        if not (self.tokens.ndim == self.xs.ndim == self.ys.ndim == self.region_ids.ndim == 1):
            raise ValueError("tokens, xs, ys, region_ids must be 1D arrays")
        if not (len(self.tokens) == len(self.xs) == len(self.ys) == len(self.region_ids)):
            raise ValueError("tokens, xs, ys, region_ids must have the same length")


@dataclass(frozen=True)
class RegionPixels:
    """Pixel membership for predicted regions.

    Encodes which pixels belong to each predicted region. One row per pixel
    in each predicted region; overlapping regions produce duplicate (x, y)
    pairs with different region_ids, encoding the overlap naturally.

    Attributes:
        region_ids: 1D integer array of predicted region ids.
        xs:         1D integer array of x pixel coordinates.
        ys:         1D integer array of y pixel coordinates.
    """

    region_ids: np.ndarray  # shape (M,) dtype int
    xs:         np.ndarray  # shape (M,) dtype int
    ys:         np.ndarray  # shape (M,) dtype int

    def __post_init__(self) -> None:
        if not (self.region_ids.ndim == self.xs.ndim == self.ys.ndim == 1):
            raise ValueError("region_ids, xs, ys must be 1D arrays")
        if not (len(self.region_ids) == len(self.xs) == len(self.ys)):
            raise ValueError("region_ids, xs, ys must have the same length")


@dataclass(frozen=True)
class GTBoxes:
    """Ground-truth SSU boxes for the analytic bounding-box COTe fast path.

    Used as the ``gt`` argument to :func:`~cotescore.layout.cote_score` when
    ground truth is available as boxes rather than a rasterized pixel map.
    Bundles image extent alongside the boxes because ``excess``'s
    background-area denominator needs it, and pure box coordinates carry no
    information about the true canvas size (mask mode gets this for free
    from ``gt_ssu_map.shape``).

    Attributes:
        boxes: (K, 4) float array of GT boxes in [x, y, width, height]
            format, in the same coordinate frame as `image_width`/`image_height`.
        ssu_ids: (K,) int array of GT SSU ids, parallel to `boxes`. 0 is
            reserved for background and must not appear here. Ties for
            overlapping GT boxes are broken by array order (index 0 wins),
            mirroring boxes_to_gt_ssu_map's first-write-wins semantics for a
            box list processed in the same order.
        image_width: Width of the full image/canvas the boxes are defined in.
        image_height: Height of the full image/canvas the boxes are defined in.
    """

    boxes: np.ndarray       # shape (K, 4) dtype float
    ssu_ids: np.ndarray     # shape (K,) dtype int
    image_width: float
    image_height: float

    def __post_init__(self) -> None:
        if self.boxes.ndim != 2 or self.boxes.shape[1] != 4:
            raise ValueError("boxes must be a (K, 4) array of [x, y, width, height]")
        if self.ssu_ids.ndim != 1:
            raise ValueError("ssu_ids must be a 1D array")
        if len(self.boxes) != len(self.ssu_ids):
            raise ValueError("boxes and ssu_ids must have the same length")
        if not np.issubdtype(self.ssu_ids.dtype, np.integer):
            raise TypeError("ssu_ids must be an integer array")
        if self.ssu_ids.size and np.any(self.ssu_ids <= 0):
            raise ValueError("ssu_ids must be positive; 0 is reserved for background")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("image_width and image_height must be positive")


@dataclass(frozen=True)
class CDDDecomposition:
    """Four-way CDD decomposition result.

    Values are produced by a pluggable metric function (default: sqrt-JSD).
    Any component whose required keys were absent from the input dict is None.

    Valid at both character and word level depending on which TokenPositions
    and token lists were used to build the input distributions.

    Attributes:
        d_pars:  metric(R, Q) — parsing error.
        d_ocr:   metric(S*, Q) — OCR error on GT regions.
        d_int:   metric(S, R)  — interaction error.
        d_total: metric(S, Q)  — total end-to-end error.
    """

    d_pars: Optional[float]
    d_ocr: Optional[float]
    d_int: Optional[float]
    d_total: Optional[float]


@dataclass(frozen=True)
class SpACERDecomposition:
    """Four-way SpACER decomposition result with macro and micro variants.

    Each of the four error components is computed at both macro (page-level
    deletion/insertion counts) and micro (per-box deletion/insertion counts)
    granularity. Both values are produced simultaneously as the computation
    is cheap.

    Any component whose required keys were absent from the input dict is None.

    Keys map to distributions as follows:
        "gt"      -> Q  (full ground truth)
        "parsing" -> R  (GT tokens captured by the parser)
        "ocr"     -> S* (OCR output on GT regions)
        "total"   -> S  (OCR output on predicted regions)

    Attributes:
        d_pars_macro:  SpACER_macro(R, Q) — parsing error, page-level D, I.
        d_pars_micro:  SpACER_micro(R, Q) — parsing error, box-level D, I.
        d_ocr_macro:   SpACER_macro(S*, Q) — OCR error, page-level D, I.
        d_ocr_micro:   SpACER_micro(S*, Q) — OCR error, box-level D, I.
        d_int_macro:   SpACER_macro(S, R)  — interaction error, page-level D, I.
        d_int_micro:   SpACER_micro(S, R)  — interaction error, box-level D, I.
        d_total_macro: SpACER_macro(S, Q)  — total error, page-level D, I.
        d_total_micro: SpACER_micro(S, Q)  — total error, box-level D, I.
    """

    d_pars_macro: Optional[float]
    d_pars_micro: Optional[float]
    d_ocr_macro: Optional[float]
    d_ocr_micro: Optional[float]
    d_int_macro: Optional[float]
    d_int_micro: Optional[float]
    d_total_macro: Optional[float]
    d_total_micro: Optional[float]


@dataclass
class ClassCOTeResult:
    """Results of the class-level COTe decomposition.

    ``classes`` defines the ordering of rows and columns in all matrices
    and the ordering of entries in all share vectors.

    Matrices are indexed [k, l] where k is the prediction class index and
    l is the ground-truth class index.
    """

    classes: List[Label]
    coverage_matrix: np.ndarray  # K×K  C[k,l]: share of class-k pred area on GT that lands on class-l GT
    overlap_matrix: np.ndarray  # K×K  O[k,l]: share of class-k overlapping GT area on class-l GT (asymmetric)
    trespass_matrix: np.ndarray  # K×K  T[k,l]: share of class-k trespass on class-l GT (diagonal = other same-class SSUs)
    coverage_share: np.ndarray  # K    fraction of total coverage attributable to class k
    overlap_share: np.ndarray  # K    fraction of total overlap attributable to class k
    trespass_share: np.ndarray  # K    fraction of total trespass attributable to class k
    coverage_precision: np.ndarray  # K  TP_k / A^P_k — background included, unlike coverage_matrix
    coverage_recall: np.ndarray  # K  TP_k / A^S_k — GT-area-normalised, not in coverage_matrix
    coverage_f1: np.ndarray  # K  harmonic mean of coverage_precision and coverage_recall
    micro_precision: float  # sum(TP_k) / sum(A^P_k) across all classes
    micro_recall: float  # sum(TP_k) / sum(A^S_k) across all classes
    micro_f1: float  # harmonic mean of micro_precision and micro_recall


@dataclass(frozen=True)
class ClassCOTeCounts:
    """Raw, unnormalised per-class pixel sums for one image (or an accumulation
    of several images via :func:`~cotescore.class_metrics.sum_class_counts`).

    Unlike :class:`ClassCOTeResult`, these are summable across images. The
    share vectors are *not* row-sums of the K×K numerators. Coverage and
    overlap shares split each pixel between classes pro rata (weight
    M^p_k / M^p), so that a pixel claimed by predictions of two classes is
    not counted twice; the K×K numerators count it once per class. The
    row-sums of ``trespass_numer`` equal ``trespass_share_numer`` whenever
    every GT SSU's class is in ``classes``; the share numerator is kept
    separately so SSUs outside the taxonomy still count towards the total.
    The three ``*_share_numer`` vectors and the three ``global_*`` scalars
    are tracked as independent accumulable quantities.

    Pass the accumulated totals to
    :func:`~cotescore.class_metrics.finalize_class_counts` to get the final
    matrices, shares, and precision/recall/F1.

    ``classes`` must be identical (same list, same order) across every
    instance that gets summed together — it is the fixed dataset-wide
    taxonomy, not derived per-image.
    """

    classes: List[Label]
    coverage_numer: np.ndarray  # K×K  sum(M^S_l & M^p,b_k)
    pred_area: np.ndarray  # K    A^P_k = sum(M^p,b_k)
    gt_area: np.ndarray  # K    A^S_k = sum(M^S_k)
    overlap_numer: np.ndarray  # K×K  sum(M^O & M^S_l * M^p_k), M^O = [M^p > 1]
    overlap_area: np.ndarray  # K    A^O_k = sum(M^O & M^S * M^p_k)
    trespass_numer: np.ndarray  # K×K  sum_{j in k} sum(M^S_{l\\i(j)} & M^p_j), owner excluded in every column
    coverage_share_numer: np.ndarray  # K  sum(M^S & M^p,b * M^p_k / M^p) — pro-rata coverage
    overlap_share_numer: np.ndarray  # K  sum(M^S * (M^p − M^p,b) * M^p_k / M^p) — pro-rata overlap
    trespass_share_numer: np.ndarray  # K  sum_{j in k} sum(M^S_{\\i(j)} & M^p_j)
    global_coverage_area: int  # sum(M^S & M^p,b_global) — union across all classes
    global_overlap_area: int  # sum(M^S & (M^p_global − M^p,b_global))
    global_trespass_pixels: int  # total trespass pixels across all classes
