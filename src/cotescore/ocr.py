
from collections import Counter
from typing import Callable, List, Dict, NamedTuple, Union, Tuple, Optional
import numpy as np

from cotescore.types import CDDDecomposition, RegionChars, SpACERDecomposition
from cotescore._distributions import PredRegions



# =============================================================================
# Token helpers
# =============================================================================


def text_to_counter(text: Union[str, List[str]], mode: str = "char") -> Counter:
    """Build a token-frequency Counter from text.

    Args:
        text: A single string or a list of strings (e.g. per-box OCR output).
              Lists are concatenated before counting.
        mode: ``'char'`` (default) counts individual characters;
              ``'word'`` splits on whitespace and counts word tokens.

    Returns:
        A Counter mapping token -> frequency.
    """
    if isinstance(text, list):
        joined = "".join(text) if mode == "char" else " ".join(text)
    else:
        joined = text

    if mode == "char":
        return Counter(joined)
    elif mode == "word":
        return Counter(joined.split())
    else:
        raise ValueError(f"mode must be 'char' or 'word', got {mode!r}")


# =============================================================================
# Distribution metrics
# =============================================================================


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability distribution.

    Args:
        p: 1D array of probabilities (must sum to 1).

    Returns:
        Entropy in bits.
    """
    p_nonzero = p[p > 0]
    return float(-np.sum(p_nonzero * np.log2(p_nonzero)))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon Divergence between two probability distributions.

    Args:
        p: 1D probability array.
        q: 1D probability array, same length as p.

    Returns:
        JSD value in [0, 1].
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    if not np.isclose(p.sum(), 1.0) and p.sum() > 0:
        p = p / p.sum()
    if not np.isclose(q.sum(), 1.0) and q.sum() > 0:
        q = q / q.sum()

    if p.sum() == 0 and q.sum() == 0:
        return 0.0
    elif p.sum() == 0:
        p = np.zeros_like(q)
    elif q.sum() == 0:
        q = np.zeros_like(p)

    m = 0.5 * (p + q)
    return float(shannon_entropy(m) - 0.5 * (shannon_entropy(p) + shannon_entropy(q)))


def jsd_distance(p: Counter, q: Counter) -> float:
    """sqrt-JSD between two token-frequency Counters.

    The default metric for :func:`cdd_decomp`. Any callable with the same
    signature ``(Counter, Counter) -> float`` can be substituted, e.g.
    total variation distance, earthmover distance, or angular distance.

    Args:
        p: First token frequency Counter.
        q: Second token frequency Counter.

    Returns:
        sqrt-JSD value in [0, 1].
    """
    all_tokens = sorted(set(p) | set(q))
    if not all_tokens:
        return 0.0

    p_arr = np.array([p.get(t, 0) for t in all_tokens], dtype=np.float64)
    q_arr = np.array([q.get(t, 0) for t in all_tokens], dtype=np.float64)

    p_total = p_arr.sum()
    q_total = q_arr.sum()

    p_dist = p_arr / p_total if p_total > 0 else np.zeros_like(p_arr)
    q_dist = q_arr / q_total if q_total > 0 else np.zeros_like(q_arr)

    return float(np.sqrt(jensen_shannon_divergence(p_dist, q_dist)))


# =============================================================================
# SpACER / SpAWER
#
# Count-based metric, analogous to CER. Kept deliberately separate from the
# distribution-based CDD metrics: SpACER operates on raw token counts, not
# normalised probability distributions.
#
#   SpACER = (E_hat + D + I) / (2 * C)
#
# E_hat is the L1 distance between the reference and predicted count vectors,
# D the net deletion count and I the net insertion count. Mirroring CER's
# (S + D + I) / N, a pure deletion of k tokens, a pure insertion of k tokens
# and k substitutions all score k / C.
#
# =============================================================================


class SpACERCounts(NamedTuple):
    """Raw SpACER components returned by :func:`_spacer_counts`.

    Attributes:
        E_hat:   L1 norm of element-wise count differences (aggregated).
        C:       Total token count in the reference.
        D_macro: max(0, total_ref - total_pred), page-level deletion count.
        I_macro: max(0, total_pred - total_ref), page-level insertion count.
        D_micro: Sum over boxes of max(0, ref_j - pred_j), box-level deletions.
        I_micro: Sum over boxes of max(0, pred_j - ref_j), box-level insertions.
    """

    E_hat: float
    C: float
    D_macro: float
    I_macro: float
    D_micro: float
    I_micro: float


def _spacer_counts(
    ref_boxes: List[Counter],
    pred_boxes: List[Counter],
) -> SpACERCounts:
    """Compute raw SpACER components from per-box token-frequency Counters.

    Args:
        ref_boxes:  Per-box reference Counters (ground truth).
        pred_boxes: Per-box prediction Counters (OCR output).
                    Must have the same length as ref_boxes.

    Returns:
        :class:`SpACERCounts`. At page level exactly one of D_macro / I_macro
        is non-zero and ``D_macro + I_macro == |total_ref - total_pred|``;
        at box level ``D_micro + I_micro == sum_j |ref_j - pred_j|``.
    """
    agg_ref: Counter = Counter()
    agg_pred: Counter = Counter()
    D_micro = 0.0
    I_micro = 0.0

    for ref_c, pred_c in zip(ref_boxes, pred_boxes):
        agg_ref += ref_c
        agg_pred += pred_c
        ref_total = sum(ref_c.values())
        pred_total = sum(pred_c.values())
        D_micro += max(0, ref_total - pred_total)
        I_micro += max(0, pred_total - ref_total)

    all_tokens = set(agg_ref) | set(agg_pred)
    E_hat = sum(abs(agg_ref.get(t, 0) - agg_pred.get(t, 0)) for t in all_tokens)

    C = float(sum(agg_ref.values()))
    total_ref = sum(agg_ref.values())
    total_pred = sum(agg_pred.values())
    D_macro = float(max(0, total_ref - total_pred))
    I_macro = float(max(0, total_pred - total_ref))

    return SpACERCounts(float(E_hat), C, D_macro, I_macro, float(D_micro), float(I_micro))


def _spacer_score(E_hat: float, D: float, I: float, C: float) -> float:
    """Apply the SpACER formula: (E_hat + D + I) / (2 * C)."""
    if C == 0:
        return 0.0
    return (E_hat + D + I) / (2.0 * C)


def spacer(reference: Counter, prediction: Counter) -> float:
    """Macro SpACER between two token-frequency Counters.

    Treats both Counters as single-box inputs so D is computed at page level.
    Equivalent to ``spacer_micro([reference], [prediction])``.

    Args:
        reference:  GT token-frequency Counter.
        prediction: Predicted token-frequency Counter.

    Returns:
        Macro SpACER score. 0.0 when both are empty. Can exceed 1.0.
    """
    k = _spacer_counts([reference], [prediction])
    return _spacer_score(k.E_hat, k.D_macro, k.I_macro, k.C)


def spacer_micro(
    ref_boxes: List[Counter],
    pred_boxes: List[Counter],
) -> float:
    """Micro SpACER from per-box token-frequency Counters.

    Deletions and insertions are accumulated per box before summing, so an
    insertion in one box cannot cancel a deletion in another (as it would in
    the page-level macro count).

    Args:
        ref_boxes:  Per-box GT Counters.
        pred_boxes: Per-box prediction Counters (same length as ref_boxes).

    Returns:
        Micro SpACER score. 0.0 when all boxes are empty. Can exceed 1.0.
    """
    k = _spacer_counts(ref_boxes, pred_boxes)
    return _spacer_score(k.E_hat, k.D_micro, k.I_micro, k.C)


# =============================================================================
# Decomposition API
#
# Both cdd_decomp and spacer_decomp accept the same named-dict interface:
#
#   {
#     "gt":      str | List[str] | Counter,   ->  Q  (full ground truth)
#     "parsing": str | List[str] | Counter,   ->  R  (GT tokens in predicted regions)
#     "ocr":     str | List[str] | Counter,   ->  S* (OCR on GT regions)
#     "total":   str | List[str] | Counter,   ->  S  (OCR on predicted regions)
#   }
#
# Counter values are used directly (no text_to_counter call), allowing
# distributions built from spatial data (e.g. build_R) to be passed in.
#
# Missing keys yield None for the components that require them:
#
#   d_pars  requires "gt" and "parsing"
#   d_ocr   requires "gt" and "ocr"
#   d_int   requires "parsing" and "total"
#   d_total requires "gt" and "total"
#
# =============================================================================


def _normalise_box_texts(value: Union[str, List[str]]) -> List[str]:
    """Wrap a plain string in a list; leave a list unchanged."""
    return [value] if isinstance(value, str) else value


def cdd_decomp(
    named_dict: Dict[str, Union[str, List[str], Counter]],
    metric: Callable[[Counter, Counter], float] = jsd_distance,
    mode: str = "char",
) -> CDDDecomposition:
    """Compute the four-way CDD decomposition from a named text dictionary.

    Args:
        named_dict: Dict with any subset of keys ``"gt"``, ``"parsing"``,
                    ``"ocr"``, ``"total"``. Values may be:

                    - A plain string or list of strings — converted via
                      :func:`text_to_counter` using ``mode``.
                    - A pre-built ``Counter`` — used directly. This is
                      necessary when a distribution was built from spatial
                      data (e.g. ``build_R`` from :mod:`cotescore._distributions`)
                      and cannot be reconstructed from text alone.

        metric:     Callable ``(Counter, Counter) -> float`` used to compare
                    each pair of distributions. Defaults to :func:`jsd_distance`
                    (sqrt-JSD). Any symmetric distribution distance can be
                    substituted (TVD, earthmover, angular distance, etc.).
        mode:       ``'char'`` (default) or ``'word'``. Passed to
                    :func:`text_to_counter` when building Counters from text.
                    Ignored for values that are already Counters.

    Returns:
        :class:`CDDDecomposition` with None for any component whose required
        keys were absent.
    """
    counters: Dict[str, Counter] = {
        k: v if isinstance(v, Counter) else text_to_counter(v, mode)
        for k, v in named_dict.items()
    }

    def _cmp(a: str, b: str) -> Optional[float]:
        if a in counters and b in counters:
            return metric(counters[a], counters[b])
        return None

    return CDDDecomposition(
        d_pars=_cmp("parsing", "gt"),
        d_ocr=_cmp("ocr", "gt"),
        d_int=_cmp("total", "parsing"),
        d_total=_cmp("total", "gt"),
    )


def spacer_decomp(
    named_dict: Dict[str, Union[str, List[str]]],
    mode: str = "char",
) -> SpACERDecomposition:
    """Compute the four-way SpACER decomposition from a named text dictionary.

    Both macro and micro variants are returned for each component since they
    share the same intermediate computation.

    Args:
        named_dict: Dict with any subset of keys ``"gt"``, ``"parsing"``,
                    ``"ocr"``, ``"total"``. Values are plain strings or
                    per-box lists of strings. Per-box lists are required for
                    a meaningful micro score; plain strings produce identical
                    macro and micro values.
        mode:       ``'char'`` (default) or ``'word'``. Passed to
                    :func:`text_to_counter` when building per-box Counters.

    Returns:
        :class:`SpACERDecomposition` with None for any component whose
        required keys were absent.
    """
    box_counters: Dict[str, List[Counter]] = {
        k: [text_to_counter(box, mode) for box in _normalise_box_texts(v)]
        for k, v in named_dict.items()
    }

    def _cmp(ref_key: str, pred_key: str) -> Tuple[Optional[float], Optional[float]]:
        if ref_key not in box_counters or pred_key not in box_counters:
            return None, None
        k = _spacer_counts(box_counters[ref_key], box_counters[pred_key])
        return (
            _spacer_score(k.E_hat, k.D_macro, k.I_macro, k.C),
            _spacer_score(k.E_hat, k.D_micro, k.I_micro, k.C),
        )

    pars_mac, pars_mic = _cmp("gt", "parsing")
    ocr_mac, ocr_mic = _cmp("gt", "ocr")
    int_mac, int_mic = _cmp("parsing", "total")
    tot_mac, tot_mic = _cmp("gt", "total")

    return SpACERDecomposition(
        d_pars_macro=pars_mac,
        d_pars_micro=pars_mic,
        d_ocr_macro=ocr_mac,
        d_ocr_micro=ocr_mic,
        d_int_macro=int_mac,
        d_int_micro=int_mic,
        d_total_macro=tot_mac,
        d_total_micro=tot_mic,
    )


# =============================================================================
# Spatial decomposition API
#
# Full-information CDD / SpACER decomposition from four structured inputs:
#
#   gt_chars          RegionChars  ->  Q  (all GT tokens) and per-GT-region
#                                      GT token counts (S* micro reference)
#   pred_region_pixels RegionPixels -> R  (GT tokens captured by predicted
#                                      regions, via pixel-coordinate join)
#   pred_gt_ocr       Dict[int,str] -> S* (OCR on GT regions, keyed by
#                                      gt_region_id)
#   pred_parse_ocr    Dict[int,str] -> S  (OCR on predicted regions, keyed
#                                      by pred_region_id)
#
# d_pars_micro is None: GT and predicted regions use different id spaces with
# no natural pairing, so per-box deletions and insertions cannot be
# meaningfully accumulated.
# All other macro and micro components are computed when data is available.
#
# =============================================================================


def cdd_decomp_spatial(
    gt_chars: RegionChars,
    pred_region_pixels: PredRegions,
    pred_gt_ocr: Dict[int, str],
    pred_parse_ocr: Dict[int, str],
    metric: Callable[[Counter, Counter], float] = jsd_distance,
    mode: str = "char",
) -> CDDDecomposition:
    """CDD decomposition from four structured spatial inputs.

    Builds the Q, R, S*, and S distributions internally and delegates to
    :func:`cdd_decomp`.

    Args:
        gt_chars:           GT characters with pixel midpoints and GT region
                            ids. Provides Q and the per-GT-region reference
                            counts used to isolate OCR error.
        pred_region_pixels: Pixel membership for predicted regions. Used to
                            build R via a pixel-coordinate join with
                            ``gt_chars``.
        pred_gt_ocr:        OCR output on GT regions, keyed by GT region id.
                            Builds S* (OCR error in isolation).
        pred_parse_ocr:     OCR output on predicted regions, keyed by
                            predicted region id. Builds S (combined error).
        metric:             Callable ``(Counter, Counter) -> float``.
                            Defaults to :func:`jsd_distance` (sqrt-JSD).
        mode:               ``'char'`` (default) or ``'word'``. Passed to
                            :func:`text_to_counter` when tokenising OCR text.

    Returns:
        :class:`CDDDecomposition`. Components requiring absent data are None.
    """
    from cotescore._distributions import build_R_spatial

    Q = Counter(gt_chars.tokens.tolist())
    R_agg, _ = build_R_spatial(gt_chars, pred_region_pixels)
    S_star = text_to_counter(list(pred_gt_ocr.values()), mode) if pred_gt_ocr else Counter()
    S = text_to_counter(list(pred_parse_ocr.values()), mode) if pred_parse_ocr else Counter()

    return cdd_decomp(
        {"gt": Q, "parsing": R_agg, "ocr": S_star, "total": S},
        metric=metric,
        mode=mode,
    )


def spacer_decomp_spatial(
    gt_chars: RegionChars,
    pred_region_pixels: PredRegions,
    pred_gt_ocr: Dict[int, str],
    pred_parse_ocr: Dict[int, str],
    mode: str = "char",
) -> SpACERDecomposition:
    """SpACER decomposition from four structured spatial inputs.

    Builds per-region paired lists for micro scores where a natural pairing
    exists. ``d_pars_micro`` is always ``None`` because GT and predicted
    regions use different id spaces with no natural 1:1 pairing.

    Args:
        gt_chars:           GT characters with pixel midpoints and GT region
                            ids.
        pred_region_pixels: Pixel membership for predicted regions.
        pred_gt_ocr:        OCR output on GT regions keyed by GT region id.
        pred_parse_ocr:     OCR output on predicted regions keyed by
                            predicted region id.
        mode:               ``'char'`` (default) or ``'word'``.

    Returns:
        :class:`SpACERDecomposition`. ``d_pars_micro`` is always ``None``.
        Other components requiring absent data are also ``None``.
    """
    from cotescore._distributions import build_R_spatial

    Q = Counter(gt_chars.tokens.tolist())
    R_agg, R_per_pred = build_R_spatial(gt_chars, pred_region_pixels)

    # --- d_pars (macro only) ---
    pars_mac: Optional[float] = None
    if Q or R_agg:
        k = _spacer_counts([Q], [R_agg])
        pars_mac = _spacer_score(k.E_hat, k.D_macro, k.I_macro, k.C)

    # --- d_ocr (macro + micro, paired on gt_region_id) ---
    ocr_mac: Optional[float] = None
    ocr_mic: Optional[float] = None
    if pred_gt_ocr:
        # Per-GT-region GT char counts, keyed by region_id.
        gt_per_region: Dict[int, Counter] = {}
        for tok, rid in zip(gt_chars.tokens.tolist(), gt_chars.region_ids.tolist()):
            if rid not in gt_per_region:
                gt_per_region[rid] = Counter()
            gt_per_region[rid][tok] += 1

        all_gt_ids = sorted(set(gt_per_region) | set(pred_gt_ocr))
        ocr_ref = [gt_per_region.get(rid, Counter()) for rid in all_gt_ids]
        ocr_pred = [
            text_to_counter(pred_gt_ocr[rid], mode) if rid in pred_gt_ocr else Counter()
            for rid in all_gt_ids
        ]
        k = _spacer_counts(ocr_ref, ocr_pred)
        ocr_mac = _spacer_score(k.E_hat, k.D_macro, k.I_macro, k.C)
        ocr_mic = _spacer_score(k.E_hat, k.D_micro, k.I_micro, k.C)

    # --- d_int and d_total (macro + micro, paired on pred_region_id) ---
    int_mac: Optional[float] = None
    int_mic: Optional[float] = None
    tot_mac: Optional[float] = None
    tot_mic: Optional[float] = None
    if pred_parse_ocr:
        all_pred_ids = sorted(set(R_per_pred) | set(pred_parse_ocr))
        pred_ref = [R_per_pred.get(rid, Counter()) for rid in all_pred_ids]
        pred_s = [
            text_to_counter(pred_parse_ocr[rid], mode) if rid in pred_parse_ocr else Counter()
            for rid in all_pred_ids
        ]

        # d_int: R vs S per predicted region.
        k = _spacer_counts(pred_ref, pred_s)
        int_mac = _spacer_score(k.E_hat, k.D_macro, k.I_macro, k.C)
        int_mic = _spacer_score(k.E_hat, k.D_micro, k.I_micro, k.C)

        # d_total: Q (aggregate) vs S (aggregate + per predicted region).
        # D_micro / I_micro use R_j as reference (= Q chars in predicted
        # region j), identical to d_int_micro. E_hat, D_macro and I_macro
        # differ: they use Q vs S_agg.
        S_agg = Counter(tok for c in pred_s for tok in c.elements())
        all_tokens = set(Q) | set(S_agg)
        E_hat_total = float(sum(abs(Q.get(t, 0) - S_agg.get(t, 0)) for t in all_tokens))
        C_total = float(sum(Q.values()))
        n_S = sum(S_agg.values())
        D_total_mac = float(max(0, C_total - n_S))
        I_total_mac = float(max(0, n_S - C_total))
        tot_mac = _spacer_score(E_hat_total, D_total_mac, I_total_mac, C_total)
        tot_mic = _spacer_score(E_hat_total, k.D_micro, k.I_micro, C_total)

    return SpACERDecomposition(
        d_pars_macro=pars_mac,
        d_pars_micro=None,
        d_ocr_macro=ocr_mac,
        d_ocr_micro=ocr_mic,
        d_int_macro=int_mac,
        d_int_micro=int_mic,
        d_total_macro=tot_mac,
        d_total_micro=tot_mic,
    )