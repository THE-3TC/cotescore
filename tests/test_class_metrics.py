"""Tests for the class-level COTe metrics (class_metrics module)."""

import pytest
import numpy as np

from cotescore.class_metrics import (
    coverage_matrix,
    overlap_matrix,
    trespass_matrix,
    cote_class,
    class_confusion_counts,
    sum_class_counts,
    finalize_class_counts,
)
from cotescore.types import MaskInstance, ClassCOTeResult, ClassCOTeCounts


TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_gt_ssu_map(h: int, w: int, regions: dict) -> np.ndarray:
    """Build a gt_ssu_map from {ssu_id: (row_start, row_end, col_start, col_end)}."""
    gt = np.zeros((h, w), dtype=np.int32)
    for ssu_id, (r0, r1, c0, c1) in regions.items():
        gt[r0:r1, c0:c1] = ssu_id
    return gt


def _mask(h: int, w: int, r0: int, r1: int, c0: int, c1: int, label: str) -> MaskInstance:
    m = np.zeros((h, w), dtype=bool)
    m[r0:r1, c0:c1] = True
    return MaskInstance(mask=m, label=label)


# Canonical 2-class 10×10 scenario:
#   SSU 1 → class "A": rows 0-4 (top half, 50 pixels)
#   SSU 2 → class "B": rows 5-9 (bottom half, 50 pixels)
#   Total GT area A^S = 100 px
SSU_TO_CLASS_2 = {1: "A", 2: "B"}
H, W = 10, 10
GT_2CLASS = _make_gt_ssu_map(H, W, {1: (0, 5, 0, 10), 2: (5, 10, 0, 10)})


# ---------------------------------------------------------------------------
# TestCoverageMatrix
# ---------------------------------------------------------------------------


class TestCoverageMatrix:
    """C[k,l] = sum(ms_l & mp_k_b) / sum(ms & mp_k_b)
    "Of the class-k prediction area on GT, what fraction lands on class-l GT?"
    Rows sum to 1; background is excluded.
    """

    def test_background_excluded_rows_sum_to_one(self):
        """SSU 1 (A) rows 0-4, SSU 2 (B) rows 5-8, row 9 background.
        An A pred over the whole page (100 px): 50 on A, 40 on B, 10 on
        background. C[A,A] = 50/90, C[A,B] = 40/90. Precision keeps the
        background in its denominator: 50/100."""
        gt = _make_gt_ssu_map(H, W, {1: (0, 5, 0, 10), 2: (5, 9, 0, 10)})
        preds = [_mask(H, W, 0, 10, 0, 10, "A")]
        result = cote_class(gt, SSU_TO_CLASS_2, preds)
        a, b = result.classes.index("A"), result.classes.index("B")
        assert abs(result.coverage_matrix[a, a] - 50 / 90) < TOLERANCE
        assert abs(result.coverage_matrix[a, b] - 40 / 90) < TOLERANCE
        assert abs(result.coverage_precision[a] - 0.5) < TOLERANCE

    def test_perfect_within_class_coverage(self):
        """When each pred exactly matches its class GT, diagonal = 1.0."""
        # A pred = 50 px, all on A GT → C[A,A] = 50/50 = 1.0
        # B pred = 50 px, all on B GT → C[B,B] = 50/50 = 1.0
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 5, 10, 0, 10, "B"),
        ]
        C, classes = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        assert abs(C[a, a] - 1.0) < TOLERANCE
        assert abs(C[b, b] - 1.0) < TOLERANCE
        assert abs(C[a, b] - 0.0) < TOLERANCE
        assert abs(C[b, a] - 0.0) < TOLERANCE

    def test_no_predictions(self):
        """All rows are zero when there are no predictions."""
        C, classes = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, [])
        assert C.shape == (2, 2)
        assert np.all(C == 0.0)

    def test_cross_class_coverage(self):
        """A pred covering B GT gives off-diagonal entry."""
        # A pred of 50 px all on B GT → C[A,B] = 50/50 = 1.0, C[A,A] = 0
        preds = [_mask(H, W, 5, 10, 0, 10, "A")]
        C, classes = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        assert abs(C[a, b] - 1.0) < TOLERANCE
        assert abs(C[a, a] - 0.0) < TOLERANCE

    def test_partial_coverage_normalised_by_pred_area(self):
        """Normalization is class-k pred area on GT, not GT area.
        An A pred of 25 px all on A GT → C[A,A] = 25/25 = 1.0 (not 0.5).
        """
        preds = [_mask(H, W, 0, 5, 0, 5, "A")]  # 25 px, all on A's GT
        C, classes = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a = classes.index("A")
        assert abs(C[a, a] - 1.0) < TOLERANCE

    def test_pred_split_across_classes(self):
        """A pred spanning both classes: fractions sum to 1."""
        # A pred covering rows 0-9, cols 0-10 = 100 px
        # 50 px on A GT, 50 px on B GT → C[A,A]=0.5, C[A,B]=0.5
        preds = [_mask(H, W, 0, 10, 0, 10, "A")]
        C, classes = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        assert abs(C[a, a] - 0.5) < TOLERANCE
        assert abs(C[a, b] - 0.5) < TOLERANCE

    def test_matrix_shape(self):
        preds = [_mask(H, W, 0, 5, 0, 10, "A")]
        C, classes = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert C.shape == (len(classes), len(classes))

    def test_unlabelled_prediction_raises(self):
        m = np.zeros((H, W), dtype=bool)
        preds = [MaskInstance(mask=m, label=None)]
        with pytest.raises(ValueError, match="label=None"):
            coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)


# ---------------------------------------------------------------------------
# TestOverlapMatrix
# ---------------------------------------------------------------------------


class TestOverlapMatrix:
    """O[k,l] = sum(M^O & ms_l * mp_k) / A^O_k, with M^O = [mp_global > 1]
    "Of the overlapping GT area of the class-k predictions, what fraction lands
    on class-l GT?" Row k is zero when no class-k prediction lies on overlap.
    """

    def test_cross_class_overlap_appears_in_both_rows(self):
        """An A pred over all GT and a B pred over B's GT overlap on B's 50px.
        Both classes are involved in that overlap, so both rows register it:
        O[A,B] = 50/50 = 1.0 and O[B,B] = 50/50 = 1.0.
        """
        preds = [
            _mask(H, W, 0, 10, 0, 10, "A"),
            _mask(H, W, 5, 10, 0, 10, "B"),
        ]
        O, classes = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        assert abs(O[a, b] - 1.0) < TOLERANCE
        assert abs(O[b, b] - 1.0) < TOLERANCE
        assert abs(O[a, a] - 0.0) < TOLERANCE

    def test_row_independent_of_other_class_pred_count(self):
        """Stacking more B preds on the same overlap must not change row A.
        A spans all GT (A's 50px, B's 50px). One extra A pred sits on A's GT;
        B preds sit on B's GT. Row A is 100px on A GT, 50px on B GT: 2/3, 1/3,
        whether there is one B pred or five.
        """
        base = [
            _mask(H, W, 0, 10, 0, 10, "A"),
            _mask(H, W, 0, 5, 0, 10, "A"),
        ]
        for n_b in (1, 5):
            preds = base + [_mask(H, W, 5, 10, 0, 10, "B")] * n_b
            O, classes = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
            a, b = classes.index("A"), classes.index("B")
            assert abs(O[a, a] - 2 / 3) < TOLERANCE, n_b
            assert abs(O[a, b] - 1 / 3) < TOLERANCE, n_b

    def test_cote_class_and_counts_match_standalone(self):
        preds = [
            _mask(H, W, 0, 10, 0, 10, "A"),
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 5, 10, 0, 10, "B"),
            _mask(H, W, 5, 10, 0, 10, "B"),
        ]
        O, classes = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        via_class = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds).overlap_matrix
        via_counts = finalize_class_counts(
            class_confusion_counts(GT_2CLASS, SSU_TO_CLASS_2, preds, classes)
        ).overlap_matrix
        np.testing.assert_allclose(via_class, O, atol=TOLERANCE)
        np.testing.assert_allclose(via_counts, O, atol=TOLERANCE)

    def test_single_pred_per_class_no_redundancy(self):
        """A single prediction per class produces no redundant area → all rows zero."""
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 5, 10, 0, 10, "B"),
        ]
        O, classes = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert np.allclose(O, 0.0)

    def test_no_predictions_all_zero(self):
        O, _ = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, [])
        assert np.all(O == 0.0)

    def test_within_class_overlap_diagonal(self):
        """Two identical A preds covering A GT: all redundancy on A GT.
        A^O_A = 50 px (rows 0-4 all covered twice). O[A,A] = 50/50 = 1.0.
        """
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 0, 5, 0, 10, "A"),
        ]
        O, classes = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a = classes.index("A")
        assert abs(O[a, a] - 1.0) < TOLERANCE
        # B row stays zero (no B preds)
        b = classes.index("B")
        assert np.allclose(O[b, :], 0.0)

    def test_within_class_overlap_spanning_two_classes(self):
        """Two A preds both spanning rows 0-9: redundant area = 100 px on GT.
        50 px on A GT, 50 px on B GT.
        O[A,A] = 50/100 = 0.5, O[A,B] = 50/100 = 0.5.
        """
        preds = [
            _mask(H, W, 0, 10, 0, 10, "A"),
            _mask(H, W, 0, 10, 0, 10, "A"),
        ]
        O, classes = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        assert abs(O[a, a] - 0.5) < TOLERANCE
        assert abs(O[a, b] - 0.5) < TOLERANCE

    def test_matrix_not_symmetric_in_general(self):
        """Overlap matrix is not required to be symmetric."""
        # Two A preds overlapping only on A GT
        # One B pred (no redundancy)
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 5, 10, 0, 10, "B"),
        ]
        O, classes = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        # O[A,B] > 0 only if A redundancy lands on B GT (it doesn't here)
        assert abs(O[a, b] - 0.0) < TOLERANCE
        # O[B,A] = 0 (no B redundancy)
        assert abs(O[b, a] - 0.0) < TOLERANCE

    def test_matrix_shape(self):
        O, classes = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, [])
        assert O.shape == (len(classes), len(classes))


# ---------------------------------------------------------------------------
# TestTrespassMatrix
# ---------------------------------------------------------------------------


class TestTrespassMatrix:
    """T[k,l] = sum_{j in k} sum(ms_{l\\i(j)} & mp_j) / sum_{j in k} sum(ms_{\\i(j)} & mp_j)
    Rows sum to 1. The owner SSU is excluded in every column. Diagonal is
    trespass against other SSUs of the same class.
    """

    def test_no_predictions(self):
        T, _ = trespass_matrix(GT_2CLASS, SSU_TO_CLASS_2, [])
        assert np.all(T == 0.0)

    def test_no_trespass_when_pred_within_owner_ssu(self):
        """Pred exactly matching its owner SSU has zero trespass on all columns."""
        # A pred covers exactly SSU 1 (A's GT). Owner = SSU 1, no pixels outside it.
        preds = [_mask(H, W, 0, 5, 0, 10, "A")]
        T, classes = trespass_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert np.allclose(T, 0.0)

    def test_misclassified_pred_does_not_trespass_on_its_owner(self):
        """An A pred (50 px) entirely on B's SSU owns that SSU, so it trespasses
        nowhere. The error is a misclassification, which C[A,B] records."""
        preds = [_mask(H, W, 5, 10, 0, 10, "A")]
        T, classes = trespass_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        assert np.allclose(T, 0.0)
        C, _ = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert abs(C[a, b] - 1.0) < TOLERANCE

    def test_owner_excluded_in_off_diagonal_column(self):
        """An A pred on rows 4-9: 10 px on A's SSU 1, 50 px on B's SSU 2, so
        its owner is SSU 2. Only the 10 px on SSU 1 are trespass:
        T[A,A] = 10/10 = 1.0, T[A,B] = 0 (the owner is excluded there too)."""
        preds = [_mask(H, W, 4, 10, 0, 10, "A")]
        T, classes = trespass_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        assert abs(T[a, a] - 1.0) < TOLERANCE
        assert abs(T[a, b] - 0.0) < TOLERANCE

    def test_row_split_between_classes_sums_to_one(self):
        """SSU 1 (A) rows 0-2, SSU 2 (A) rows 3-4, SSU 3 (B) rows 5-9.
        An A pred on rows 0-6 owns SSU 1 (30 px) and trespasses 20 px on
        SSU 2 (A) and 20 px on SSU 3 (B): T[A,A] = T[A,B] = 0.5."""
        gt = _make_gt_ssu_map(H, W, {1: (0, 3, 0, 10), 2: (3, 5, 0, 10), 3: (5, 10, 0, 10)})
        preds = [_mask(H, W, 0, 7, 0, 10, "A")]
        T, classes = trespass_matrix(gt, {1: "A", 2: "A", 3: "B"}, preds)
        a, b = classes.index("A"), classes.index("B")
        assert abs(T[a, a] - 0.5) < TOLERANCE
        assert abs(T[a, b] - 0.5) < TOLERANCE

    def test_partial_off_diagonal_trespass(self):
        """A pred spanning rows 2-6 (50 px total):
          - rows 2-4 → 30 px on A GT (SSU 1)
          - rows 5-6 → 20 px on B GT (SSU 2)
        Owner = SSU 1 (30 px overlap > 20 px). All 20 px of trespass are on B.
        T[A,B] = 20/20 = 1.0 (off-diagonal).
        T[A,A] = 0 (only one A SSU, so no within-class trespass possible).
        """
        preds = [_mask(H, W, 2, 7, 0, 10, "A")]  # rows 2-6: 30 on A GT, 20 on B GT
        T, classes = trespass_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = classes.index("A"), classes.index("B")
        assert abs(T[a, b] - 1.0) < TOLERANCE
        assert abs(T[a, a] - 0.0) < TOLERANCE  # only one A SSU, no within-class trespass

    def test_diagonal_nonzero_within_class_trespass(self):
        """Two A SSUs exist. An A pred covers both → diagonal T[A,A] > 0.
        Setup: 4-class-region map with two separate A SSUs.
          SSU 1 (A): rows 0-2
          SSU 2 (A): rows 3-4
          SSU 3 (B): rows 5-9
        A pred covering rows 0-4 (50 px): owner = SSU 1 or SSU 2 depending on overlap.
        """
        gt = _make_gt_ssu_map(H, W, {1: (0, 3, 0, 10), 2: (3, 5, 0, 10), 3: (5, 10, 0, 10)})
        ssu_to_class = {1: "A", 2: "A", 3: "B"}
        # Pred covers rows 0-4 (50 px). SSU 1 has 30 px, SSU 2 has 20 px → owner = SSU 1.
        # A GT minus SSU 1 = rows 3-4 (SSU 2, 20 px). Pred covers those 20 px.
        # All trespass is on A GT: T[A,A] = 20 / 20 = 1.0
        preds = [_mask(H, W, 0, 5, 0, 10, "A")]
        T, classes = trespass_matrix(gt, ssu_to_class, preds)
        a = classes.index("A")
        assert abs(T[a, a] - 1.0) < TOLERANCE

    def test_matrix_shape(self):
        T, classes = trespass_matrix(GT_2CLASS, SSU_TO_CLASS_2, [])
        assert T.shape == (len(classes), len(classes))


# ---------------------------------------------------------------------------
# TestCOTeClass (combined wrapper)
# ---------------------------------------------------------------------------


class TestCOTeClass:

    def test_returns_correct_type(self):
        preds = [_mask(H, W, 0, 5, 0, 10, "A")]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert isinstance(result, ClassCOTeResult)

    def test_classes_ordered(self):
        preds = [_mask(H, W, 0, 5, 0, 10, "A")]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert result.classes == sorted(result.classes)

    def test_coverage_share_sums_to_one(self):
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 5, 10, 0, 10, "B"),
        ]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert abs(result.coverage_share.sum() - 1.0) < TOLERANCE

    def test_overlap_share_sums_to_one(self):
        # Two A preds create redundancy
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 0, 5, 0, 10, "A"),
        ]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        if result.overlap_share.sum() > 0:
            assert abs(result.overlap_share.sum() - 1.0) < TOLERANCE

    def test_trespass_share_sums_to_one(self):
        preds = [
            _mask(H, W, 5, 10, 0, 10, "A"),  # A pred on B GT
            _mask(H, W, 0, 5, 0, 10, "B"),  # B pred on A GT
        ]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        if result.trespass_share.sum() > 0:
            assert abs(result.trespass_share.sum() - 1.0) < TOLERANCE

    def test_shares_split_cross_class_pixels_pro_rata(self):
        # A pred covers all 50px of A's GT; a B pred sits on 25 of those pixels.
        # The 25 shared pixels count 0.5 to each class, for both the coverage
        # (50px total) and the redundancy (25px total) they carry. Counting them
        # once per class instead would give coverage shares summing to 1.5.
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 0, 5, 0, 5, "B"),
        ]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = result.classes.index("A"), result.classes.index("B")
        assert abs(result.coverage_share[a] - 0.75) < TOLERANCE
        assert abs(result.coverage_share[b] - 0.25) < TOLERANCE
        assert abs(result.overlap_share[a] - 0.5) < TOLERANCE
        assert abs(result.overlap_share[b] - 0.5) < TOLERANCE

    def test_share_vectors_all_zero_when_no_preds(self):
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, [])
        assert np.all(result.coverage_share == 0.0)
        assert np.all(result.overlap_share == 0.0)
        assert np.all(result.trespass_share == 0.0)

    def test_matrices_have_correct_shape(self):
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, [])
        K = len(result.classes)
        assert result.coverage_matrix.shape == (K, K)
        assert result.overlap_matrix.shape == (K, K)
        assert result.trespass_matrix.shape == (K, K)
        assert result.coverage_share.shape == (K,)
        assert result.overlap_share.shape == (K,)
        assert result.trespass_share.shape == (K,)

    def test_integration_perfect_coverage_no_overlap_no_trespass(self):
        """Perfect non-overlapping coverage: diagonal C=1, O=0, T=0."""
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 5, 10, 0, 10, "B"),
        ]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = result.classes.index("A"), result.classes.index("B")

        # Coverage: C[A,A] = 50/50 = 1.0, C[B,B] = 1.0, off-diagonal = 0
        assert abs(result.coverage_matrix[a, a] - 1.0) < TOLERANCE
        assert abs(result.coverage_matrix[b, b] - 1.0) < TOLERANCE
        assert abs(result.coverage_matrix[a, b] - 0.0) < TOLERANCE
        assert abs(result.coverage_matrix[b, a] - 0.0) < TOLERANCE

        # No redundant preds → overlap matrix all zero
        assert np.allclose(result.overlap_matrix, 0.0)

        # No trespass (each pred within its own class GT, owner SSU)
        assert np.allclose(result.trespass_matrix, 0.0)

        # Coverage shares: each class contributes 50/100 of covered GT
        assert abs(result.coverage_share[a] - 0.5) < TOLERANCE
        assert abs(result.coverage_share[b] - 0.5) < TOLERANCE

    def test_coverage_matrix_matches_standalone(self):
        preds = [_mask(H, W, 0, 7, 0, 10, "A"), _mask(H, W, 3, 10, 0, 10, "B")]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        C_standalone, _ = coverage_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert np.allclose(result.coverage_matrix, C_standalone)

    def test_overlap_matrix_matches_standalone(self):
        preds = [_mask(H, W, 0, 5, 0, 10, "A"), _mask(H, W, 0, 5, 0, 10, "A")]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        O_standalone, _ = overlap_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert np.allclose(result.overlap_matrix, O_standalone)

    def test_trespass_matrix_matches_standalone(self):
        preds = [_mask(H, W, 5, 10, 0, 10, "A")]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        T_standalone, _ = trespass_matrix(GT_2CLASS, SSU_TO_CLASS_2, preds)
        assert np.allclose(result.trespass_matrix, T_standalone)


# ---------------------------------------------------------------------------
# TestCoveragePrecisionRecallF1
# ---------------------------------------------------------------------------


class TestCoveragePrecisionRecallF1:
    """coverage_precision[k] = TP_k / A^P_k; coverage_recall[k] = TP_k / A^S_k."""

    def test_precision_matches_coverage_diagonal_without_background(self):
        """Every pred lies on GT here, so the row-normalised C keeps the same
        denominator as precision (A^P_k). With background they differ; see
        TestCoverageMatrix.test_background_excluded_rows_sum_to_one."""
        preds = [_mask(H, W, 0, 7, 0, 10, "A"), _mask(H, W, 3, 10, 0, 10, "B")]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = result.classes.index("A"), result.classes.index("B")
        assert abs(result.coverage_precision[a] - result.coverage_matrix[a, a]) < TOLERANCE
        assert abs(result.coverage_precision[b] - result.coverage_matrix[b, b]) < TOLERANCE

    def test_recall_uses_gt_area_not_pred_area(self):
        # A pred covers only half of A's 50px GT area (25px), all correct.
        # Precision = 25/25 = 1.0 (normalised by pred area); recall = 25/50 = 0.5
        # (normalised by GT area) — these must differ.
        preds = [_mask(H, W, 0, 5, 0, 5, "A")]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a = result.classes.index("A")
        assert abs(result.coverage_precision[a] - 1.0) < TOLERANCE
        assert abs(result.coverage_recall[a] - 0.5) < TOLERANCE

    def test_f1_is_harmonic_mean(self):
        preds = [_mask(H, W, 0, 5, 0, 5, "A")]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a = result.classes.index("A")
        p, r = result.coverage_precision[a], result.coverage_recall[a]
        expected = 2 * p * r / (p + r)
        assert abs(result.coverage_f1[a] - expected) < TOLERANCE

    def test_zero_predictions_gives_zero_precision_and_f1(self):
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, [])
        assert np.all(result.coverage_precision == 0.0)
        assert np.all(result.coverage_f1 == 0.0)
        assert result.micro_precision == 0.0
        assert result.micro_f1 == 0.0

    def test_micro_aggregates_across_classes_not_averages_per_class(self):
        # A: pred = 50px, all correct (TP=50, A^P=50, A^S=50)
        # B: pred = 10px, all correct, but B's GT area is 50px (TP=10, A^P=10, A^S=50)
        # Per-class precision is 1.0/1.0 for both (macro avg would be 1.0), but
        # micro precision must be sum(TP)/sum(A^P) = 60/60 = 1.0 and micro
        # recall = sum(TP)/sum(A^S) = 60/100 = 0.6 (pulled down by B's low recall).
        preds = [
            _mask(H, W, 0, 5, 0, 10, "A"),  # 50px, all on A's 50px GT
            _mask(H, W, 5, 6, 0, 10, "B"),  # 10px, all on B's 50px GT
        ]
        result = cote_class(GT_2CLASS, SSU_TO_CLASS_2, preds)
        a, b = result.classes.index("A"), result.classes.index("B")
        assert abs(result.coverage_precision[a] - 1.0) < TOLERANCE
        assert abs(result.coverage_precision[b] - 1.0) < TOLERANCE
        assert abs(result.micro_precision - 1.0) < TOLERANCE
        assert abs(result.micro_recall - 0.6) < TOLERANCE
        expected_f1 = 2 * 1.0 * 0.6 / (1.0 + 0.6)
        assert abs(result.micro_f1 - expected_f1) < TOLERANCE


# ---------------------------------------------------------------------------
# TestAccumulation
# ---------------------------------------------------------------------------


class TestAccumulation:
    """class_confusion_counts + sum_class_counts + finalize_class_counts must
    match calling cote_class once on the union of the accumulated images —
    this is the multi-page dataset workflow the notebook relies on."""

    CLASSES = ["A", "B"]

    def _two_page_scenario(self):
        """Two 10x10 pages, each with the canonical A/B split, but with
        deliberately mismatched and overlapping predictions (including a
        wrong-class prediction spatially overlapping a same-class one) so
        the cross-class-overlap edge case in trespass_share is exercised."""
        gt1 = GT_2CLASS
        ssu1 = SSU_TO_CLASS_2
        preds1 = [
            _mask(H, W, 0, 5, 0, 10, "A"),
            _mask(H, W, 0, 3, 0, 10, "A"),  # redundant -> overlap
            _mask(H, W, 0, 5, 0, 5, "B"),  # wrong-class pred on A's GT
        ]

        gt2 = GT_2CLASS
        ssu2 = SSU_TO_CLASS_2
        preds2 = [
            _mask(H, W, 5, 10, 0, 10, "B"),
            _mask(H, W, 4, 10, 0, 10, "A"),  # A pred trespassing onto B's GT
        ]
        return (gt1, ssu1, preds1), (gt2, ssu2, preds2)

    def _stitched_reference(self, page1, page2):
        """Stack the two pages vertically into one combined gt_ssu_map (with
        disjoint SSU ids) and call cote_class once, as the ground truth for
        what accumulation should produce."""
        gt1, ssu1, preds1 = page1
        gt2, ssu2, preds2 = page2
        combined_gt = np.zeros((2 * H, W), dtype=np.int32)
        combined_gt[:H, :] = gt1
        combined_gt[H:, :] = np.where(gt2 > 0, gt2 + 10, 0)
        combined_ssu = dict(ssu1)
        combined_ssu.update({sid + 10: cls for sid, cls in ssu2.items()})

        combined_preds = []
        for p in preds1:
            m = np.zeros((2 * H, W), dtype=bool)
            m[:H, :] = p.mask
            combined_preds.append(MaskInstance(mask=m, label=p.label))
        for p in preds2:
            m = np.zeros((2 * H, W), dtype=bool)
            m[H:, :] = p.mask
            combined_preds.append(MaskInstance(mask=m, label=p.label))

        return cote_class(combined_gt, combined_ssu, combined_preds)

    def test_class_confusion_counts_returns_correct_type(self):
        gt, ssu, preds = self._two_page_scenario()[0]
        counts = class_confusion_counts(gt, ssu, preds, self.CLASSES)
        assert isinstance(counts, ClassCOTeCounts)
        assert counts.classes == self.CLASSES
        K = len(self.CLASSES)
        assert counts.coverage_numer.shape == (K, K)
        assert counts.pred_area.shape == (K,)

    def test_accumulation_matches_single_pass_on_stitched_pages(self):
        page1, page2 = self._two_page_scenario()
        c1 = class_confusion_counts(*page1, self.CLASSES)
        c2 = class_confusion_counts(*page2, self.CLASSES)
        total = sum_class_counts(c1, c2)
        result_accum = finalize_class_counts(total)

        result_single = self._stitched_reference(page1, page2)

        for field in (
            "coverage_matrix",
            "overlap_matrix",
            "trespass_matrix",
            "coverage_share",
            "overlap_share",
            "trespass_share",
            "coverage_precision",
            "coverage_recall",
            "coverage_f1",
        ):
            np.testing.assert_allclose(
                getattr(result_accum, field), getattr(result_single, field),
                atol=TOLERANCE, err_msg=field,
            )
        for scalar_field in ("micro_precision", "micro_recall", "micro_f1"):
            assert abs(getattr(result_accum, scalar_field) - getattr(result_single, scalar_field)) < TOLERANCE, scalar_field

    def test_sum_class_counts_rejects_mismatched_classes(self):
        gt, ssu, preds = self._two_page_scenario()[0]
        counts_ab = class_confusion_counts(gt, ssu, preds, ["A", "B"])
        counts_ba = class_confusion_counts(gt, ssu, preds, ["B", "A"])
        with pytest.raises(ValueError, match="same fixed"):
            sum_class_counts(counts_ab, counts_ba)

    def test_predictions_outside_fixed_classes_are_ignored(self):
        gt, ssu, preds = self._two_page_scenario()[0]
        preds_with_extra = preds + [_mask(H, W, 0, 5, 0, 5, "C")]
        counts_without = class_confusion_counts(gt, ssu, preds, self.CLASSES)
        counts_with = class_confusion_counts(gt, ssu, preds_with_extra, self.CLASSES)
        np.testing.assert_allclose(counts_without.coverage_numer, counts_with.coverage_numer)
        np.testing.assert_allclose(counts_without.pred_area, counts_with.pred_area)
