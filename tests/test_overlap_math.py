from cotescore import overlap
from cotescore.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks

TOLERANCE = 1e-5


def _overlap(pred, gt, size=100):
    """Rasterize dict boxes on a size x size canvas and return the overlap metric."""
    gt_with_ids = [{**g, "ssu_id": i} for i, g in enumerate(gt, start=1)]
    gt_map = boxes_to_gt_ssu_map(gt_with_ids, size, size, size, size)
    return overlap(gt_map, boxes_to_pred_masks(pred, size, size, size, size))


class TestOverlapMath:
    """
    Strict verification of the Overlap metric against the paper's mathematical definition.

    Formula:
    O = sum(M_S * (M_p - M_p,b)) / A_S   (no (n - 1) normalisation)

    Where:
    - M_S: Binary mask of ground truth
    - M_p: Sum of prediction masks (count of predictions at each pixel)
    - M_p,b: Binary union of prediction masks (1 if covered by any pred, 0 otherwise)
    - M_p - M_p,b: Redundancy count at each pixel (0 if covered 1x, 1 if covered 2x, etc.)
    - A_S: Total area of ground truth
    - n: Number of predictions
    """

    def test_overlap_two_identical_predictions(self):
        """
        Case: 2 identical predictions matching 1 GT perfectly.

        n = 2
        A_S = 100

        At each pixel in GT:
        - M_S = 1
        - M_p = 2 (covered by both preds)
        - M_p,b = 1
        - M_p - M_p,b = 1

        Sum = Area * 1 = 100
        O_raw = 100 / 100 = 1.0
        """
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]
        pred = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 0, "y": 0, "width": 10, "height": 10},
        ]

        result = _overlap(pred, gt)
        assert abs(result - 1.0) < TOLERANCE

    def test_overlap_three_identical_predictions(self):
        """
        Case: 3 identical predictions matching 1 GT perfectly.

        n = 3
        A_S = 100

        At each pixel in GT:
        - M_S = 1
        - M_p = 3
        - M_p,b = 1
        - M_p - M_p,b = 2 (redundancy is 2)

        Sum = Area * 2 = 200
        O_raw = 200 / 100 = 2.0

        This distinguishes linear redundancy (paper) vs combinatorial/pairwise.
        If pairwise logic is used incorrectly: 3 pairs (1-2, 1-3, 2-3).
        Each pair overlaps 100%. Sum = 3 * 100 = 300.
        O (pairwise) = 300 / 100 = 3.0 -> Incorrect!
        """
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]
        pred = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 0, "y": 0, "width": 10, "height": 10},
        ]

        result = _overlap(pred, gt)
        assert abs(result - 2.0) < TOLERANCE

    def test_overlap_disjoint_predictions(self):
        """
        Case: 2 disjoint predictions.

        n = 2
        At any pixel:
        - Max coverage is 1.
        - M_p - M_p,b = 0.

        O = 0.0
        """
        gt = [{"x": 0, "y": 0, "width": 100, "height": 100}]
        pred = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 20, "y": 0, "width": 10, "height": 10},
        ]

        result = _overlap(pred, gt)
        assert abs(result - 0.0) < TOLERANCE

    def test_overlap_partial_overlap(self):
        """
        Case: 2 predictions, one covers 100%, one covers 50% of GT.

        GT: 0,0 10x10. Area = 100.
        P1: 0,0 10x10. (Covers whole GT)
        P2: 0,0 5x10. (Covers left half of GT)

        n = 2

        Region 1 (left half, 5x10, area 50):
        - Covered by P1 and P2.
        - M_p = 2, M_p,b = 1.
        - Redundancy = 1.
        - Contribution = 50 * 1 = 50.

        Region 2 (right half, 5x10, area 50):
        - Covered by P1 only.
        - M_p = 1, M_p,b = 1.
        - Redundancy = 0.
        - Contribution = 0.

        O_raw = 50 / 100 = 0.5
        """
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]
        pred = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 0, "y": 0, "width": 5, "height": 10},
        ]

        result = _overlap(pred, gt)
        assert abs(result - 0.5) < TOLERANCE

    def test_overlap_multi_gt_bridging(self):
        """
        Case: Predictions bridging two GTs.

        GT1: 0,0 10x10.
        GT2: 20,0 10x10.
        Total A_S = 200.

        P1: 0,0 30x10. (Covers GT1, GT2, and gap)
        P2: 0,0 30x10. (Identical)

        n = 2

        GT1 Area (100): P1, P2 cover -> Redundancy 1. Contrib = 100.
        GT2 Area (100): P1, P2 cover -> Redundancy 1. Contrib = 100.
        Gap Area (100): P1, P2 cover. But Gap is NOT in M_S (GT mask).
        So M_S dot (...) is 0 for gap. Contrib = 0.

        Total distinct overlap area = 200.
        O_raw = 200 / 200 = 1.0.
        """
        gt = [
            {"x": 0, "y": 0, "width": 10, "height": 10},
            {"x": 20, "y": 0, "width": 10, "height": 10},
        ]
        pred = [
            {"x": 0, "y": 0, "width": 30, "height": 10},
            {"x": 0, "y": 0, "width": 30, "height": 10},
        ]

        result = _overlap(pred, gt)
        assert abs(result - 1.0) < TOLERANCE

    def test_overlap_ignore_background_overlap(self):
        """
        Case: Overlap occurs only in background, not on GT.

        GT: 0,0 10x10.
        P1: 20,0 10x10.
        P2: 20,0 10x10.

        P1 and P2 overlap perfectly, but they are outside GT.
        M_S is 0 there.
        O_raw should be 0.
        """
        gt = [{"x": 0, "y": 0, "width": 10, "height": 10}]
        pred = [
            {"x": 20, "y": 0, "width": 10, "height": 10},
            {"x": 20, "y": 0, "width": 10, "height": 10},
        ]

        result = _overlap(pred, gt)
        assert abs(result - 0.0) < TOLERANCE
