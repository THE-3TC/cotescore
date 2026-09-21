"""
Tests for the pycocotools-backed MAPMetric.
"""

import numpy as np
import pytest

from cotescore.map_metric import MAPMetric


def _box(x, y, w, h, cls="text", conf=None):
    d = {"x": x, "y": y, "width": w, "height": h, "class": cls}
    if conf is not None:
        d["confidence"] = conf
    return d


class TestMAPMetricBBox:
    def test_perfect_match(self):
        metric = MAPMetric()
        preds = [_box(10, 10, 50, 50, "text", 1.0), _box(100, 100, 30, 30, "figure", 1.0)]
        gt = [_box(10, 10, 50, 50, "text"), _box(100, 100, 30, 30, "figure")]
        metric.update(preds, gt)
        results = metric.compute()

        assert results["map"] == pytest.approx(1.0)
        assert results["map_50"] == pytest.approx(1.0)
        assert results["map_75"] == pytest.approx(1.0)
        assert results["classes"] == pytest.approx({"text": 1.0, "figure": 1.0})

    def test_no_overlap(self):
        metric = MAPMetric()
        metric.update([_box(10, 10, 20, 20, conf=1.0)], [_box(100, 100, 20, 20)])
        results = metric.compute()
        assert results["map"] == pytest.approx(0.0)
        assert results["map_50"] == pytest.approx(0.0)

    def test_class_mismatch(self):
        """Overlapping box with the wrong class is a FP for its class and a FN for the GT class."""
        metric = MAPMetric()
        metric.update([_box(10, 10, 50, 50, "figure", 1.0)], [_box(10, 10, 50, 50, "text")])
        results = metric.compute()

        assert results["map"] == pytest.approx(0.0)
        assert results["classes"]["text"] == pytest.approx(0.0)
        # 'figure' has predictions but no GT: undefined under the COCO convention.
        assert results["classes"]["figure"] == -1.0

    def test_partial_match(self):
        """Shift by 5px on a 50px box: IoU = 45*45 / (2*2500 - 2025) = 0.68."""
        metric = MAPMetric()
        metric.update([_box(15, 15, 50, 50, conf=1.0)], [_box(10, 10, 50, 50)])
        results = metric.compute()

        assert results["map_50"] == pytest.approx(1.0)
        assert results["map_75"] == pytest.approx(0.0)
        # IoU 0.68 passes thresholds 0.50, 0.55, 0.60, 0.65 -> 4 of 10
        assert results["map"] == pytest.approx(0.4)

    def test_interpolated_ap(self):
        """Ranking TP, FP, TP over 2 GT: 101-point interpolated AP = (51*1 + 50*2/3) / 101."""
        metric = MAPMetric()
        preds = [
            _box(0, 0, 10, 10, conf=0.9),      # TP
            _box(500, 500, 10, 10, conf=0.8),  # FP
            _box(100, 100, 10, 10, conf=0.7),  # TP
        ]
        gt = [_box(0, 0, 10, 10), _box(100, 100, 10, 10)]
        metric.update(preds, gt)
        results = metric.compute()

        expected = (51 * 1.0 + 50 * (2 / 3)) / 101
        assert results["map_50"] == pytest.approx(expected)
        assert results["map"] == pytest.approx(expected)  # IoU = 1 at every threshold

    def test_confidence_ordering_matters(self):
        """Same boxes, but the FP now outranks both TPs: AP drops."""
        gt = [_box(0, 0, 10, 10), _box(100, 100, 10, 10)]
        good = MAPMetric()
        good.update([_box(0, 0, 10, 10, conf=0.9), _box(100, 100, 10, 10, conf=0.8),
                     _box(500, 500, 10, 10, conf=0.1)], gt)
        bad = MAPMetric()
        bad.update([_box(0, 0, 10, 10, conf=0.1), _box(100, 100, 10, 10, conf=0.2),
                    _box(500, 500, 10, 10, conf=0.9)], gt)
        assert good.compute()["map_50"] == pytest.approx(1.0)
        assert bad.compute()["map_50"] == pytest.approx(2 / 3)

    def test_multiple_images_accumulate(self):
        """AP is computed over the pooled ranking, not averaged per image."""
        metric = MAPMetric()
        metric.update([_box(0, 0, 10, 10, conf=1.0)], [_box(0, 0, 10, 10)])
        metric.update([_box(500, 500, 10, 10, conf=0.5)], [_box(0, 0, 10, 10)])
        results = metric.compute()
        # 1 TP (rank 1), 1 FP (rank 2), 2 GT -> recall caps at 0.5 with precision 1.0
        assert results["map_50"] == pytest.approx(51 / 101)

    def test_max_dets_not_truncated(self):
        """More than 100 predictions on one image are all scored (COCO default would cap at 100)."""
        n = 150
        gt = [_box(i * 20, 0, 10, 10) for i in range(n)]
        preds = [_box(i * 20, 0, 10, 10, conf=1.0) for i in range(n)]
        m = MAPMetric()
        m.update(preds, gt)
        assert m.compute()["map"] == pytest.approx(1.0)
        capped = MAPMetric(max_dets=(1, 10, 100))
        capped.update(preds, gt)
        assert capped.compute()["map"] < 1.0

    def test_empty_predictions(self):
        metric = MAPMetric()
        metric.update([], [_box(0, 0, 10, 10)])
        results = metric.compute()
        assert results == {"map": 0.0, "map_50": 0.0, "map_75": 0.0, "classes": {"text": 0.0}}

    def test_empty_ground_truth(self):
        metric = MAPMetric()
        metric.update([_box(0, 0, 10, 10, conf=1.0)], [])
        results = metric.compute()
        assert results["map"] == -1.0
        assert results["classes"] == {"text": -1.0}

    def test_reset(self):
        metric = MAPMetric()
        metric.update([_box(500, 500, 10, 10, conf=1.0)], [_box(0, 0, 10, 10)])
        metric.reset()
        metric.update([_box(0, 0, 10, 10, conf=1.0)], [_box(0, 0, 10, 10)])
        assert metric.compute()["map"] == pytest.approx(1.0)

    def test_invalid_iou_type(self):
        with pytest.raises(ValueError):
            MAPMetric(iou_type="polygon")


class TestMAPMetricSegm:
    @staticmethod
    def _mask(h, w, y0, y1, x0, x1):
        m = np.zeros((h, w), dtype=bool)
        m[y0:y1, x0:x1] = True
        return m

    def test_perfect_mask_match(self):
        metric = MAPMetric(iou_type="segm")
        m = self._mask(64, 64, 10, 40, 10, 40)
        metric.update([{"mask": m, "class": "text", "confidence": 1.0}], [{"mask": m, "class": "text"}])
        results = metric.compute()
        assert results["map"] == pytest.approx(1.0)

    def test_partial_mask_match(self):
        """30x30 GT vs 30x30 pred shifted by 5px: IoU = 625 / (1800 - 625) = 0.53."""
        metric = MAPMetric(iou_type="segm")
        gt = self._mask(64, 64, 10, 40, 10, 40)
        pred = self._mask(64, 64, 15, 45, 15, 45)
        metric.update([{"mask": pred, "class": "text", "confidence": 1.0}], [{"mask": gt, "class": "text"}])
        results = metric.compute()
        assert results["map_50"] == pytest.approx(1.0)
        assert results["map_75"] == pytest.approx(0.0)
        assert results["map"] == pytest.approx(0.1)  # only the 0.50 threshold passes

    def test_non_rectangular_mask_differs_from_bbox(self):
        """A diagonal mask fills half its bbox, so segm IoU differs from bbox IoU."""
        h = w = 32
        tri = np.tril(np.ones((h, w), dtype=bool))       # lower triangle
        full = self._mask(h, w, 0, h, 0, w)               # full square (= tri's bbox)

        segm = MAPMetric(iou_type="segm")
        segm.update([{"mask": tri, "class": "t", "confidence": 1.0}], [{"mask": full, "class": "t"}])
        # tri covers (32*33/2)/1024 = 0.516 of the square -> passes only IoU 0.50
        assert segm.compute()["map"] == pytest.approx(0.1)

        bbox = MAPMetric(iou_type="bbox")
        bbox.update([_box(0, 0, w, h, "t", 1.0)], [_box(0, 0, w, h, "t")])
        assert bbox.compute()["map"] == pytest.approx(1.0)

    def test_polygon_segmentation_requires_image_size(self):
        metric = MAPMetric(iou_type="segm")
        poly = [[10, 10, 40, 10, 40, 40, 10, 40]]
        with pytest.raises(ValueError):
            metric.update([{"segmentation": poly, "class": "t", "confidence": 1.0}],
                          [{"segmentation": poly, "class": "t"}])
        metric.update([{"segmentation": poly, "class": "t", "confidence": 1.0}],
                      [{"segmentation": poly, "class": "t"}], image_size=(64, 64))
        assert metric.compute()["map"] == pytest.approx(1.0)
