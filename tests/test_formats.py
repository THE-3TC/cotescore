import numpy as np
import pytest
from cotescore import GTBoxes, cote_score
from cotescore.layout import _standardize_box_format, _standardize_input_format


class TestBoxFormats:
    """Verify box format conversion to the canonical {'x', 'y', 'width', 'height'} dict."""

    def test_list_xywh_format(self):
        assert _standardize_box_format([20, 20, 40, 40], "xywh") == {
            "x": 20.0,
            "y": 20.0,
            "width": 40.0,
            "height": 40.0,
        }

    def test_tuple_xyxy_format(self):
        assert _standardize_box_format((10, 10, 30, 30), "xyxy") == {
            "x": 10.0,
            "y": 10.0,
            "width": 20.0,
            "height": 20.0,
        }

    def test_cxcywh_format(self):
        assert _standardize_box_format([50, 50, 100, 100], "cxcywh") == {
            "x": 0.0,
            "y": 0.0,
            "width": 100.0,
            "height": 100.0,
        }

    def test_format_is_case_insensitive(self):
        assert _standardize_box_format([1, 2, 3, 4], "XYWH") == {
            "x": 1.0,
            "y": 2.0,
            "width": 3.0,
            "height": 4.0,
        }

    def test_canonical_dict_passes_through(self):
        box = {"x": 1, "y": 2, "width": 3, "height": 4}
        assert _standardize_box_format(box) is box

    def test_explicit_xyxy_dict(self):
        box = {"xmin": 0, "ymin": 0, "xmax": 100, "ymax": 50}
        assert _standardize_box_format(box) == {"x": 0, "y": 0, "width": 100, "height": 50}

    def test_input_format_converts_every_box(self):
        boxes = _standardize_input_format([[0, 0, 10, 10], [5, 5, 15, 20]], "xyxy")
        assert [(b["width"], b["height"]) for b in boxes] == [(10.0, 10.0), (10.0, 15.0)]

    def test_input_format_empty(self):
        assert _standardize_input_format([], "xywh") == []

    def test_mixed_cot_formats(self):
        """xyxy lists converted to xywh give the same COTe score as the xywh boxes."""
        gt_xywh = [[10, 20, 30, 30], [50, 20, 30, 30]]
        pred_xyxy = [[10.0, 20.0, 55.0, 50.0], [45.0, 20.0, 80.0, 50.0]]
        pred_xywh = [
            [b["x"], b["y"], b["width"], b["height"]]
            for b in _standardize_input_format(pred_xyxy, "xyxy")
        ]
        gt = GTBoxes(
            boxes=np.array(gt_xywh, dtype=float),
            ssu_ids=np.array([1, 2]),
            image_width=100,
            image_height=100,
        )

        score, *_ = cote_score(gt, np.array(pred_xywh, dtype=float))
        expected, *_ = cote_score(gt, np.array([[10, 20, 45, 30], [45, 20, 35, 30]], dtype=float))
        assert score == pytest.approx(expected)

    def test_missing_format_raises_error(self):
        with pytest.raises(ValueError, match="format_str must be provided"):
            _standardize_box_format([10, 10, 20, 20])

    def test_invalid_format_raises_error(self):
        with pytest.raises(ValueError, match="Unknown box format"):
            _standardize_box_format([10, 10, 20, 20], "invalid")

    def test_short_list_raises_error(self):
        with pytest.raises(ValueError, match="must have at least 4 elements"):
            _standardize_box_format([10, 10, 20], "xywh")
