import sys
from pathlib import Path
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
pytest.importorskip("pipeline.dofns.transforms", reason="pipeline source is not on this branch")
from pipeline.dofns.transforms import CropImageRegion


def _make_record(image_path, x, y, width, height):
    return {
        "image_id": "test",
        "image_path": str(image_path),
        "box_id": "box1",
        "source": "gt",
        "x": x, "y": y, "width": width, "height": height,
        "class": "text", "confidence": 1.0,
        "ocr_text": None, "ocr_model": None, "image_crop": None,
    }


@pytest.fixture
def test_image(tmp_path):
    img = Image.new("RGB", (200, 100), color=(128, 64, 32))
    img_path = tmp_path / "test.jpg"
    img.save(img_path)
    return img_path


def test_crop_produces_pil_image(test_image):
    record = _make_record(test_image, x=10, y=10, width=50, height=30)
    dofn = CropImageRegion()
    results = list(dofn.process(record))
    assert len(results) == 1
    out = results[0]
    assert out["image_crop"] is not None
    assert isinstance(out["image_crop"], Image.Image)


def test_crop_correct_size(test_image):
    record = _make_record(test_image, x=10, y=10, width=50, height=30)
    dofn = CropImageRegion()
    out = list(dofn.process(record))[0]
    assert out["image_crop"].size == (50, 30)


def test_crop_clamps_to_image_bounds(test_image):
    # Box extends beyond image boundary (image is 200x100)
    record = _make_record(test_image, x=180, y=80, width=100, height=100)
    dofn = CropImageRegion()
    out = list(dofn.process(record))[0]
    crop = out["image_crop"]
    assert crop.size == (20, 20)  # clamped: x=[180,200], y=[80,100]


def test_crop_fully_outside_bounds_yields_nothing(test_image):
    # Box starts at x=300 on a 200-wide image — fully outside
    record = _make_record(test_image, x=300, y=0, width=50, height=50)
    dofn = CropImageRegion()
    results = list(dofn.process(record))
    assert results == []


def test_crop_preserves_other_fields(test_image):
    record = _make_record(test_image, x=0, y=0, width=10, height=10)
    dofn = CropImageRegion()
    out = list(dofn.process(record))[0]
    assert out["box_id"] == "box1"
    assert out["source"] == "gt"


from collections import Counter
from pipeline.dofns.transforms import AggregatePerPage


def _make_qr_record(image_id="p001"):
    return {
        "image_id": image_id,
        "image_path": "/fake/p001.jpg",
        "Q": {"T": 3, "h": 2, "e": 1},
        "R": {"T": 3, "h": 2},
        "gt_textline_texts": ["The", "hello world"],
    }


def _make_box_record(image_id, source, ocr_text):
    return {
        "image_id": image_id,
        "image_path": "/fake/p001.jpg",
        "box_id": f"{source}_box",
        "source": source,
        "x": 0.0, "y": 0.0, "width": 100.0, "height": 30.0,
        "class": "text", "confidence": 1.0,
        "ocr_text": ocr_text, "ocr_model": "mock", "image_crop": None,
    }


def test_aggregate_produces_page_result():
    dofn = AggregatePerPage(ocr_model_name="mock", layout_model_name=None)
    grouped = {
        "gt_boxes": [_make_box_record("p001", "gt", "The")],
        "pred_boxes": [],
        "qr": [_make_qr_record("p001")],
    }
    results = list(dofn.process(("p001", grouped)))
    assert len(results) == 1
    page = results[0]
    assert page["image_id"] == "p001"
    assert "Q" in page and "R" in page
    assert "S_star" in page and "S" in page
    assert "cdd" in page and "spacer" in page


def test_aggregate_cdd_has_four_keys():
    dofn = AggregatePerPage(ocr_model_name="mock", layout_model_name=None)
    grouped = {
        "gt_boxes": [_make_box_record("p001", "gt", "The")],
        "pred_boxes": [],
        "qr": [_make_qr_record()],
    }
    results = list(dofn.process(("p001", grouped)))
    cdd = results[0]["cdd"]
    assert set(cdd.keys()) == {"d_pars", "d_ocr", "d_int", "d_total"}


def test_aggregate_spacer_d_pars_is_none():
    # spacer_decomp cannot compute d_pars (R is Counter, not text list)
    dofn = AggregatePerPage(ocr_model_name="mock", layout_model_name=None)
    grouped = {
        "gt_boxes": [_make_box_record("p001", "gt", "The")],
        "pred_boxes": [],
        "qr": [_make_qr_record()],
    }
    results = list(dofn.process(("p001", grouped)))
    spacer = results[0]["spacer"]
    assert spacer["d_pars_macro"] is None
    assert spacer["d_pars_micro"] is None


def test_aggregate_skips_page_with_no_qr():
    dofn = AggregatePerPage(ocr_model_name="mock", layout_model_name=None)
    grouped = {"gt_boxes": [], "pred_boxes": [], "qr": []}
    results = list(dofn.process(("p001", grouped)))
    assert results == []
