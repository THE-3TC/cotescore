import sys
from pathlib import Path
import numpy as np
import pytest
import json
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
pytest.importorskip("pipeline.dofns.io", reason="pipeline source is not on this branch")
from pipeline.dofns.io import parse_alto_xml, write_json, write_parquet

ALTO_PATH = Path("data/the_spiritualist/ocr_gt_with_ssu/0001_p001.xml")
IMAGE_PATH = Path("data/the_spiritualist/spiritualist_images/0001_p001.jpg")

_spiritualist_data_missing = not ALTO_PATH.exists() or not IMAGE_PATH.exists()
requires_spiritualist_data = pytest.mark.skipif(
    _spiritualist_data_missing,
    reason="Spiritualist dataset not present (data/the_spiritualist/)",
)


@pytest.fixture
def parsed():
    return parse_alto_xml(ALTO_PATH, IMAGE_PATH)


@requires_spiritualist_data
def test_parse_returns_box_records(parsed):
    gt_boxes, _, _ = parsed
    assert len(gt_boxes) > 0
    box = gt_boxes[0]
    assert "image_id" in box
    assert "box_id" in box
    assert box["source"] == "gt"
    for key in ("x", "y", "width", "height"):
        assert key in box
    assert "class" in box


@requires_spiritualist_data
def test_box_id_comes_from_ssu(parsed):
    gt_boxes, _, _ = parsed
    # First TextLine in 0001_p001.xml has SSU="ssu_1_span_tr_1743007782"
    assert gt_boxes[0]["box_id"] == "ssu_1_span_tr_1743007782"


@requires_spiritualist_data
def test_token_positions_are_numpy_arrays(parsed):
    _, tp, _ = parsed
    assert hasattr(tp, "tokens") and hasattr(tp, "xs") and hasattr(tp, "ys")
    assert len(tp.tokens) == len(tp.xs) == len(tp.ys)
    assert len(tp.tokens) > 0


@requires_spiritualist_data
def test_token_positions_have_valid_midpoints(parsed):
    _, tp, _ = parsed
    assert np.all(tp.xs >= 0)
    assert np.all(tp.ys >= 0)


@requires_spiritualist_data
def test_gt_textline_texts_nonempty(parsed):
    _, _, tl_texts = parsed
    assert len(tl_texts) > 0
    assert all(isinstance(t, str) for t in tl_texts)
    assert any(len(t) > 0 for t in tl_texts)


@requires_spiritualist_data
def test_textline_text_contains_content(parsed):
    _, _, tl_texts = parsed
    combined = " ".join(tl_texts)
    assert "Spiritualist" in combined


SAMPLE_PAGE_RESULT = {
    "image_id": "0001_p001",
    "image_path": "/fake/0001_p001.jpg",
    "ocr_model": "tesseract",
    "layout_model": None,
    "Q": {"T": 5, "h": 3},
    "R": {},
    "S_star": {"T": 4},
    "S": {},
    "boxes": [
        {"image_id": "0001_p001", "box_id": "ssu_1", "source": "gt",
         "x": 10.0, "y": 10.0, "width": 100.0, "height": 30.0,
         "class": "text", "confidence": 1.0, "ocr_text": "The", "ocr_model": "tesseract"},
    ],
    "cdd": {"d_pars": None, "d_ocr": 0.05, "d_int": None, "d_total": 0.07},
    "spacer": {"d_pars_macro": None, "d_pars_micro": None,
               "d_ocr_macro": 0.04, "d_ocr_micro": 0.04,
               "d_int_macro": None, "d_int_micro": None,
               "d_total_macro": 0.06, "d_total_micro": 0.06},
}


def test_write_json_creates_file(tmp_path):
    write_json(SAMPLE_PAGE_RESULT, tmp_path)
    expected = tmp_path / "0001_p001.json"
    assert expected.exists()
    data = json.loads(expected.read_text())
    assert data["image_id"] == "0001_p001"
    assert "cdd" in data


def test_write_parquet_creates_file(tmp_path):
    write_parquet([SAMPLE_PAGE_RESULT], tmp_path / "results.parquet")
    assert (tmp_path / "results.parquet").exists()
    df = pd.read_parquet(tmp_path / "results.parquet")
    assert len(df) == 1  # one box row
    assert "ocr_text" in df.columns
    assert "image_id" in df.columns
