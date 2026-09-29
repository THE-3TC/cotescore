import pytest
from PIL import Image
pytest.importorskip("pipeline.ocr_models.base", reason="pipeline source is not on this branch")
from pipeline.ocr_models.base import OCRModel, MockOCR


def test_mock_ocr_returns_string():
    model = MockOCR(return_text="hello world")
    model.load()
    img = Image.new("RGB", (100, 50), color="white")
    result = model.run(img)
    assert result == "hello world"


def test_ocr_model_is_abstract():
    with pytest.raises(TypeError):
        OCRModel()  # cannot instantiate abstract class


def test_mock_ocr_default_text():
    model = MockOCR()
    model.load()
    img = Image.new("RGB", (10, 10))
    assert model.run(img) == "mock text"


from pipeline.dofns.ocr import RunOCRModel


def _box_record_with_crop(image_id="p001", box_id="b1"):
    img = Image.new("RGB", (100, 30), color="white")
    return {
        "image_id": image_id,
        "image_path": "/fake/path.jpg",
        "box_id": box_id,
        "source": "gt",
        "x": 0.0, "y": 0.0, "width": 100.0, "height": 30.0,
        "class": "text", "confidence": 1.0,
        "ocr_text": None, "ocr_model": None,
        "image_crop": img,
    }


def test_run_ocr_model_sets_ocr_text():
    mock = MockOCR(return_text="extracted text")
    mock.load()
    dofn = RunOCRModel(mock, model_name="mock")
    record = _box_record_with_crop()
    results = list(dofn.process(record))
    assert len(results) == 1
    assert results[0]["ocr_text"] == "extracted text"
    assert results[0]["ocr_model"] == "mock"


def test_run_ocr_model_drops_image_crop():
    mock = MockOCR()
    mock.load()
    dofn = RunOCRModel(mock, model_name="mock")
    record = _box_record_with_crop()
    results = list(dofn.process(record))
    assert results[0]["image_crop"] is None


def test_run_ocr_model_returns_empty_string_on_failure():
    class FailingOCR(MockOCR):
        def run(self, crop):
            raise RuntimeError("model exploded")

    model = FailingOCR()
    model.load()
    dofn = RunOCRModel(model, model_name="failing")
    record = _box_record_with_crop()
    results = list(dofn.process(record))
    assert len(results) == 1
    assert results[0]["ocr_text"] == ""
