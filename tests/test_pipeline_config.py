import pytest
from pathlib import Path
pytest.importorskip("pipeline.config", reason="pipeline source is not on this branch")
from pipeline.config import ExperimentConfig, load_config, ConfigError

SAMPLE_YAML = """
experiment:
  name: "test_run"
  description: "test"

data:
  images_dir: "data/the_spiritualist/spiritualist_images"
  alto_dir: "data/the_spiritualist/ocr_gt_with_ssu"
  image_ext: "jpg"

box_sources:
  gt: true
  predicted:
    enabled: false
    layout_model: null
    device: "cpu"

ocr:
  model: "tesseract"
  config:
    lang: "eng"
    psm: 6

output:
  dir: "results/test_run"
  parquet: true
  json: true
"""

YAML_PREDICTED_MISSING_MODEL = """
experiment:
  name: "bad"
  description: ""
data:
  images_dir: "x"
  alto_dir: "y"
  image_ext: "jpg"
box_sources:
  gt: true
  predicted:
    enabled: true
    layout_model: null   # missing — should fail
    device: "cpu"
ocr:
  model: "tesseract"
  config: {}
output:
  dir: "results/bad"
  parquet: true
  json: false
"""

YAML_UNKNOWN_OCR = SAMPLE_YAML.replace('"tesseract"', '"fakeocr"')
YAML_UNKNOWN_LAYOUT = SAMPLE_YAML.replace("enabled: false", "enabled: true").replace("layout_model: null", 'layout_model: "garbage"')


def test_load_valid_config(tmp_path):
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(SAMPLE_YAML)
    cfg = load_config(cfg_file)
    assert cfg.name == "test_run"
    assert cfg.gt_enabled is True
    assert cfg.predicted_enabled is False
    assert cfg.ocr_model == "tesseract"
    assert cfg.ocr_config == {"lang": "eng", "psm": 6}
    assert cfg.output_parquet is True
    assert cfg.output_json is True


def test_predicted_enabled_requires_layout_model(tmp_path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(YAML_PREDICTED_MISSING_MODEL)
    with pytest.raises(ConfigError, match="layout_model"):
        load_config(cfg_file)


def test_unknown_ocr_model_raises(tmp_path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(YAML_UNKNOWN_OCR)
    with pytest.raises(ConfigError, match="fakeocr"):
        load_config(cfg_file)


def test_unknown_layout_model_raises(tmp_path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text(YAML_UNKNOWN_LAYOUT)
    with pytest.raises(ConfigError, match="garbage"):
        load_config(cfg_file)
