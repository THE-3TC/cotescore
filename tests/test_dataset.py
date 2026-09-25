"""Tests for the NCSE dataset loader."""

import pytest
from pathlib import Path
from unittest.mock import Mock
from cotescore.dataset import NCSEDataset, DocLayNetDataset


class TestNCSEDataset:
    """Test suite for NCSEDataset class."""

    @pytest.fixture
    def dataset_path(self):
        """Provide the path to the test dataset."""
        return Path(__file__).parent.parent / "data" / "ncse"

    @pytest.fixture
    def dataset(self, dataset_path):
        """Create a dataset instance."""
        csv_path = dataset_path / "ncse_testset_bboxes.csv"
        images_dir = dataset_path / "ncse_test_png_120"
        if not csv_path.exists() or not images_dir.exists():
            pytest.skip("NCSE dataset files not present; skipping integration tests")
        return NCSEDataset(dataset_path, split="test")

    def test_dataset_initialization(self, dataset_path):
        """Test dataset initialization."""
        dataset = NCSEDataset(dataset_path, split="test")
        assert dataset.dataset_path == dataset_path
        assert dataset.split == "test"
        assert not dataset._loaded

    def test_dataset_load(self, dataset):
        """Test dataset loading."""
        dataset.load()
        assert dataset._loaded
        assert len(dataset.images) > 0
        assert len(dataset.annotations_by_image) > 0

    def test_dataset_len(self, dataset):
        """Test dataset length."""
        length = len(dataset)
        assert length > 0
        assert dataset._loaded

    def test_dataset_getitem(self, dataset):
        """Test getting an item from the dataset."""
        sample = dataset[0]
        assert "image_path" in sample
        assert "annotations" in sample
        assert "filename" in sample
        assert isinstance(sample["annotations"], list)
        assert len(sample["annotations"]) > 0

    def test_annotation_format(self, dataset):
        """Test that annotations have the correct format."""
        sample = dataset[0]
        annotation = sample["annotations"][0]

        # Check required keys
        assert "x" in annotation
        assert "y" in annotation
        assert "width" in annotation
        assert "height" in annotation
        assert "class" in annotation
        assert "ssu_id" in annotation
        assert "ssu_class" in annotation

        # Check types
        assert isinstance(annotation["x"], float)
        assert isinstance(annotation["y"], float)
        assert isinstance(annotation["width"], float)
        assert isinstance(annotation["height"], float)
        assert isinstance(annotation["ssu_id"], int)

        # Check valid values
        assert annotation["width"] > 0
        assert annotation["height"] > 0

    def test_get_annotations(self, dataset):
        """Test getting annotations by index."""
        annotations = dataset.get_annotations(0)
        assert isinstance(annotations, list)
        assert len(annotations) > 0

    def test_index_out_of_range(self, dataset):
        """Test that accessing invalid index raises IndexError."""
        with pytest.raises(IndexError):
            dataset[len(dataset)]

        with pytest.raises(IndexError):
            dataset[-1 - len(dataset)]

    def test_unsupported_split(self, dataset_path):
        """Test that unsupported split raises ValueError."""
        dataset = NCSEDataset(dataset_path, split="train")
        with pytest.raises(ValueError, match="Only 'test' split is currently supported."):
            dataset.load()

    def test_multiple_annotations_per_image(self, dataset):
        """Test that images can have multiple annotations."""
        # Most newspaper pages should have multiple text regions
        sample = dataset[0]
        assert len(sample["annotations"]) >= 1

    def test_lazy_loading(self, dataset_path):
        """Test that dataset loading is lazy."""
        csv_path = dataset_path / "ncse_testset_bboxes.csv"
        images_dir = dataset_path / "ncse_test_png_120"
        if not csv_path.exists() or not images_dir.exists():
            pytest.skip("NCSE dataset files not present; skipping integration tests")
        dataset = NCSEDataset(dataset_path, split="test")
        assert not dataset._loaded

        # Accessing length triggers loading
        _ = len(dataset)
        assert dataset._loaded

    def test_idempotent_loading(self, dataset):
        """Test that calling load() multiple times is safe."""
        dataset.load()
        initial_length = len(dataset.images)

        dataset.load()
        assert len(dataset.images) == initial_length


class TestFilenameMapping:
    """Test suite for filename mapping logic."""

    @pytest.fixture
    def dataset_instance(self):
        """Create a dataset instance for testing mapping."""
        return NCSEDataset(Path("dummy"), split="test")

    def test_basic_mapping(self, dataset_instance):
        """Test basic filename mapping with standard format."""
        csv_filenames = [
            "EWJ_1858-08-01_page_5.png",
            "TEC_1884-03-15_page_23.png",
        ]
        actual_filenames = [
            "EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png",
            "TEC_pageid_140098_pagenum_23_1884-03-15_page_1.png",
        ]

        mapping = dataset_instance._create_filename_mapping(csv_filenames, actual_filenames)

        assert len(mapping) == 2
        assert (
            mapping["EWJ_1858-08-01_page_5.png"]
            == "EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png"
        )
        assert (
            mapping["TEC_1884-03-15_page_23.png"]
            == "TEC_pageid_140098_pagenum_23_1884-03-15_page_1.png"
        )

    def test_alphanumeric_prefix(self, dataset_instance):
        """Test mapping with alphanumeric prefix like NS2."""
        csv_filenames = ["NS2_1843-04-01_page_4.png"]
        actual_filenames = ["NS2_pageid_163094_pagenum_4_1843-04-01_page_1.png"]

        mapping = dataset_instance._create_filename_mapping(csv_filenames, actual_filenames)

        assert len(mapping) == 1
        assert (
            mapping["NS2_1843-04-01_page_4.png"]
            == "NS2_pageid_163094_pagenum_4_1843-04-01_page_1.png"
        )

    def test_multiple_page_numbers(self, dataset_instance):
        """Test mapping with various page numbers including double digits."""
        csv_filenames = [
            "MRP_1834-06-02_page_1.png",
            "MRP_1834-06-02_page_41.png",
            "TEC_1889-05-01_page_8.png",
        ]
        actual_filenames = [
            "MRP_pageid_100001_pagenum_1_1834-06-02_page_1.png",
            "MRP_pageid_122756_pagenum_41_1834-06-02_page_1.png",
            "TEC_pageid_146451_pagenum_8_1889-05-01_page_1.png",
        ]

        mapping = dataset_instance._create_filename_mapping(csv_filenames, actual_filenames)

        assert len(mapping) == 3
        assert (
            mapping["MRP_1834-06-02_page_1.png"]
            == "MRP_pageid_100001_pagenum_1_1834-06-02_page_1.png"
        )
        assert (
            mapping["MRP_1834-06-02_page_41.png"]
            == "MRP_pageid_122756_pagenum_41_1834-06-02_page_1.png"
        )
        assert (
            mapping["TEC_1889-05-01_page_8.png"]
            == "TEC_pageid_146451_pagenum_8_1889-05-01_page_1.png"
        )

    def test_no_match_found(self, dataset_instance):
        """Test mapping when actual file doesn't exist."""
        csv_filenames = ["EWJ_1858-08-01_page_5.png"]
        actual_filenames = ["TEC_pageid_140098_pagenum_23_1884-03-15_page_1.png"]  # Different file

        mapping = dataset_instance._create_filename_mapping(csv_filenames, actual_filenames)

        assert len(mapping) == 0  # No match should be found

    def test_empty_inputs(self, dataset_instance):
        """Test mapping with empty inputs."""
        # Empty CSV filenames
        mapping = dataset_instance._create_filename_mapping([], ["some_file.png"])
        assert len(mapping) == 0

        # Empty actual filenames
        mapping = dataset_instance._create_filename_mapping(["EWJ_1858-08-01_page_5.png"], [])
        assert len(mapping) == 0

        # Both empty
        mapping = dataset_instance._create_filename_mapping([], [])
        assert len(mapping) == 0

    def test_prefix_matching(self, dataset_instance):
        """Test that prefix matching is strict (not substring matching)."""
        csv_filenames = ["EWJ_1858-08-01_page_5.png"]
        actual_filenames = [
            "EWJX_pageid_91483_pagenum_5_1858-08-01_page_1.png",  # Wrong prefix
            "EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png",  # Correct
        ]

        mapping = dataset_instance._create_filename_mapping(csv_filenames, actual_filenames)

        assert len(mapping) == 1
        assert (
            mapping["EWJ_1858-08-01_page_5.png"]
            == "EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png"
        )

    def test_date_matching(self, dataset_instance):
        """Test that date matching works correctly."""
        csv_filenames = ["EWJ_1858-08-01_page_5.png"]
        actual_filenames = [
            "EWJ_pageid_91483_pagenum_5_1858-08-02_page_1.png",  # Wrong date
            "EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png",  # Correct date
        ]

        mapping = dataset_instance._create_filename_mapping(csv_filenames, actual_filenames)

        assert len(mapping) == 1
        assert (
            mapping["EWJ_1858-08-01_page_5.png"]
            == "EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png"
        )

    def test_page_number_matching(self, dataset_instance):
        """Test that page number matching is exact."""
        csv_filenames = ["EWJ_1858-08-01_page_5.png"]
        actual_filenames = [
            # Wrong page (15 vs 5)
            "EWJ_pageid_91483_pagenum_15_1858-08-01_page_1.png",
            # Correct page (5)
            "EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png",
        ]

        mapping = dataset_instance._create_filename_mapping(csv_filenames, actual_filenames)

        assert len(mapping) == 1
        assert (
            mapping["EWJ_1858-08-01_page_5.png"]
            == "EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png"
        )

    def test_multiple_csv_same_actual(self, dataset_instance):
        """Test that each CSV file maps to only one actual file."""
        csv_filenames = [
            "EWJ_1858-08-01_page_5.png",
            "EWJ_1858-08-01_page_6.png",
        ]
        actual_filenames = [
            "EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png",
            "EWJ_pageid_91484_pagenum_6_1858-08-01_page_1.png",
        ]

        mapping = dataset_instance._create_filename_mapping(csv_filenames, actual_filenames)

        assert len(mapping) == 2
        assert mapping["EWJ_1858-08-01_page_5.png"] != mapping["EWJ_1858-08-01_page_6.png"]

    def test_invalid_csv_filename_format(self, dataset_instance):
        """Test handling of CSV filenames that don't match expected pattern."""
        csv_filenames = [
            "invalid_filename.png",
            "no_date_here.png",
            "123_invalid.png",
        ]
        actual_filenames = [
            "EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png",
        ]

        mapping = dataset_instance._create_filename_mapping(csv_filenames, actual_filenames)

        # None of these should match
        assert len(mapping) == 0

    def test_different_prefixes(self, dataset_instance):
        """Test mapping with different newspaper prefixes."""
        csv_filenames = [
            "EWJ_1858-08-01_page_5.png",
            "TEC_1884-03-15_page_23.png",
            "MRP_1834-06-02_page_41.png",
            "CLD_1855-08-18_page_7.png",
            "TTW_1868-01-25_page_3.png",
        ]
        actual_filenames = [
            "EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png",
            "TEC_pageid_140098_pagenum_23_1884-03-15_page_1.png",
            "MRP_pageid_122756_pagenum_41_1834-06-02_page_1.png",
            "CLD_pageid_108039_pagenum_7_1855-08-18_page_1.png",
            "TTW_pageid_160759_pagenum_3_1868-01-25_page_1.png",
        ]

        mapping = dataset_instance._create_filename_mapping(csv_filenames, actual_filenames)

        assert len(mapping) == 5
        # Each prefix should map correctly
        for csv_name in csv_filenames:
            assert csv_name in mapping

    def test_mapping_with_real_dataset(self, dataset_instance):
        """Test mapping with actual dataset files if available."""
        dataset_path = Path(__file__).parent.parent / "data" / "ncse"

        if not dataset_path.exists():
            pytest.skip("Dataset not available for integration test")

        csv_path = dataset_path / "ncse_testset_bboxes.csv"
        images_dir = dataset_path / "ncse_test_png_120"

        if not csv_path.exists() or not images_dir.exists():
            pytest.skip("Dataset files not found")

        import pandas as pd

        df = pd.read_csv(csv_path)
        csv_filenames = df["filename"].unique().tolist()
        actual_filenames = [f.name for f in images_dir.glob("*.png")]

        dataset = NCSEDataset(dataset_path, split="test")
        mapping = dataset._create_filename_mapping(csv_filenames, actual_filenames)

        # Should map all CSV files to actual files
        assert len(mapping) == len(csv_filenames)
        assert len(mapping) == len(actual_filenames)

        # All values should be in actual_filenames
        for actual_name in mapping.values():
            assert actual_name in actual_filenames

    def test_duplicate_matches_take_first(self, dataset_instance):
        """Test that if multiple actual files match, the first one is used."""
        csv_filenames = ["EWJ_1858-08-01_page_5.png"]
        actual_filenames = [
            "EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png",
            "EWJ_pageid_91484_pagenum_5_1858-08-01_page_1.png",  # Duplicate match
        ]

        mapping = dataset_instance._create_filename_mapping(csv_filenames, actual_filenames)

        # Should map to the first match
        assert len(mapping) == 1
        assert (
            mapping["EWJ_1858-08-01_page_5.png"]
            == "EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png"
        )


class TestDocLayNetDataset:
    """Test suite for DocLayNetDataset."""

    def test_pages_of_same_pdf_are_distinct_samples(self, tmp_path):
        """Pages sharing an ``original_filename`` must keep their own image and GT."""
        datasets = pytest.importorskip("datasets")
        from PIL import Image

        rows = []
        for page_no, (colour, bbox) in enumerate([("red", [10.0, 20.0, 30.0, 40.0]),
                                                   ("blue", [50.0, 60.0, 70.0, 80.0])]):
            rows.append(
                {
                    "image": Image.new("RGB", (100, 100), colour),
                    "bboxes": [bbox],
                    "category_id": [10],
                    "area": [bbox[2] * bbox[3]],
                    "metadata": {
                        "original_filename": "report.pdf",
                        "page_no": page_no,
                        "page_hash": f"{page_no:064x}",
                    },
                }
            )
        ds = datasets.Dataset.from_list(
            rows, features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("float64"))),
                    "category_id": datasets.Sequence(datasets.Value("int64")),
                    "area": datasets.Sequence(datasets.Value("float64")),
                    "metadata": {
                        "original_filename": datasets.Value("string"),
                        "page_no": datasets.Value("int64"),
                        "page_hash": datasets.Value("string"),
                    },
                }
            )
        )
        ds.to_parquet(str(tmp_path / "test-00000-of-00001.parquet"))

        dataset = DocLayNetDataset(tmp_path, split="test")
        samples = [dataset[i] for i in range(len(dataset))]

        assert len({s["image_path"] for s in samples}) == 2
        for s, row in zip(samples, rows):
            with Image.open(s["image_path"]) as img:
                assert img.getpixel((0, 0)) == row["image"].getpixel((0, 0))
            gt = s["annotations"][0]
            assert [gt["x"], gt["y"], gt["width"], gt["height"]] == row["bboxes"][0]
