"""
Dataset loading utilities for the NCSE v2 dataset.

This module provides functionality for loading and preprocessing the
NCSE v2.0 Dataset of OCR-Processed 19th Century English Newspapers.
"""

from importlib.resources import files, as_file
from pathlib import Path
from typing import Dict, List, Any, Optional, Sequence, Tuple
import re
import json
import logging
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


class NCSEDataset:
    """Loader for the NCSE v2 dataset."""

    def __init__(
        self,
        dataset_path: Path,
        split: str = "test",
        csv_filename: Optional[str] = None,
        images_subdir: Optional[str] = None,
        image_ext: str = "png",
    ):
        """
        Initialize the NCSE dataset loader.

        Args:
            dataset_path: Path to the NCSE dataset directory
            split: Dataset split to load ('test' is currently supported)
            csv_filename: Name of the annotations CSV file (default: 'ncse_testset_bboxes.csv')
            images_subdir: Name of the images subdirectory (default: 'ncse_test_png_120')
            image_ext: Image file extension to glob for (default: 'png')
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.csv_filename = csv_filename or "ncse_testset_bboxes.csv"
        self.images_subdir = images_subdir or "ncse_test_png_120"
        self.image_ext = image_ext
        self.images = []
        self.annotations_by_image = {}
        self._loaded = False

    def load(self):
        """Load the dataset from disk."""
        if self._loaded:
            return

        if self.split != "test":
            raise ValueError("Only 'test' split is currently supported.")

        csv_path = self.dataset_path / self.csv_filename
        images_dir = self.dataset_path / self.images_subdir

        if not csv_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {csv_path}")

        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        df = pd.read_csv(csv_path)

        actual_files = {f.name: f for f in images_dir.glob(f"*.{self.image_ext}")}

        # NOTE: Filenames in the ground-truth CSV are now expected to match the actual image
        # filenames in `images_dir` exactly (no translation/mapping required).
        #
        # The previous mapping logic is kept commented out for safety and to make it easy
        # to re-enable if a future dataset release reintroduces differing filename formats.
        # filename_mapping = self._create_filename_mapping(
        #     df["filename"].unique(), list(actual_files.keys())
        # )

        for csv_filename in df["filename"].unique():
            # Previous mapping-based resolution (kept for reference):
            # actual_filename = filename_mapping.get(csv_filename)
            # if not actual_filename:
            #     continue

            if csv_filename not in actual_files:
                continue

            image_path = images_dir / csv_filename
            if not image_path.exists():
                continue

            self.images.append(str(image_path))

            image_annotations = df[df["filename"] == csv_filename]
            annotations = self._build_annotations(image_annotations)

            # Rescale GT coordinates if the annotations were recorded at a different
            # resolution than the image on disk (mirrors HNLA2013Dataset behaviour).
            if "image_width" in image_annotations.columns:
                csv_ann_w = int(image_annotations["image_width"].iloc[0])
                with Image.open(image_path) as im:
                    img_w, img_h = im.size
                if img_w != csv_ann_w and csv_ann_w > 0:
                    scale = img_w / csv_ann_w
                    for ann in annotations:
                        ann["x"] *= scale
                        ann["y"] *= scale
                        ann["width"] *= scale
                        ann["height"] *= scale

            self.annotations_by_image[str(image_path)] = annotations

        self._loaded = True

    def _create_filename_mapping(self, csv_filenames: list, actual_filenames: list) -> dict:
        """
        Create a mapping from CSV filenames to actual filenames in the directory.

        CSV format: {prefix}_{date}_page_{num}.png
            Example: EWJ_1858-08-01_page_5.png

        Actual format: {prefix}_pageid_{id}_pagenum_{num}_{date}_page_1.png
            Example: EWJ_pageid_91483_pagenum_5_1858-08-01_page_1.png

        The mapping matches on three components: prefix, date, and page number.

        Args:
            csv_filenames: List of filenames from CSV
            actual_filenames: List of actual filenames in directory

        Returns:
            Dictionary mapping CSV filenames to actual filenames
        """
        mapping = {}
        unmatched = []

        # Pattern to extract: PREFIX, DATE (YYYY-MM-DD), PAGE_NUM from CSV filename
        # Handles formats like: EWJ_1858-08-01_page_5.png, TEC_1884-03-15_page_23.png, NS2_1843-04-01_page_4.png
        # PREFIX can be letters and numbers (e.g., EWJ, TEC, NS2)
        csv_pattern = re.compile(r"([A-Z0-9]+)_.*?(\d{4}-\d{2}-\d{2})_page_(\d+)\.png")

        for csv_name in csv_filenames:
            match = csv_pattern.search(csv_name)
            if not match:
                logger.warning(f"Could not parse CSV filename: {csv_name}")
                unmatched.append(csv_name)
                continue

            prefix, date, page_num = match.groups()

            matching_file = self._find_matching_file(actual_filenames, prefix, page_num, date)

            if not matching_file:
                logger.warning(
                    f"No matching file found for {csv_name} "
                    f"(looking for: {prefix}_*_pagenum_{page_num}_*_{date}_*)"
                )
                unmatched.append(csv_name)
                continue

            mapping[csv_name] = matching_file

        logger.info(f"Mapped {len(mapping)}/{len(csv_filenames)}" f"CSV files to actual files")
        if unmatched:
            logger.warning(f"Failed to map {len(unmatched)}")

        return mapping

    @staticmethod
    def _find_matching_file(
        actual_filenames: list, prefix: str, page_num: str, date: str
    ) -> Optional[str]:
        """
        Find the actual filename that matches the given CSV components.

        Args:
            actual_filenames: List of actual filenames in the directory
            prefix: Publication prefix (e.g., 'EWJ', 'TEC')
            page_num: Page number as string
            date: Date in YYYY-MM-DD format

        Returns:
            Matching filename or None if not found
        """
        for actual_name in actual_filenames:
            if (
                actual_name.startswith(f"{prefix}_")
                and f"pagenum_{page_num}_" in actual_name
                and date in actual_name
            ):
                return actual_name
        return None

    @staticmethod
    def _build_annotations(image_annotations: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Build annotation dictionaries from DataFrame rows.

        Converts from (x1, y1, x2, y2) format to (x, y, width, height) format.

        Args:
            image_annotations: DataFrame rows for a single image

        Returns:
            List of annotation dictionaries
        """
        required_cols = {"x1", "y1", "x2", "y2", "class", "ssu_id", "ssu_class"}
        missing = required_cols - set(image_annotations.columns)
        if missing:
            raise ValueError(
                "NCSE annotations CSV is missing required columns: " + ", ".join(sorted(missing))
            )

        annotations = []
        for _, row in image_annotations.iterrows():
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            annotation = {
                "x": float(x1),
                "y": float(y1),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "class": row["class"],
                "ssu_id": int(row["ssu_id"]),
                "ssu_class": row["ssu_class"],
                "confidence": row.get("confidence", 1.0),
                "page_id": row.get("page_id", ""),
            }
            annotations.append(annotation)
        return annotations

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if not self._loaded:
            self.load()
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing image path and annotations
        """
        if not self._loaded:
            self.load()

        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size " f"{len(self.images)}")

        image_path = self.images[idx]
        return {
            "image_path": image_path,
            "annotations": self.annotations_by_image[image_path],
            "filename": Path(image_path).name,
        }

    def get_annotations(self, idx: int) -> List[Dict[str, Any]]:
        """
        Get ground truth annotations for a sample.

        Args:
            idx: Index of the sample

        Returns:
            List of ground truth regions with bounding boxes and labels
        """
        if not self._loaded:
            self.load()

        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size " f"{len(self.images)}")

        return self.annotations_by_image[self.images[idx]]


class DocLayNetDataset:
    """Loader for the DocLayNet dataset from local HuggingFace parquet files."""

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        split: str = "test",
    ):
        """
        Initialize the DocLayNet dataset loader.

        Args:
            dataset_path: Path to a directory containing DocLayNet HuggingFace parquet
                files (e.g. ``test-00001-of-00001.parquet``).  Images are extracted from
                the parquet rows and cached under ``dataset_path/PNG/``.
                If None or the directory contains no parquet files, the dataset is
                downloaded from HuggingFace (``docling-project/DocLayNet-v1.2``) and
                images are cached under ``/teamspace/lightning_storage/doclayout/PNG/``.
            split: Dataset split to load ('train', 'val', or 'test').  Used both as a
                logging label and (when downloading from HF) to select the correct split.
        """
        self.dataset_path = Path(dataset_path) if dataset_path is not None else None
        self.split = split
        _cache_root = (
            self.dataset_path
            if self.dataset_path is not None
            else Path("/teamspace/lightning_storage/doclayout")
        )
        self.images_dir = _cache_root / "PNG"

        self.images = []
        self.annotations_by_image = {}
        self._loaded = False

        # DocLayNet-v1.1/v1.2 HF category ID mapping
        self.category_names = {
            1: "Caption",
            2: "Footnote",
            3: "Formula",
            4: "List-item",
            5: "Page-footer",
            6: "Page-header",
            7: "Picture",
            8: "Section-header",
            9: "Table",
            10: "Text",
            11: "Title",
        }

    def load(self):
        """Load the dataset, from local parquet files if available, else from HuggingFace."""
        if self._loaded:
            return

        import datasets

        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Determine whether local parquet files are available
        local_parquet = (
            sorted(self.dataset_path.glob("*.parquet")) if self.dataset_path is not None else []
        )

        if local_parquet:
            logger.info(
                f"Loading DocLayNet from {self.dataset_path} ({len(local_parquet)} parquet files)..."
            )
            ds = datasets.load_dataset(
                "parquet",
                data_files=[str(p) for p in local_parquet],
                split="train",
            )
        else:
            # Fall back to HuggingFace download
            hf_split = "validation" if self.split == "val" else self.split
            logger.info(
                f"No local parquet files found — downloading DocLayNet-v1.2 '{hf_split}' split from HuggingFace..."
            )
            ds = datasets.load_dataset(
                "docling-project/DocLayNet-v1.2",
                data_files=f"data/{hf_split}-*.parquet",
                split="train",
                verification_mode="no_checks",
            )

        for i, row in enumerate(ds):
            # Row has: image, bboxes, category_id, area, metadata
            metadata = row.get("metadata") or {}
            # ``original_filename`` is the source PDF, shared by every page taken from
            # it, so the page number and hash are needed to give each page its own file.
            if "original_filename" in metadata and "page_hash" in metadata:
                stem = Path(metadata["original_filename"]).stem
                page_filename = f"{stem}_p{metadata.get('page_no', 0)}_{metadata['page_hash'][:12]}.png"
            else:
                page_filename = f"doclaynet_{self.split}_{i}.png"

            image_path = self.images_dir / page_filename

            # Cache image to disk if not already present
            if not image_path.exists():
                img = row["image"]
                img.save(image_path)

            self.images.append(str(image_path))

            bboxes = row.get("bboxes", [])
            categories = row.get("category_id", [])
            areas = row.get("area", [])

            annotations = []
            for j in range(len(bboxes)):
                # Bbox format in DocLayNet is COCO: [x, y, width, height]
                bbox = bboxes[j]
                if len(bbox) != 4:
                    continue

                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

                cat_id = categories[j] if j < len(categories) else 0
                class_name = self.category_names.get(cat_id, str(cat_id))
                area = areas[j] if j < len(areas) else (w * h)

                annotations.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "width": float(w),
                        "height": float(h),
                        "class": class_name,
                        "ssu_id": j + 1,
                        "ssu_class": "object",
                        "confidence": 1.0,
                        "area": float(area),
                    }
                )

            self.annotations_by_image[str(image_path)] = annotations

        self._loaded = True

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if not self._loaded:
            self.load()
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing image path and annotations
        """
        if not self._loaded:
            self.load()

        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size " f"{len(self.images)}")

        image_path = self.images[idx]
        return {
            "image_path": image_path,
            "annotations": self.annotations_by_image[image_path],
            "filename": Path(image_path).name,
        }

    def get_annotations(self, idx: int) -> List[Dict[str, Any]]:
        """
        Get ground truth annotations for a sample.

        Args:
            idx: Index of the sample

        Returns:
            List of ground truth regions with bounding boxes and labels
        """
        if not self._loaded:
            self.load()

        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size " f"{len(self.images)}")

        return self.annotations_by_image[self.images[idx]]


class HNLA2013Dataset:
    """Loader for the HNLA2013 dataset with SSU-tagged PAGE XML ground truth."""

    _NS = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19"}
    _SSU_ID_RE = re.compile(r"id:([^;]+);")

    def __init__(
        self,
        images_path: Path,
        groundtruth_path: Path,
        image_ext: str = "png",
    ):
        """
        Initialize the HNLA2013 dataset loader.

        Args:
            images_path: Flat directory containing TIFF image files
            groundtruth_path: Directory containing PAGE XML files with SSU annotations
                              (e.g. groundtruth_with_ssu/)
            image_ext: Image file extension to glob for (default: 'png')
        """
        self.images_path = Path(images_path)
        self.groundtruth_path = Path(groundtruth_path)
        self.image_ext = image_ext
        self.images = []
        self.annotations_by_image = {}
        self._loaded = False

    def load(self):
        """Load the dataset from disk."""
        if self._loaded:
            return

        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.groundtruth_path.exists():
            raise FileNotFoundError(f"Ground truth directory not found: {self.groundtruth_path}")

        # Build stem -> xml_path map, stripping the "pc-" prefix used in HNLA2013 XML filenames
        xml_by_stem = {}
        for xml_path in self.groundtruth_path.glob("*.xml"):
            stem = xml_path.stem
            if stem.startswith("pc-"):
                stem = stem[3:]
            xml_by_stem[stem] = xml_path

        for img_path in sorted(self.images_path.glob(f"*.{self.image_ext}")):
            xml_path = xml_by_stem.get(img_path.stem)
            if xml_path is None:
                logger.warning(f"No ground truth XML found for {img_path.name}, skipping")
                continue

            annotations, orig_w, orig_h = self._parse_xml(xml_path, img_path.stem)

            # If the image on disk has been downsampled (e.g. TIFF → smaller PNG),
            # rescale GT coordinates from original XML space to the image file space
            # so that GT and model predictions share the same coordinate system.
            if orig_w > 0:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
                if img_w != orig_w:
                    scale = img_w / orig_w
                    for ann in annotations:
                        ann["x"] *= scale
                        ann["y"] *= scale
                        ann["width"] *= scale
                        ann["height"] *= scale

            self.images.append(str(img_path))
            self.annotations_by_image[str(img_path)] = annotations

        self._loaded = True

    def _parse_xml(self, xml_path: Path, page_id: str) -> tuple:
        """Parse a PAGE XML file and return (annotations, orig_width, orig_height)."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        ns = self._NS

        page_elem = root.find("page:Page", ns)
        orig_w = int(page_elem.get("imageWidth", 0)) if page_elem is not None else 0
        orig_h = int(page_elem.get("imageHeight", 0)) if page_elem is not None else 0

        regions = root.findall(".//page:TextRegion", ns)

        # Assign stable per-page integer SSU IDs (1-indexed; 0 = background)
        ssu_strings: List[str] = []
        for region in regions:
            m = self._SSU_ID_RE.search(region.get("custom", ""))
            if m:
                ssu_str = m.group(1).strip()
                if ssu_str not in ssu_strings:
                    ssu_strings.append(ssu_str)
        ssu_to_int = {s: i + 1 for i, s in enumerate(ssu_strings)}

        annotations = []
        for region in regions:
            coords_elem = region.find("page:Coords", ns)
            if coords_elem is None:
                continue
            points_str = coords_elem.get("points", "")
            try:
                if points_str:
                    points = [tuple(int(v) for v in pt.split(",")) for pt in points_str.split()]
                else:
                    points = [
                        (int(pt.get("x")), int(pt.get("y")))
                        for pt in coords_elem.findall("page:Point", ns)
                    ]
                if not points:
                    continue
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            except (ValueError, IndexError):
                logger.warning(f"Could not parse Coords in {xml_path.name}, skipping region")
                continue

            m = self._SSU_ID_RE.search(region.get("custom", ""))
            ssu_str = m.group(1).strip() if m else None
            ssu_id = ssu_to_int.get(ssu_str, 0) if ssu_str else 0

            annotations.append(
                {
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                    "class": region.get("type", "text"),
                    "ssu_id": ssu_id,
                    "ssu_class": "object",
                    "confidence": 1.0,
                    "page_id": page_id,
                }
            )

        return annotations, orig_w, orig_h

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        if not self._loaded:
            self.load()
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.

        Args:
            idx: Zero-based index of the sample to retrieve.

        Returns:
            A dict with keys ``"image_path"``, ``"annotations"``, and
            ``"filename"``.

        Raises:
            IndexError: If ``idx`` is out of range.
        """
        if not self._loaded:
            self.load()
        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")
        image_path = self.images[idx]
        return {
            "image_path": image_path,
            "annotations": self.annotations_by_image[image_path],
            "filename": Path(image_path).name,
        }

    def get_annotations(self, idx: int) -> List[Dict[str, Any]]:
        """Get ground-truth annotations for a sample.

        Args:
            idx: Zero-based index of the sample.

        Returns:
            List of annotation dicts for the requested sample.

        Raises:
            IndexError: If ``idx`` is out of range.
        """
        if not self._loaded:
            self.load()
        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")
        return self.annotations_by_image[self.images[idx]]


class SpiritualistDataset:
    """Loader for the Spiritualist dataset with SSU-tagged Transkribus ALTO XML ground truth.

    Assumes SSU attributes have already been assigned to TextLine elements
    (e.g. via ALTOSSUTagger). Each TextLine becomes one annotation.
    """

    def __init__(
        self,
        images_path: Path,
        groundtruth_path: Path,
        image_ext: str = "jpg",
    ):
        """
        Args:
            images_path: Directory containing JPEG image files.
            groundtruth_path: Directory containing SSU-tagged ALTO XML files.
            image_ext: Image file extension to glob for (default: 'jpg').
        """
        self.images_path = Path(images_path)
        self.groundtruth_path = Path(groundtruth_path)
        self.image_ext = image_ext
        self.images = []
        self.annotations_by_image = {}
        self._loaded = False

    def load(self):
        """Load the dataset from disk."""
        if self._loaded:
            return

        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.groundtruth_path.exists():
            raise FileNotFoundError(f"Ground truth directory not found: {self.groundtruth_path}")

        xml_by_stem = {xml_path.stem: xml_path for xml_path in self.groundtruth_path.glob("*.xml")}

        for img_path in sorted(self.images_path.glob(f"*.{self.image_ext}")):
            xml_path = xml_by_stem.get(img_path.stem)
            if xml_path is None:
                logger.warning(f"No ground truth XML found for {img_path.name}, skipping")
                continue

            annotations, orig_w, orig_h = self._parse_xml(xml_path, img_path.stem)

            if orig_w > 0:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
                if img_w != orig_w:
                    scale = img_w / orig_w
                    for ann in annotations:
                        ann["x"] *= scale
                        ann["y"] *= scale
                        ann["width"] *= scale
                        ann["height"] *= scale

            self.images.append(str(img_path))
            self.annotations_by_image[str(img_path)] = annotations

        self._loaded = True

    def _parse_xml(self, xml_path: Path, page_id: str) -> tuple:
        """Parse an SSU-tagged ALTO XML file and return (annotations, orig_width, orig_height)."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Detect namespace from root tag
        tag = root.tag
        ns_uri = tag[1: tag.index("}")] if tag.startswith("{") else ""
        ns = {"alto": ns_uri} if ns_uri else {}

        def q(name):
            return f"alto:{name}" if ns else name

        layout = root.find(q("Layout"), ns)
        page = layout.find(q("Page"), ns) if layout is not None else None
        orig_w = int(page.get("WIDTH", 0)) if page is not None else 0
        orig_h = int(page.get("HEIGHT", 0)) if page is not None else 0

        print_space = page.find(q("PrintSpace"), ns) if page is not None else None
        if print_space is None:
            return [], orig_w, orig_h

        # Collect unique SSU strings in document order for stable integer mapping
        ssu_strings: List[str] = []
        for block in print_space.findall(q("TextBlock"), ns):
            for line in block.findall(q("TextLine"), ns):
                ssu = line.get("SSU")
                if ssu and ssu not in ssu_strings:
                    ssu_strings.append(ssu)
        ssu_to_int = {s: i + 1 for i, s in enumerate(ssu_strings)}

        annotations = []
        for block in print_space.findall(q("TextBlock"), ns):
            for line in block.findall(q("TextLine"), ns):
                ssu = line.get("SSU")
                if not ssu:
                    continue
                try:
                    x = int(line.get("HPOS", 0))
                    y = int(line.get("VPOS", 0))
                    w = int(line.get("WIDTH", 0))
                    h = int(line.get("HEIGHT", 0))
                except (ValueError, TypeError):
                    logger.warning(f"Could not parse TextLine bbox in {xml_path.name}, skipping")
                    continue

                annotations.append({
                    "x": float(x),
                    "y": float(y),
                    "width": float(w),
                    "height": float(h),
                    "class": "text",
                    "ssu_id": ssu_to_int[ssu],
                    "ssu_class": "object",
                    "confidence": 1.0,
                    "page_id": page_id,
                })

        return annotations, orig_w, orig_h

    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if not self._loaded:
            self.load()
        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")
        image_path = self.images[idx]
        return {
            "image_path": image_path,
            "annotations": self.annotations_by_image[image_path],
            "filename": Path(image_path).name,
        }


class HierTextDataset:
    """Loader for the HierText dataset (word/line/paragraph hierarchy).

    Each text *line* becomes one bounding-box region; lines belonging to the
    same paragraph share an ``ssu_id``. Polygons are reduced to axis-aligned
    bounding boxes. All lines are kept regardless of the ``legible`` flag.

    Reference: https://github.com/google-research-datasets/hiertext
    """

    def __init__(
        self,
        images_path: Path,
        groundtruth_path: Path,
        image_ext: str = "jpg",
    ):
        """
        Args:
            images_path: Directory containing ``<image_id>.<image_ext>`` files.
            groundtruth_path: Path to the HierText GT JSON file (e.g.
                ``validation.jsonl`` — a single JSON object despite the name).
            image_ext: Image file extension to glob for (default: 'jpg').
        """
        self.images_path = Path(images_path)
        self.groundtruth_path = Path(groundtruth_path)
        self.image_ext = image_ext
        self.images = []
        self.annotations_by_image = {}
        self._loaded = False

    def load(self):
        """Load the dataset from disk."""
        if self._loaded:
            return

        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.groundtruth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.groundtruth_path}")

        with open(self.groundtruth_path) as f:
            gt = json.load(f)

        entries_by_id = {entry["image_id"]: entry for entry in gt.get("annotations", [])}

        available = {p.stem: p for p in self.images_path.glob(f"*.{self.image_ext}")}

        for image_id, entry in entries_by_id.items():
            img_path = available.get(image_id)
            if img_path is None:
                logger.warning(f"No image found for GT entry {image_id}, skipping")
                continue

            annotations = self._build_annotations(entry, image_id)

            self.images.append(str(img_path))
            self.annotations_by_image[str(img_path)] = annotations

        self._loaded = True

    @staticmethod
    def _build_annotations(entry: dict, image_id: str) -> List[Dict[str, Any]]:
        """Map a GT entry's paragraphs/lines to line-level region annotations."""
        annotations: List[Dict[str, Any]] = []
        for para_idx, paragraph in enumerate(entry.get("paragraphs", []), start=1):
            for line in paragraph.get("lines", []):
                vertices = line.get("vertices", [])
                if len(vertices) < 3:
                    continue
                xs = [pt[0] for pt in vertices]
                ys = [pt[1] for pt in vertices]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                if x2 <= x1 or y2 <= y1:
                    continue
                annotations.append(
                    {
                        "x": float(x1),
                        "y": float(y1),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "class": "text",
                        "ssu_id": para_idx,
                        "ssu_class": "object",
                        "confidence": 1.0,
                        "page_id": image_id,
                    }
                )
        return annotations

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        if not self._loaded:
            self.load()
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample as ``{image_path, annotations, filename}``."""
        if not self._loaded:
            self.load()
        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")
        image_path = self.images[idx]
        return {
            "image_path": image_path,
            "annotations": self.annotations_by_image[image_path],
            "filename": Path(image_path).name,
        }

    def get_annotations(self, idx: int) -> List[Dict[str, Any]]:
        """Get ground-truth annotations for a sample."""
        if not self._loaded:
            self.load()
        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")
        return self.annotations_by_image[self.images[idx]]


class DocBankDataset:
    """Loader for a local DocBank subset (SpACER's data/docbank pipeline).

    Ground truth comes from a flat CSV of SSU-level regions (one row per
    region, already the finished output of that repo's
    ``scripts/extract_docbank_ssu_text.py`` — words grouped into runs of
    consecutive identical labels), rather than DocBank's own per-word or
    MSCOCO annotation formats. Column names deliberately match
    :class:`HierTextDataset`'s annotation dict so both plug into
    :mod:`scripts.export_predictions` identically.

    Reference: https://github.com/doc-analysis/DocBank
    """

    def __init__(
        self,
        images_path: Path,
        groundtruth_path: Path,
        image_ext: str = "jpg",
    ):
        """
        Args:
            images_path: Directory containing ``<page_id>_ori.<image_ext>`` files.
            groundtruth_path: Path to a gt_ssu_bboxes.csv-style CSV with columns
                filename, page_id, image_width, image_height, x, y, width,
                height, ssu_id, label, gt_text.
            image_ext: Image file extension to glob for (default: 'jpg').
        """
        self.images_path = Path(images_path)
        self.groundtruth_path = Path(groundtruth_path)
        self.image_ext = image_ext
        self.images: List[str] = []
        self.annotations_by_image: Dict[str, List[Dict[str, Any]]] = {}
        self._loaded = False

    def load(self):
        """Load the dataset from disk."""
        if self._loaded:
            return

        if not self.images_path.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_path}")
        if not self.groundtruth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.groundtruth_path}")

        gt_df = pd.read_csv(self.groundtruth_path)
        available = {p.stem: p for p in self.images_path.glob(f"*.{self.image_ext}")}

        for filename, page_df in gt_df.groupby("filename"):
            img_path = available.get(Path(filename).stem)
            if img_path is None:
                logger.warning(f"No image found for GT entry {filename}, skipping")
                continue

            page_id = page_df["page_id"].iloc[0]
            annotations = [
                {
                    "x": float(row.x),
                    "y": float(row.y),
                    "width": float(row.width),
                    "height": float(row.height),
                    "class": row.label,
                    "ssu_id": int(row.ssu_id),
                    "ssu_class": "object",
                    "confidence": 1.0,
                    "page_id": page_id,
                }
                for row in page_df.itertuples(index=False)
            ]

            self.images.append(str(img_path))
            self.annotations_by_image[str(img_path)] = annotations

        self._loaded = True

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        if not self._loaded:
            self.load()
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample as ``{image_path, annotations, filename}``."""
        if not self._loaded:
            self.load()
        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")
        image_path = self.images[idx]
        return {
            "image_path": image_path,
            "annotations": self.annotations_by_image[image_path],
            "filename": Path(image_path).name,
        }

    def get_annotations(self, idx: int) -> List[Dict[str, Any]]:
        """Get ground-truth annotations for a sample."""
        if not self._loaded:
            self.load()
        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.images)}")
        return self.annotations_by_image[self.images[idx]]


def _boxes_from_chars(chars: pd.DataFrame, group_key: str, carry: Sequence[str] = ()) -> list:
    """Group a character table and return the extent box of each group.

    Args:
        chars: Character table as returned by :func:`load_limerick_example`.
        group_key: Column to group on, e.g. ``"ssu_id"`` or ``"line_id"``.
        carry: Additional columns copied onto each box from the group's first row.

    Returns:
        List of XYWH dicts, ordered by ``group_key``, each carrying that key.
    """
    boxes: list = []
    for key, group in chars.groupby(group_key, sort=True):
        x_min = float(group["x"].min())
        y_min = float(group["y"].min())
        x_max = float((group["x"] + group["width"]).max())
        y_max = float((group["y"] + group["height"]).max())
        box = {
            "x": x_min,
            "y": y_min,
            "width": x_max - x_min,
            "height": y_max - y_min,
            group_key: int(key),
        }
        for col in carry:
            value = group[col].iloc[0]
            box[col] = value.item() if hasattr(value, "item") else value
        boxes.append(box)
    return boxes


def extract_ssu_boxes(chars: pd.DataFrame) -> list:
    """Extract SSU-level ground-truth boxes from the character table.

    Each SSU's box is the axis-aligned extent of its characters, so the box is
    ink-derived rather than a typeset rectangle padded with whitespace. The
    result is ready to pass straight to
    :func:`~cotescore.adapters.boxes_to_gt_ssu_map`.

    ``ssu_id`` is 1-based in the character table because cotescore reserves 0
    for background; it is carried through unchanged.

    Args:
        chars: Character table as returned by :func:`load_limerick_example`.

    Returns:
        List of dicts with keys ``x``, ``y``, ``width``, ``height``, ``ssu_id``,
        ``ssu_class`` and ``semantic_unit``.
    """
    return _boxes_from_chars(chars, "ssu_id", carry=("ssu_class", "semantic_unit"))


def extract_line_boxes(chars: pd.DataFrame) -> list:
    """Extract line-level boxes from the character table.

    Useful as a finer-grained alternative ground truth when demonstrating how
    detection metrics behave under a granularity mismatch.

    Args:
        chars: Character table as returned by :func:`load_limerick_example`.

    Returns:
        List of dicts with keys ``x``, ``y``, ``width``, ``height``, ``line_id``,
        ``ssu_id`` and ``semantic_unit``.
    """
    return _boxes_from_chars(chars, "line_id", carry=("ssu_id", "semantic_unit"))


def extract_word_boxes(chars: pd.DataFrame) -> list:
    """Extract word-level boxes from the character table.

    Args:
        chars: Character table as returned by :func:`load_limerick_example`.

    Returns:
        List of dicts with keys ``x``, ``y``, ``width``, ``height``, ``word_id``,
        ``line_id`` and ``ssu_id``.
    """
    return _boxes_from_chars(chars, "word_id", carry=("line_id", "ssu_id"))


def chars_to_region_chars(chars: pd.DataFrame) -> "RegionChars":
    """Convert the character table to :class:`~cotescore.types.RegionChars`.

    This is the input the spatial OCR decompositions want — see
    :func:`~cotescore.ocr.cdd_decomp_spatial`. Character midpoints are derived
    from the stored extents.

    Args:
        chars: Character table as returned by :func:`load_limerick_example`.

    Returns:
        A ``RegionChars`` whose ``region_ids`` are the table's ``ssu_id`` values.
    """
    from cotescore.types import RegionChars

    return RegionChars(
        tokens=chars["char"].to_numpy(dtype=object),
        xs=np.rint(chars["x"] + chars["width"] / 2.0).astype(int).to_numpy(),
        ys=np.rint(chars["y"] + chars["height"] / 2.0).astype(int).to_numpy(),
        region_ids=chars["ssu_id"].astype(int).to_numpy(),
    )


def reconstruct_text(chars: pd.DataFrame) -> str:
    """Rebuild the page text from the character table.

    Characters are ordered by ``(line_id, x)`` and grouped into words by
    ``word_id``. Space characters are not stored — word boundaries carry that
    information instead.

    Args:
        chars: Character table as returned by :func:`load_limerick_example`.

    Returns:
        The page text, one line per ``line_id``, in reading order.
    """
    lines = []
    ordered = chars.sort_values(["line_id", "x"])
    for _, line in ordered.groupby("line_id", sort=True):
        words = [
            "".join(word.sort_values("x")["char"])
            for _, word in line.groupby("word_id", sort=False)
        ]
        lines.append(" ".join(words))
    return "\n".join(lines)


def load_limerick_example() -> Tuple[pd.DataFrame, np.ndarray, list]:
    """Load the bundled limerick case study example.

    The ground truth is a character table — one row per non-space character,
    with its pixel extent and its word, line and SSU groupings. Everything the
    analysis needs is derived from it: region boxes at any granularity
    (:func:`extract_ssu_boxes`, :func:`extract_line_boxes`,
    :func:`extract_word_boxes`), character positions for the spatial OCR
    decompositions (:func:`chars_to_region_chars`), and the page text
    (:func:`reconstruct_text`).

    Columns: ``char``, ``x``, ``y``, ``width``, ``height``, ``word_id``,
    ``line_id``, ``ssu_id``, ``ssu_class``, ``semantic_unit``.

    ``semantic_unit`` identifies the work an SSU belongs to. A title and its body
    share one, and so do the two halves of a poem that continues across a column
    boundary — which is what makes the page a meaningful COTe example.

    Returns:
        Tuple of (character table, image as a numpy array, prediction boxes).
        The predictions are XYWH dicts standing in for a layout detector.
    """
    pkg = files("cotescore") / "assets" / "limerick_case_study"
    with as_file(pkg / "characters.parquet") as parquet_path:
        chars = pd.read_parquet(parquet_path)
    image = np.array(Image.open(pkg / "limerick_image.png"))
    predictions = json.loads((pkg / "predictions.json").read_text())
    return chars, image, predictions
