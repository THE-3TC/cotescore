"""
COTe-Score: Coverage, Overlap, Trespass, Excess

A library for evaluating document layout analysis models using Coverage, Overlap,
Trespass, and Excess metrics.
"""

__version__ = "0.1.0"

from .layout import cote_score, coverage, overlap, trespass, excess, iou, mean_iou
from .ocr import (
    shannon_entropy,
    jensen_shannon_divergence,
    text_to_counter,
    jsd_distance,
    spacer,
    spacer_micro,
    cdd_decomp,
    spacer_decomp,
    cdd_decomp_spatial,
    spacer_decomp_spatial,
)
from .class_metrics import (
    cote_class,
    coverage_matrix,
    overlap_matrix,
    trespass_matrix,
    class_confusion_counts,
    sum_class_counts,
    finalize_class_counts,
)
from .types import ClassCOTeResult, ClassCOTeCounts, TokenPositions, RegionChars, RegionPixels, GTBoxes, CDDDecomposition, SpACERDecomposition
from .adapters import boxes_to_region_pixels, polygons_to_panoptic_mask
from .panoptic_quality import panoptic_quality
from .visualisation import (
    MIXED_KEY,
    class_palette,
    compute_class_masks,
    compute_cote_masks,
    visualize_class_masks,
    visualize_cote_states,
)
from .dataset import (
    load_limerick_example,
    extract_ssu_boxes,
    extract_line_boxes,
    extract_word_boxes,
    chars_to_region_chars,
    reconstruct_text,
)
from .alto_ssu_tagger import ALTOSSUTagger, assign_alto_ssu

__all__ = [
    "cote_score",
    "coverage",
    "overlap",
    "trespass",
    "excess",
    "iou",
    "mean_iou",
    "shannon_entropy",
    "jensen_shannon_divergence",
    "text_to_counter",
    "jsd_distance",
    "spacer",
    "spacer_micro",
    "cdd_decomp",
    "spacer_decomp",
    "cdd_decomp_spatial",
    "spacer_decomp_spatial",
    "cote_class",
    "coverage_matrix",
    "overlap_matrix",
    "trespass_matrix",
    "class_confusion_counts",
    "sum_class_counts",
    "finalize_class_counts",
    "ClassCOTeResult",
    "ClassCOTeCounts",
    "TokenPositions",
    "RegionChars",
    "RegionPixels",
    "GTBoxes",
    "CDDDecomposition",
    "SpACERDecomposition",
    "boxes_to_region_pixels",
    "polygons_to_panoptic_mask",
    "panoptic_quality",
    "compute_cote_masks",
    "visualize_cote_states",
    "compute_class_masks",
    "visualize_class_masks",
    "class_palette",
    "MIXED_KEY",
    "load_limerick_example",
    "extract_ssu_boxes",
    "extract_line_boxes",
    "extract_word_boxes",
    "chars_to_region_chars",
    "reconstruct_text",
    "ALTOSSUTagger",
    "assign_alto_ssu",
]
