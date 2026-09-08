<p align="center">
  <img src="https://raw.githubusercontent.com/JonnoB/cotescore/main/docs/cotescore_banner.png"
       alt="cotescore" width="700">
</p>


**Decomposable evaluation metrics for document understanding pipelines — the COTe score for layout parsing, and the Character Error Vector for page-level OCR**

[![PyPI version](https://img.shields.io/pypi/v/cotescore)](https://pypi.org/project/cotescore/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)


## Overview

Document Layout Analysis (DLA) is the process of parsing a page into meaningful elements, typically using machine learning models. Traditional evaluation metrics such as IoU, F1, and mAP were designed for 3D-to-2D image projections (e.g. photographs) and can give misleading results for natively 2D printed documents.

Document understanding pipelines generally have two stages. The first stage parses the page into individual visual elements or regions, effectively focusing on answering **where is the text?**. The second stage is the Optical Character Recognition, where the text is transcribed in to machine readable format, answering **what does the text say?**. When the output text quality is poor conventional metrics can struggle to identify the root cause. 

The **cotescore** library provides two decomposable metrics one for each stage of the pipeline.  

### Layout parsing — the COTe score


- **Structural Semantic Units (SSUs)** — a relational labelling approach that shifts focus from the physical bounding boxes of regions to the semantic structure of the content.
- **COTe Score** — a decomposable metric that breaks page-parsing quality into four interpretable components:
  - **C**overage — how well predictions cover ground-truth regions
  - **O**verlap — redundant predictions within the same ground-truth region
  - **T**respass — predictions that cross semantic boundaries into a different SSU
  - **e**xcess — A support metric for predictions that fall on background or white-space

COTe is more informative than traditional metrics, reveals distinct model failure modes, and remains useful even when explicit SSU labels are unavailable.

### Page-level OCR — the Character Error Vector

Character Error Rate assumes the text was parsed perfectly. When it was not, the reading order is wrong, the alignment CER depends on becomes meaningless, and the metric breaks down precisely when a diagnosis is most needed.

- **Character Error Vector (CEV)** — a bag-of-characters evaluator that needs no alignment, and decomposes page-level OCR error into **parsing**, **OCR** and **interaction** components.
- **Two instantiations** ship with the library: **SpACER** (Spatially Aware Character Error Rate), a count-based metric analogous to CER, and **CDD**, a character-distribution distance built on the Jensen–Shannon distance.

Because the components separate the stages, the CEV can say whether the transcription was let down by the parser or by the OCR engine. This library is the reference implementation for the CEV paper.

## Installation

```bash
pip install cotescore
```

That is all you need to compute COTe and CEV scores on predictions you already have. Running the layout models themselves requires extra dependencies — see [Optional extras](#optional-extras--running-the-benchmark-models) at the end.

## Quick Start

The library ships with a bundled limerick case study that you can use to try it out immediately.

```python
from cotescore import cote_score, load_limerick_example, extract_ssu_boxes
from cotescore.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks, compute_canvas

# Load the bundled example: ground-truth dict, document image, and example predictions
ground_truth, image, pred_boxes = load_limerick_example()

h, w = image.shape[:2]
canvas_w, canvas_h = compute_canvas(w, h)

# Build tagged SSU-level GT boxes and rasterize to a 2-D SSU id map
gt_boxes = extract_ssu_boxes(ground_truth)
gt_ssu_map = boxes_to_gt_ssu_map(gt_boxes, w, h, canvas_w, canvas_h)
preds = boxes_to_pred_masks(pred_boxes, w, h, canvas_w, canvas_h)

# Compute the COTe score
cote, C, O, T, E = cote_score(gt_ssu_map, preds)
print(f"COTe={cote:.3f}  C={C:.3f}  O={O:.3f}  T={T:.3f}  E={E:.3f}")
```
---

![COTe pixel-state visualisation showing Coverage, Overlap, Trespass and Excess regions](docs/example_cote_components.png)

*COTe pixel-state visualisation: each pixel in the document is classified as Coverage (green), Overlap (yellow), Trespass (red), Trespass AND Overlap (purple), or Excess (blue). Produced by `notebooks/limerick_analysis.py`.*

---

See [`notebooks/limerick_analysis.py`](notebooks/limerick_analysis.py) for a full worked example comparing COTe against F1 and mean IoU at different granularity levels.

## Core Metrics — COTe

| Function | Description |
|---|---|
| `cote_score(gt_ssu_map, preds)` | Returns `(cote, C, O, T, E)` — the full decomposition |
| `coverage(gt_ssu_map, preds)` | Fraction of GT area correctly covered `[0, 1]` |
| `overlap(gt_ssu_map, preds)` | Redundant prediction area within GT `[0, ∞)` |
| `trespass(gt_ssu_map, preds)` | GT area covered by wrong-SSU predictions `[0, ∞]` |
| `excess(gt_ssu_map, preds)` | Predicted area falling on background, over total background `[0, 1]` |
| `cote_class(gt_ssu_map, ssu_to_class, preds)` | Per-class interaction matrices (`ClassCOTeResult`) |

All functions are importable directly from `cotescore`:

```python
from cotescore import cote_score, coverage, overlap, trespass, excess, cote_class
```

The conventional instance-matching baselines COTe is compared against in the paper — `iou`, `mean_iou` and `f1` — remain available from `cotescore.layout` if you want to reproduce those comparisons.

For an alternative visual overview of the what the different elements of the COTe score mean please see [`notebooks/metrics_exploration.py`](notebooks/metrics_exploration.py)

## Core Metrics — the CEV

The CEV compares four bags of characters. Which ones you can build determines which components you get:

| Symbol | Key | What it is |
|---|---|---|
| $Q$ | `gt` | all ground-truth characters on the page |
| $R$ | `parsing` | ground-truth characters falling inside **predicted** regions |
| $S^*$ | `ocr` | OCR run on **ground-truth** regions |
| $S$ | `total` | OCR run on **predicted** regions |

From those come the four components — `d_pars` (parsing alone), `d_ocr` (OCR given perfect regions), `d_int` (their interaction), and `d_total` (the whole pipeline).

| Function | Description |
|---|---|
| `spacer(reference, prediction)` | Macro SpACER between two token Counters `[0, ∞)` |
| `spacer_micro(ref_boxes, pred_boxes)` | Per-box SpACER, so deletions in one box are not masked by insertions in another |
| `spacer_decomp(named_dict)` | Four-way SpACER decomposition (macro and micro) from page or per-box text |
| `cdd_decomp(named_dict)` | Four-way CDD decomposition; any `(Counter, Counter) -> float` metric can be substituted |
| `spacer_decomp_spatial(gt_chars, pred_regions, ...)` | As above, building $R$ by joining character positions to predicted regions |
| `cdd_decomp_spatial(gt_chars, pred_regions, ...)` | Spatial CDD equivalent |
| `jsd_distance(p, q)` | Square root of the Jensen–Shannon divergence `[0, 1]` — the default CDD metric |
| `text_to_counter(text, mode)` | Build a character (or word, for SpAWER) frequency Counter |

```python
from cotescore import spacer_decomp, cdd_decomp

bags = {"gt": gt_text, "ocr": ocr_of_gt_regions, "total": ocr_of_predicted_regions}
sp = spacer_decomp(bags)
cd = cdd_decomp(bags)
print(sp.d_ocr_macro, sp.d_total_macro)
```

Components whose inputs are absent come back as `None` rather than failing. Building $R$ needs character-level positions, so a dataset with only region-level text yields `d_ocr` and `d_total` while `d_pars` and `d_int` are `None`. That pair is still enough to triage: combining COTe with the ratio `d_ocr / d_total` predicts the dominant error source with an F1 of 0.91, using only values that are cheap to obtain. The `*_spatial` functions recover the full vector when character boxes are available.

## Visualisation

```python
from cotescore import compute_cote_masks, visualize_cote_states
import matplotlib.pyplot as plt

masks = compute_cote_masks(gt_ssu_map, preds)

fig, ax = plt.subplots()
visualize_cote_states(image, masks, ax=ax)
plt.show()
```

## Datasets

The library includes loaders for the datasets used in the two papers, all sharing one annotation format:

```python
from cotescore.dataset import (
    NCSEDataset, DocLayNetDataset, HNLA2013Dataset,
    SpiritualistDataset, HierTextDataset, DocBankDataset,
)

# Bundled toy example (no download required)
from cotescore import load_limerick_example
chars_df, image, pred_boxes = load_limerick_example()
```

## Examples

The primary interactive example is the Marimo notebook at [`notebooks/limerick_analysis.py`](notebooks/limerick_analysis.py). It demonstrates:

- Visualising SSU, line, and character-level bounding boxes
- The granularity-mismatch problem and how COTe handles it
- Side-by-side comparison of COTe, F1, and mean IoU
- COTe pixel-state visualisation

To run the notebook (requires [Marimo](https://marimo.io)):

```bash
pip install marimo
marimo edit notebooks/limerick_analysis.py
```

### End-to-end walkthrough on a real dataset

Two notebooks evaluate a complete two-stage document pipeline — layout parsing, then OCR — on
the NCSE newspaper test set. Run them in order; the first writes the predictions the second
reads.

1. [`notebooks/Tutorial_1_the_COTe.py`](notebooks/Tutorial_1_the_COTe.py) — scores two layout
   models (DocLayout-YOLO and Docling Heron) with COTe, compares it against F1 and mean IoU,
   and writes a folder of COTe pixel-state diagnostics per model.
2. [`notebooks/Tutorial_2_the_CEV.py`](notebooks/Tutorial_2_the_CEV.py) — runs EasyOCR and
   Tesseract over both models' regions and scores them with the CEV (SpACER and CDD), then uses
   COTe together with `d_ocr / d_total` to attribute each page's error to parsing or to OCR.

Both notebooks take the images folder and the ground-truth CSV as plain string paths at the top,
so they can be pointed at your own dataset. The layout models need a GPU and the
`doclayout-yolo`, `easyocr` and `pytesseract` packages (plus the `tesseract` binary).

```bash
marimo edit notebooks/Tutorial_1_the_COTe.py
marimo edit notebooks/Tutorial_2_the_CEV.py
```

## Optional extras — running the benchmark models

Only needed if you want to run the document layout models yourself, for example to reproduce the papers' benchmarks. Scoring existing predictions needs none of this.

The benchmarking extras include either PyTorch or PaddlePaddle — **do not install both in the same environment**. These frameworks require different CUDA versions and will conflict:

| Extra | Use case | GPU framework |
|---|---|---|
| `cotescore[benchmarks]` | Torch-based DLA models (DocLayout-YOLO, Heron) | PyTorch |
| `cotescore[paddle-benchmark]` | PaddleOCR / PP-DocLayout | PaddlePaddle |

```bash
# Torch-based benchmarks
pip install "cotescore[benchmarks]"

# PaddlePaddle benchmarks (install PaddlePaddle first, separately)
# See: https://www.paddlepaddle.org.cn/install/quick
pip install "cotescore[paddle-benchmark]"
```

> **Note:** PaddlePaddle must be installed before `cotescore[paddle-benchmark]` — it is not listed as a pip dependency because the correct wheel depends on your CUDA version. Follow the [PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/install/quick) to get the right version.

## Questions and Bug Reports

If you have questions, find a bug, or want to request a feature, please [open an issue](https://github.com/JonnoB/cot_analysis/issues) on GitHub.

## Citation

If you use cotescore in your research, please cite the paper for the metric you used.

**COTe** — layout parsing:

Bourne, Jonathan, Mwiza Simbeye, and Ishtar Govia. “The COTe Score: A Decomposable Framework for Evaluating Document Layout Analysis Models.” arXiv:2603.12718. Preprint, arXiv, March 13, 2026. https://doi.org/10.48550/arXiv.2603.12718.


```bibtex
@misc{bourne2026cote,
  title         = {The {COTe} Score: A Decomposable Framework for Evaluating {Document Layout Analysis} Models},
  author        = {Bourne, Jonathan and Simbeye, Mwiza and Govia, Ishtar},
  year          = {2026},
  month         = mar,
  publisher     = {arXiv},
  doi           = {10.48550/arXiv.2603.12718},
}
```

**CEV / SpACER** — page-level OCR:

Bourne, Jonathan, Mwiza Simbeye, and Joseph Nockels. "The Character Error Vector: Decomposable Errors for Page-Level OCR Evaluation." arXiv:2604.06160. Preprint, arXiv, April 2026. https://doi.org/10.48550/arXiv.2604.06160.

```bibtex
@misc{bourne2026charactererrorvector,
  title         = {The {Character Error Vector}: Decomposable errors for page-level {OCR} evaluation},
  author        = {Bourne, Jonathan and Simbeye, Mwiza and Nockels, Joseph},
  year          = {2026},
  month         = apr,
  eprint        = {2604.06160},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2604.06160},
}
```
