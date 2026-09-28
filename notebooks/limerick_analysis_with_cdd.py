import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    """Imports for analysis (no rendering dependencies)."""
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import json
    import pandas as pd

    from plotnine import aes, geom_hline, geom_line, ggplot, labs, theme, theme_minimal, geom_vline

    # Import cote_score metrics for evaluation
    from cotescore.metrics import mean_iou, iou, overlap, cote_score, cdd
    from cotescore.visualisation import compute_cote_masks, visualize_cote_states
    from cotescore.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks

    figure_path = Path("data/figures")
    return (
        Path,
        aes,
        boxes_to_gt_ssu_map,
        boxes_to_pred_masks,
        cdd,
        compute_cote_masks,
        cote_score,
        figure_path,
        geom_hline,
        geom_line,
        geom_vline,
        ggplot,
        iou,
        json,
        labs,
        mean_iou,
        np,
        pd,
        plt,
        theme,
        theme_minimal,
        visualize_cote_states,
    )


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(Path, json, np):
    """Load the pre-generated image and ground truth from disk."""
    from PIL import Image as _Image

    _data_dir = Path("data/limerick_case_study")

    # Load image
    _img = _Image.open(_data_dir / "limerick_image.png")
    image_array = np.array(_img)

    # Load ground truth (simplified format: page + stories with lines)
    with open(_data_dir / "ground_truth.json", "r") as _f:
        ground_truth = json.load(_f)

    print(f"Loaded image shape: {image_array.shape}")
    print(f"Loaded GT with {len(ground_truth['stories'])} stories")
    return ground_truth, image_array


@app.cell
def _(iou):
    """Helper functions for granularity comparison metrics."""

    def extract_line_boxes(gt):
        """Extract all line-level boxes from simplified ground truth."""
        line_boxes = []
        for story in gt["stories"].values():
            for line in story["lines"]:
                bbox = line["bbox"]
                line_boxes.append({"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]})
        return line_boxes

    def extract_ssu_boxes(gt):
        """Extract SSU-level boxes (union of lines per SSU within each story)."""
        ssu_boxes = []
        for story in gt["stories"].values():
            groups = {}
            for line in story["lines"]:
                ssu = line["ssu"]
                if ssu not in groups:
                    groups[ssu] = []
                groups[ssu].append(line["bbox"])

            for ssu in sorted(groups):
                bboxes = groups[ssu]
                x_min = min(b[0] for b in bboxes)
                y_min = min(b[1] for b in bboxes)
                x_max = max(b[0] + b[2] for b in bboxes)
                y_max = max(b[1] + b[3] for b in bboxes)
                ssu_boxes.append(
                    {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min}
                )
        return ssu_boxes

    def calculate_detection_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
        """Calculate TP/FP/FN using IoU-based greedy matching."""
        if not pred_boxes or not gt_boxes:
            return {
                "tp": 0,
                "fp": len(pred_boxes),
                "fn": len(gt_boxes),
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        n_pred, n_gt = len(pred_boxes), len(gt_boxes)

        # Compute IoU matrix and find pairs above threshold
        pairs = []
        for i in range(n_pred):
            for j in range(n_gt):
                iou_val = iou(pred_boxes[i], gt_boxes[j])
                if iou_val >= iou_threshold:
                    pairs.append((iou_val, i, j))

        # Greedy matching: highest IoU first
        pairs.sort(reverse=True, key=lambda x: x[0])
        matched_pred, matched_gt = set(), set()
        for _, pred_idx, gt_idx in pairs:
            if pred_idx not in matched_pred and gt_idx not in matched_gt:
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)

        tp = len(matched_pred)
        fp = n_pred - tp
        fn = n_gt - len(matched_gt)
        precision = tp / n_pred if n_pred > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    return calculate_detection_metrics, extract_line_boxes, extract_ssu_boxes


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Ground Truth Visualization

    The following cells show the bounding boxes at different granularity levels:
    1. **SSU level** - Paragraphs colored by Semantic-Structural Unit (SSU)
    2. **Line level** - Each line of text has its own bbox (derived from characters)
    3. **Character level** - Each character has its own bbox

    The SSU coloring shows how the split limerick (Broken Computer) has:
    - SSU 0: Title
    - SSU 1: Body text in column 1
    - SSU 2: Continuation in column 2
    """
    )
    return


@app.cell
def _(figure_path, ground_truth, image_array, np, plt):
    """Visualize SSU-level bounding boxes colored by SSU id."""
    import matplotlib.patches as _patches
    import matplotlib.cm as _cm

    # Collect all (ssu, bbox) pairs across all stories
    _ssu_items = []
    for _story in ground_truth["stories"].values():
        _groups = {}
        for _line in _story["lines"]:
            _ssu = _line["ssu"]
            _groups.setdefault(_ssu, []).append(_line["bbox"])
        for _ssu in sorted(_groups):
            _bboxes = _groups[_ssu]
            _x = min(b[0] for b in _bboxes)
            _y = min(b[1] for b in _bboxes)
            _x2 = max(b[0] + b[2] for b in _bboxes)
            _y2 = max(b[1] + b[3] for b in _bboxes)
            _ssu_items.append((_ssu, _x, _y, _x2 - _x, _y2 - _y))

    _all_ssus = sorted(set(s for s, *_ in _ssu_items))
    _colors = ["red", "blue", "green"]
    _ssu_to_color = {ssu: _colors[i % len(_colors)] for i, ssu in enumerate(_all_ssus)}

    _fig_ssu, _ax_ssu = plt.subplots(figsize=(12, 6.2), dpi=300)
    _ax_ssu.imshow(image_array, cmap="gray", vmin=0, vmax=255)
    _ax_ssu.axis("off")

    _legend_patches = []
    for _ssu, _x, _y, _w, _h in _ssu_items:
        _color = _ssu_to_color[_ssu]
        _ax_ssu.add_patch(
            _patches.Rectangle(
                (_x, _y),
                _w,
                _h,
                linewidth=2,
                edgecolor=_color,
                facecolor="none",
                alpha=0.8,
                linestyle="--",
            )
        )

    for _ssu in _all_ssus:
        _legend_patches.append(_patches.Patch(color=_ssu_to_color[_ssu], label=f"SSU {_ssu + 1}"))

    _fig_ssu.legend(
        handles=_legend_patches,
        loc="lower center",
        ncol=len(_legend_patches),
        fontsize=15,
        framealpha=0.9,
    )
    plt.title("Structural Semantic Units", fontsize=25)
    plt.tight_layout()
    plt.savefig(figure_path / "example_ssu.png", bbox_inches="tight", pad_inches=0)
    plt.show()
    return


@app.cell
def _(ground_truth, image_array, np, plt):
    """Visualize line-level bounding boxes (custom implementation)."""
    import matplotlib.patches as _patches
    import matplotlib.cm as _cm

    _fig_line, _ax_line = plt.subplots(figsize=(12, 10), dpi=100)
    _ax_line.imshow(image_array, cmap="gray", vmin=0, vmax=255)
    _ax_line.set_title("Line-Level Bounding Boxes (colored by story)")
    _ax_line.axis("off")

    # Create color map for different stories
    _story_colors = _cm.Set2(np.linspace(0, 1, max(len(ground_truth["stories"]), 1)))

    _legend_patches = []
    for _story_idx, (_story_id, _story) in enumerate(ground_truth["stories"].items()):
        _color = _story_colors[_story_idx % len(_story_colors)]
        _legend_patches.append(_patches.Patch(color=_color, label=_story_id))

        for _i, _line in enumerate(_story["lines"]):
            _bbox = _line["bbox"]
            _x, _y, _width, _height = _bbox

            _rect = _patches.Rectangle(
                (_x, _y),
                _width,
                _height,
                linewidth=1.5,
                edgecolor=_color,
                facecolor="none",
                alpha=0.8,
            )
            _ax_line.add_patch(_rect)

            _ax_line.text(
                _x - 10,
                _y + _height / 2,
                f"L{_i}",
                fontsize=6,
                color=_color,
                verticalalignment="center",
                horizontalalignment="right",
            )

    _ax_line.legend(handles=_legend_patches, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6)
    plt.tight_layout()
    return


@app.cell
def _(ground_truth, image_array, np, plt):
    """Visualize character-level bounding boxes colored by line index."""
    import matplotlib.patches as _patches
    import matplotlib.cm as _cm

    # Collect all characters with their global line index
    _char_items = []  # (line_idx, x, y, w, h)
    _line_idx = 0
    for _story in ground_truth["stories"].values():
        for _line in _story["lines"]:
            for _char in _line["characters"]:
                _b = _char["bbox"]
                _char_items.append((_line_idx, _b[0], _b[1], _b[2], _b[3]))
            _line_idx += 1

    _n_lines = _line_idx
    _colors = _cm.tab20(np.linspace(0, 1, max(_n_lines, 1)))

    _fig_char, _ax_char = plt.subplots(figsize=(12, 10), dpi=100)
    _ax_char.imshow(image_array, cmap="gray", vmin=0, vmax=255)
    _ax_char.axis("off")

    for _li, _x, _y, _w, _h in _char_items:
        _color = _colors[_li % len(_colors)]
        _ax_char.add_patch(
            _patches.Rectangle(
                (_x, _y),
                _w,
                _h,
                linewidth=0.5,
                edgecolor=_color,
                facecolor="none",
                alpha=0.7,
            )
        )

    plt.title("Character-Level Bounding Boxes (colored by line)")
    plt.tight_layout()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Granularity Mismatch in Bounding Box Evaluation

    Traditional detection metrics (Precision/Recall/F1) assume ground truth and predictions
    have the same granularity. When there's a mismatch:

    - **GT: Line, Pred: Para** - If predictions are at paragraph level but GT is at line level,
      each paragraph box can only match ONE line, causing many false negatives.

    - **GT: Para, Pred: Line** - If predictions are at line level but GT is at paragraph level,
      each line can only match ONE paragraph, causing many false positives.

    The **SSU-based Coverage** metric handles this correctly by measuring what fraction of
    the ground truth area is covered by predictions, regardless of box count.
    """
    )
    return


@app.cell
def _(extract_ssu_boxes, ground_truth):
    pred_boxes = extract_ssu_boxes(ground_truth)
    pred_boxes[0]["x"] = 83.33333333333334
    pred_boxes[0]["y"] = 460
    pred_boxes[0]["width"] = 750
    pred_boxes[0]["height"] = 200

    pred_boxes[1]["width"] = 2000
    pred_boxes.pop(4)
    return (pred_boxes,)


@app.cell
def _(
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    calculate_detection_metrics,
    cote_score,
    extract_line_boxes,
    extract_ssu_boxes,
    ground_truth,
    gt_boxes,
    mean_iou,
    mo,
    pd,
):
    """Compute and display granularity comparison table."""

    # Extract line-level and SSU-level boxes
    line_boxes = extract_line_boxes(ground_truth)
    para_boxes = extract_ssu_boxes(ground_truth)

    # Get image dimensions
    img_w = int(ground_truth["page"]["width"])
    img_h = int(ground_truth["page"]["height"])

    # Convert GT boxes to SSU map (required by new cote_score API)
    _gt_tagged = [{**b, "ssu_id": i + 1} for i, b in enumerate(gt_boxes)]
    _gt_ssu_map = boxes_to_gt_ssu_map(_gt_tagged, img_w, img_h, img_w, img_h)

    # Scenario 1: GT=Line, Pred=Para (paragraph predictions vs line ground truth)
    m1 = calculate_detection_metrics(para_boxes, line_boxes)
    mean_iou_1 = mean_iou(para_boxes, line_boxes)
    coverage_1, _, _, _, _ = cote_score(_gt_ssu_map, boxes_to_pred_masks(para_boxes, img_w, img_h, img_w, img_h))

    # Scenario 2: GT=Para, Pred=Line (line predictions vs paragraph ground truth)
    m2 = calculate_detection_metrics(line_boxes, para_boxes)
    mean_iou_2 = mean_iou(line_boxes, para_boxes)
    coverage_2, _, _, _, _ = cote_score(_gt_ssu_map, boxes_to_pred_masks(line_boxes, img_w, img_h, img_w, img_h))

    # Build comparison DataFrame
    comparison_df = pd.DataFrame(
        {
            "Metric": [
                "True Positive",
                "False Positive",
                "False Negative",
                "Precision",
                "Recall",
                "F1",
                "Mean IoU",
                "COTe Score",
            ],
            "GT: Line, Pred: Para": [
                m1["tp"],
                m1["fp"],
                m1["fn"],
                f"{m1['precision']:.2f}",
                f"{m1['recall']:.2f}",
                f"{m1['f1']:.2f}",
                f"{mean_iou_1:.2f}",
                f"{coverage_1:.2f}",
            ],
            "GT: Para, Pred: Line": [
                m2["tp"],
                m2["fp"],
                m2["fn"],
                f"{m2['precision']:.2f}",
                f"{m2['recall']:.2f}",
                f"{m2['f1']:.2f}",
                f"{mean_iou_2:.2f}",
                f"{coverage_2:.2f}",
            ],
        }
    )

    import latex_tables as _lt

    _latex = _lt.latex_table(
        comparison_df.set_index("Metric"),
        caption="Detection metrics when line and paragraph ground truth are scored "
        "against each other.",
        label="tab:line_vs_para",
        echo=False,
    )
    mo.md(f"### LaTeX Table Output\n```latex\n{_latex}\n```")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## COTe Pixel State Visualization

    This section provides tools to visualize the pixel-level states used in COTe scoring:
    - **Coverage**: Pixels in GT covered by a correctly-assigned prediction
    - **Overlap**: Pixels in GT covered by multiple predictions (all correctly assigned)
    - **Trespass**: Pixels in GT covered by a prediction assigned to a different GT
    - **Overlap+Trespass**: Pixels with multiple predictions, at least one trespassing
    - **Excess**: Pixels outside GT but covered by predictions
    """
    )
    return


@app.cell
def _(
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    compute_cote_masks,
    extract_ssu_boxes,
    figure_path,
    ground_truth,
    image_array,
    plt,
    pred_boxes,
    visualize_cote_states,
):
    # Extract ground truth boxes at SSU level
    gt_boxes = extract_ssu_boxes(ground_truth)

    # Rasterize GT boxes to an SSU id map and pred boxes to binary masks
    _h, _w = image_array.shape[:2]
    _gt_tagged = [{**b, "ssu_id": i + 1} for i, b in enumerate(gt_boxes)]
    _gt_ssu_map = boxes_to_gt_ssu_map(_gt_tagged, _w, _h, _w, _h)
    _pred_masks = boxes_to_pred_masks(pred_boxes, _w, _h, _w, _h)

    # Compute COTe pixel-state masks
    cote_masks = compute_cote_masks(_gt_ssu_map, _pred_masks)

    # Visualize
    fig = visualize_cote_states(
        image_array,
        cote_masks,
        figsize=(12, 6.2),
        dpi=300,
    )
    fig.savefig(figure_path / "example_cote_components", bbox_inches="tight", pad_inches=0)
    plt.show()
    return (gt_boxes,)


@app.cell
def _(
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    cote_score,
    ground_truth,
    gt_boxes,
    iou,
    mean_iou,
    mo,
    np,
    pd,
    pred_boxes,
):

    img_wx = int(ground_truth["page"]["width"])
    img_hx = int(ground_truth["page"]["height"])

    mIoU = mean_iou(pred_boxes, gt_boxes)

    # Convert to SSU map and pred masks (required by new cote_score API)
    _gt_tagged = [{**b, "ssu_id": i + 1} for i, b in enumerate(gt_boxes)]
    _gt_ssu_map = boxes_to_gt_ssu_map(_gt_tagged, img_wx, img_hx, img_wx, img_hx)
    _pred_masks = boxes_to_pred_masks(pred_boxes, img_wx, img_hx, img_wx, img_hx)

    # Compute COTe score (C - O - T)
    cote, C, O, T, E = cote_score(_gt_ssu_map, _pred_masks)

    # Compute F1 score (IoU threshold = 0.5) using optimal Hungarian matching
    from scipy.optimize import linear_sum_assignment

    _iou_threshold = 0.5
    _iou_matrix = np.array([[iou(p, g) for g in gt_boxes] for p in pred_boxes])
    _row_ind, _col_ind = linear_sum_assignment(_iou_matrix, maximize=True)
    _tp = sum(1 for r, c in zip(_row_ind, _col_ind) if _iou_matrix[r, c] >= _iou_threshold)
    _precision = _tp / len(pred_boxes) if pred_boxes else 1.0
    _recall = _tp / len(gt_boxes) if gt_boxes else 1.0
    f1 = 2 * _precision * _recall / (_precision + _recall) if (_precision + _recall) > 0 else 0.0

    import latex_tables as _lt

    _example = pd.DataFrame(
        {
            "Perfect": ["1", "1", "0", "0", "0", "1", "1"],
            "Fig 2": [f"{_v:.3f}" for _v in (cote, C, O, T, E, mIoU, f1)],
        },
        index=pd.Index(
            ["COTe", "Coverage", "Overlap", "Trespass", "Excess", "Mean IoU",
             r"F1 (IoU$\geq$0.5)"],
            name="Metric",
        ),
    )
    cote_example_latex_table = _lt.latex_table(
        _example,
        caption="Performance metrics across different figures.",
        label="tab:cote_example",
        echo=False,
    )
    mo.md(f"### LaTeX Table Output\n```latex\n{cote_example_latex_table}\n```")
    return


@app.cell
def _(cdd, ground_truth, mo, pred_boxes):
    """Compute character-level CDD for erroneous parsing and LLM hallucination."""
    from jiwer import cer

    def _chars_in_regions(all_chars, regions):
        """Return characters whose centre falls within any of the given regions."""
        result = []
        for char in all_chars:
            cx = char["bbox"][0] + char["bbox"][2] / 2
            cy = char["bbox"][1] + char["bbox"][3] / 2
            for r in regions:
                if r["x"] <= cx <= r["x"] + r["width"] and r["y"] <= cy <= r["y"] + r["height"]:
                    result.append(char["char"])
                    break
        return result

    # Collect all characters with their bboxes
    all_chars = []
    for story in ground_truth["stories"].values():
        for _line in story["lines"]:
            for char in _line["characters"]:
                all_chars.append(char)

    gt_text_str = "".join(c["char"] for c in all_chars)
    detected_text_chars = _chars_in_regions(all_chars, pred_boxes)
    detected_text_str = "".join(detected_text_chars)

    # Erroneous parsing CDD (CER not meaningful — order is not comparable across box sets)
    err_cdd, _ = cdd([gt_text_str], [detected_text_str])

    # LLM hallucination: model gets stuck repeating a phrase
    hallucinated_text = (
        "There was a young man who read text, "
        "Whose mind was extremely vexed; " + "His boxes unbounded, " * 50
    )
    halluc_cdd, _ = cdd([gt_text_str], [hallucinated_text])
    halluc_cer = cer(gt_text_str, hallucinated_text)

    # Perfect parsing (no errors)
    perfect_cdd, _ = cdd([gt_text_str], [gt_text_str])

    mo.md(
        f"## Character Distribution Divergence (CDD)\n\n"
        f"| Scenario | Characters | CER | CDD |\n"
        f"|----------|-----------|-----|-----|\n"
        f"| Perfect parsing | {len(gt_text_str)} | 0 | {perfect_cdd:.4f} |\n"
        f"| Erroneous parsing | {len(detected_text_chars)} | — | {err_cdd:.4f} |\n"
        f"| LLM hallucination | {len(hallucinated_text)} | {halluc_cer:.4f} | {halluc_cdd:.4f} |\n\n"
        f"CER is not reported for erroneous parsing because character order is not "
        f"comparable across different bounding box sets — this is precisely the scenario "
        f"where CDD provides value as an order-agnostic metric.\n\n"
        f"The hallucination simulates an LLM that starts correctly but gets locked "
        f'repeating *"His boxes unbounded,"* 50 times — producing astronomical CER '
        f"and a distinctive CDD from the skewed character distribution."
    )
    return all_chars, cer


@app.cell
def _(all_chars, cdd, cer, gt_boxes, np, pd, pred_boxes):
    """Sweep OCR corruption levels and compute CDD vs actual CER."""
    import random
    import string
    import pkg_resources

    from collections import Counter
    from scrambledtext import ProbabilityDistributions, CorruptionEngine

    # Load built-in probability distributions
    _json_path = pkg_resources.resource_filename("scrambledtext", "corruption_distribs.json")
    _distribs = ProbabilityDistributions.load_from_json(_json_path)

    # Printable characters for random corruption
    _random_chars = string.ascii_letters + string.digits + string.punctuation + " "

    def _texts_per_region(all_chars, regions):
        """Return list of text strings, one per region."""
        region_chars = [[] for _ in regions]
        for char in all_chars:
            cx = char["bbox"][0] + char["bbox"][2] / 2
            cy = char["bbox"][1] + char["bbox"][3] / 2
            for i, r in enumerate(regions):
                if r["x"] <= cx <= r["x"] + r["width"] and r["y"] <= cy <= r["y"] + r["height"]:
                    region_chars[i].append(char["char"])
                    break
        return ["".join(chars) for chars in region_chars]

    def _random_corrupt(text, rate):
        return "".join(random.choice(_random_chars) if random.random() < rate else c for c in text)

    def _corrupt_regions_and_measure(region_texts, target_cer, engine, corrupt_fn):
        """Corrupt each region independently, return weighted CER and concatenated text."""
        corrupted_parts = []
        weighted_cer_num = 0.0
        total_chars = 0

        for text in region_texts:
            if not text:
                continue
            n = len(text)
            corrupted = text if target_cer == 0 else corrupt_fn(text)
            corrupted_parts.append(corrupted)
            weighted_cer_num += cer(text, corrupted) * n
            total_chars += n

        weighted_cer = weighted_cer_num / total_chars if total_chars > 0 else 0.0
        return weighted_cer, "".join(corrupted_parts)

    _gt_text_str = "".join(c["char"] for c in all_chars)
    _perfect_region_texts = _texts_per_region(all_chars, gt_boxes)
    _erroneous_region_texts = _texts_per_region(all_chars, pred_boxes)
    _erroneous_uncorrupted_str = "".join(_erroneous_region_texts)

    _gt_counter = Counter(_gt_text_str)
    _parsing_counter = Counter(_erroneous_uncorrupted_str)

    # Corruption levels to sweep
    _cer_levels = np.arange(0, 0.1, 0.001)
    _n_repeats = 30

    _rows = []
    _char_count_rows = []
    for _target_cer in _cer_levels:
        for _repeat in range(_n_repeats):
            if _target_cer == 0:
                _e_cer_val, _e_text = 0.0, _erroneous_uncorrupted_str
                _r_cer_val, _r_text = 0.0, "".join(_perfect_region_texts)
            else:
                _engine = CorruptionEngine(
                    _distribs.conditional,
                    _distribs.substitutions,
                    _distribs.insertions,
                    target_wer=1.0,
                    target_cer=_target_cer,
                )
                _realistic_fn = lambda t: _engine.corrupt_text(t)[0]
                _random_fn = lambda t: _random_corrupt(t, _target_cer)

                # Corrupt the erroneous text once — derive both d_total and d_ocr from it
                _e_cer_val, _e_text = _corrupt_regions_and_measure(
                    _erroneous_region_texts, _target_cer, _engine, _realistic_fn
                )
                _r_cer_val, _r_text = _corrupt_regions_and_measure(
                    _perfect_region_texts, _target_cer, _engine, _random_fn
                )

            # d_total: GT vs corrupted erroneous text (parsing + OCR combined)
            _d_total_val, _ = cdd([_gt_text_str], [_e_text])
            # d_ocr: erroneous uncorrupted vs corrupted erroneous text (OCR only)
            _d_ocr_val, _ = cdd([_erroneous_uncorrupted_str], [_e_text])
            # d_random: GT vs randomly corrupted perfect text
            _d_random_val, _ = cdd([_gt_text_str], [_r_text])

            _rows.extend(
                [
                    {
                        "target_cer": float(_target_cer),
                        "repeat": int(_repeat),
                        "condition": "$d_{total}$",
                        "actual_cer": float(_e_cer_val),
                        "cdd": float(_d_total_val),
                    },
                    {
                        "target_cer": float(_target_cer),
                        "repeat": int(_repeat),
                        "condition": "$d_{ocr}$",
                        "actual_cer": float(_e_cer_val),
                        "cdd": float(_d_ocr_val),
                    },
                    {
                        "target_cer": float(_target_cer),
                        "repeat": int(_repeat),
                        "condition": "$d_{random}$",
                        "actual_cer": float(_r_cer_val),
                        "cdd": float(_d_random_val),
                    },
                ]
            )

            _char_count_rows.append(
                {
                    "target_cer": float(_target_cer),
                    "repeat": int(_repeat),
                    "actual_cer_realistic": float(_e_cer_val),
                    "actual_cer_random": float(_r_cer_val),
                    "counter_realistic": Counter(_e_text),
                    "counter_random": Counter(_r_text),
                }
            )

    sweep_df = pd.DataFrame(_rows)
    sweep_summary_df = (
        sweep_df.groupby(["target_cer", "condition"], as_index=False)
        .agg(actual_cer_mean=("actual_cer", "mean"), cdd_mean=("cdd", "mean"))
        .sort_values(["condition", "target_cer"], kind="stable")
    )

    sweep_gt_text_str = _gt_text_str
    sweep_erroneous_uncorrupted_str = _erroneous_uncorrupted_str
    sweep_char_counts = _char_count_rows
    sweep_gt_counter = _gt_counter
    sweep_parsing_counter = _parsing_counter
    return (
        Counter,
        sweep_char_counts,
        sweep_erroneous_uncorrupted_str,
        sweep_gt_counter,
        sweep_gt_text_str,
        sweep_parsing_counter,
        sweep_summary_df,
    )


@app.cell
def _(
    Counter,
    aes,
    figure_path,
    ggplot,
    labs,
    mo,
    pd,
    sweep_char_counts,
    sweep_gt_counter,
    sweep_parsing_counter,
    theme,
    theme_minimal,
):
    """Character distribution comparison at ~5% CER for different corruption types."""

    from plotnine import element_text, facet_wrap, geom_col, scale_x_discrete

    _cer_lo, _cer_hi = 0.045, 0.055

    # Aggregate character counters for samples within the CER range
    _realistic_agg = Counter()
    _random_agg = Counter()
    _n_realistic = 0
    _n_random = 0

    for _row in sweep_char_counts:
        if _cer_lo <= _row["actual_cer_realistic"] <= _cer_hi:
            _realistic_agg += _row["counter_realistic"]
            _n_realistic += 1
        if _cer_lo <= _row["actual_cer_random"] <= _cer_hi:
            _random_agg += _row["counter_random"]
            _n_random += 1

    # Top 30 characters by ground truth frequency (descending)
    _top_chars = [ch for ch, _ in sweep_gt_counter.most_common(15)]
    _gt_total = sum(sweep_gt_counter.values())
    _top_coverage = sum(sweep_gt_counter[ch] for ch in _top_chars) / _gt_total

    # Build long-format DataFrame
    def _make_rows(counter, condition, chars):
        total = sum(counter.values())
        if total == 0:
            return []
        return [
            {
                "char": "␣" if ch == " " else ch,
                "proportion": counter.get(ch, 0) / total,
                "condition": condition,
            }
            for ch in chars
        ]

    _plot_rows = []
    _plot_rows.extend(_make_rows(sweep_gt_counter, "Ground truth", _top_chars))
    _plot_rows.extend(_make_rows(sweep_parsing_counter, "Parsing only", _top_chars))
    _plot_rows.extend(_make_rows(_realistic_agg, "Parsing + realistic OCR", _top_chars))
    _plot_rows.extend(_make_rows(_random_agg, "Random corruption", _top_chars))

    _plot_df = pd.DataFrame(_plot_rows)
    _plot_df["condition"] = pd.Categorical(
        _plot_df["condition"],
        categories=["Ground truth", "Parsing only", "Parsing + realistic OCR", "Random corruption"],
        ordered=True,
    )
    _char_order = ["␣" if ch == " " else ch for ch in _top_chars]

    _char_dist_plot = (
        ggplot(_plot_df, aes(x="char", y="proportion", fill="condition"))
        + geom_col(show_legend=True, position="dodge")
        + scale_x_discrete(limits=_char_order)
        + labs(
            x="Character",
            y="Proportion",
            title="Character distributions at ~5% CER\n(top 30 characters by ground truth frequency)",
        )
        + theme_minimal()
        + theme(
            axis_text_x=element_text(size=7, family="monospace"),
            figure_size=(10, 10),
        )
    )

    _char_dist_plot.save(figure_path / "char_distribution_5pct_cer.png", dpi=300, verbose=False)

    mo.vstack(
        [
            _char_dist_plot,
            mo.md(
                f"Top 30 characters account for **{_top_coverage:.1%}** of ground truth text. "
                f"Samples used — realistic OCR: {_n_realistic}, random corruption: {_n_random} "
                f"(filtered to {_cer_lo*100:.1f}–{_cer_hi*100:.1f}% actual CER)."
            ),
        ]
    )
    return element_text, facet_wrap, geom_col, scale_x_discrete


@app.cell
def _(
    Counter,
    aes,
    element_text,
    facet_wrap,
    figure_path,
    geom_col,
    geom_vline,
    ggplot,
    labs,
    mo,
    pd,
    scale_x_discrete,
    sweep_char_counts,
    sweep_gt_counter,
    sweep_parsing_counter,
    theme,
    theme_minimal,
):
    """Delta from ground truth character distribution at ~5% CER."""
    from plotnine import scale_fill_manual

    _cer_lo, _cer_hi = 0.045, 0.055

    # Aggregate character counters for samples within the CER range
    _realistic_agg = Counter()
    _random_agg = Counter()
    _n_realistic = 0
    _n_random = 0

    for _row in sweep_char_counts:
        if _cer_lo <= _row["actual_cer_realistic"] <= _cer_hi:
            _realistic_agg += _row["counter_realistic"]
            _n_realistic += 1
        if _cer_lo <= _row["actual_cer_random"] <= _cer_hi:
            _random_agg += _row["counter_random"]
            _n_random += 1

    # Proportions for each distribution
    def _to_props(counter):
        total = sum(counter.values())
        return {ch: count / total for ch, count in counter.items()} if total > 0 else {}

    _gt_props = _to_props(sweep_gt_counter)
    _parsing_props = _to_props(sweep_parsing_counter)
    _realistic_props = _to_props(_realistic_agg)
    _random_props = _to_props(_random_agg)

    # All characters across all distributions
    _all_chars = (
        set(sweep_gt_counter) | set(sweep_parsing_counter) | set(_realistic_agg) | set(_random_agg)
    )
    _gt_chars = set(sweep_gt_counter)

    # Order: GT characters by frequency (descending), then new characters alphabetically
    _gt_ordered = [ch for ch, _ in sweep_gt_counter.most_common()]
    _new_ordered = sorted(_all_chars - _gt_chars)
    _char_order_raw = _gt_ordered + _new_ordered
    _separator_pos = len(_gt_ordered) + 0.5  # for visual separator

    def _display(ch):
        return "␣" if ch == " " else ch

    _char_order = [_display(ch) for ch in _char_order_raw]

    # Build delta rows
    _conditions = {
        "Parsing only": _parsing_props,
        "Parsing + realistic OCR": _realistic_props,
        "Random corruption": _random_props,
    }

    _plot_rows = []
    for cond_name, props in _conditions.items():
        for ch in _char_order_raw:
            delta = props.get(ch, 0.0) - _gt_props.get(ch, 0.0)
            in_gt = ch in _gt_chars
            _plot_rows.append(
                {
                    "char": _display(ch),
                    "delta": delta,
                    "condition": cond_name,
                    "char_type": "In ground truth" if in_gt else "New character",
                }
            )

    _plot_df = pd.DataFrame(_plot_rows)
    _plot_df["condition"] = pd.Categorical(
        _plot_df["condition"],
        categories=["Parsing only", "Parsing + realistic OCR", "Random corruption"],
        ordered=True,
    )

    _delta_plot = (
        ggplot(_plot_df, aes(x="char", y="delta", fill="char_type"))
        + geom_col()
        + geom_vline(xintercept=_separator_pos, linetype="--", color="grey", alpha=0.5)
        + facet_wrap("~condition", ncol=1)
        + scale_x_discrete(limits=_char_order)
        + scale_fill_manual(values={"In ground truth": "#4878D0", "New character": "#D65F5F"})
        + labs(
            x="Character",
            y="Change in proportion (vs ground truth)",
            title="Change in character distribution at ~5% CER",
            fill="",
        )
        + theme_minimal()
        + theme(
            axis_text_x=element_text(size=6, family="monospace"),
            legend_position="bottom",
            figure_size=(12, 8),
        )
    )

    # Summary stats per condition
    _summary_parts = []
    for cond_name, props in _conditions.items():
        new_chars = {ch for ch in props if ch not in _gt_chars}
        new_mass = sum(props.get(ch, 0.0) for ch in new_chars)
        _summary_parts.append(
            f"**{cond_name}**: {len(new_chars)} new characters, "
            f"{new_mass:.2%} of mass in new characters"
        )

    _delta_plot.save(figure_path / "char_distribution_delta_5pct_cer.png", dpi=300, verbose=False)

    mo.vstack(
        [
            _delta_plot,
            mo.md(
                f"Characters left of the dashed line exist in the ground truth (ordered by frequency); "
                f"characters to the right are newly introduced by corruption.\n\n"
                + " | ".join(_summary_parts)
            ),
        ]
    )
    return


@app.cell
def _(
    aes,
    cdd,
    figure_path,
    geom_hline,
    geom_line,
    ggplot,
    labs,
    mo,
    sweep_erroneous_uncorrupted_str,
    sweep_gt_text_str,
    sweep_summary_df,
    theme,
    theme_minimal,
):
    """Combined CDD decomposition plot: d_total, d_ocr, d_random, and d_parsing."""

    # d_parsing is constant — the irreducible cost of wrong bounding boxes
    _d_parsing, _ = cdd([sweep_gt_text_str], [sweep_erroneous_uncorrupted_str])

    _combined_plot = (
        ggplot(sweep_summary_df, aes(x="actual_cer_mean", y="cdd_mean", color="condition"))
        + geom_line()
        + geom_hline(yintercept=_d_parsing, linetype="--", color="black")
        + labs(
            x="Measured CER",
            y="CDD",
            title="Impact of different corruption types on\nCharacter Distribution Distance",
            color="",
        )
        + theme_minimal()
        + theme(legend_position="bottom")
    )

    _combined_plot.save(figure_path / "cdd_decomposition.png", dpi=300, verbose=False)

    mo.vstack(
        [
            _combined_plot,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
