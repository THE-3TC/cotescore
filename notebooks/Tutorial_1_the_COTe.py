import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # COTe on NCSE: a worked example

    This is the first of two notebooks. Here we evaluate **document layout parsing** with the
    **COTe** score. The second notebook (`ncse_cev_tutorial.py`) picks up the predictions saved
    here and evaluates the **OCR** stage with the CEV.

    COTe decomposes layout quality into four quantities, each defined over the pixels of a
    **Structural Semantic Unit** (SSU) — a semantically complete region such as a whole article,
    which may be split across several columns:

    - **Coverage (C)** — the fraction of ground-truth SSU pixels that some prediction covers.
      Higher is better.
    - **Overlap (O)** — redundant prediction: SSU pixels claimed by more than one predicted
      region, counted once per extra claim. Lower is better.
    - **Trespass (T)** — predictions that straddle a boundary, pulling pixels from an SSU other
      than the one they mostly belong to. This is what punishes merging two articles into one
      region. Lower is better.
    - **excess (E)** — predicted pixels that fall on background. Reported alongside, but *not*
      part of the composite score.

    $$\text{COTe} = C - O - T$$

    Every pixel is assigned exactly one state, so the components cannot double-count.

    ## Why not IoU or F1?

    IoU and F1 match *one predicted box to one ground-truth box*. That assumes the prediction and
    the ground truth agree on what a "thing" is. In newspapers they do not: a layout model emits
    columns and blocks, while the semantically meaningful unit is the article. In the NCSE test
    set the ground truth averages about **12 boxes but only about 5 SSUs per page** — so a model
    can find every column perfectly and still score badly under F1 simply for disagreeing about
    granularity.

    COTe scores the pixels, not the box count, so it is indifferent to how a correct region is
    subdivided. We compute F1 and mean IoU alongside COTe so you can watch them disagree.

    By the end of this notebook you will have, for **two** layout models, a per-page score table,
    a dataset-level summary, and a folder of colour-coded diagnostic images showing exactly where
    each model's pixels went.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Configuration

    Everything the notebook needs is a folder of page images and a ground-truth CSV. Both are
    plain strings — point them at your own data to reuse this notebook.

    The ground-truth CSV needs the columns `filename, x1, y1, x2, y2, class, ssu_id, ssu_class`.
    If it also carries `image_width`, coordinates are rescaled automatically when the CSV was
    recorded at a different resolution than the images on disk.
    """
    )
    return


@app.cell
def _():
    # --- Point these at your data -------------------------------------------------
    NCSE_IMAGES_DIR = "/teamspace/studios/this_studio/ncse/images"
    NCSE_GT_CSV = "/teamspace/studios/this_studio/ncse/ncse_testset_bboxes.csv"

    # --- Where results are written ------------------------------------------------
    PRED_DIR = "outputs/predictions"
    FIG_DIR = "outputs/figures"

    # --- Compute ------------------------------------------------------------------
    DEVICE = "cuda"  # "cuda", "mps" or "cpu"
    BATCH_SIZE = 8
    N_FIGURES = 5  # pages to render per model; None renders all
    return (
        BATCH_SIZE,
        DEVICE,
        FIG_DIR,
        NCSE_GT_CSV,
        NCSE_IMAGES_DIR,
        N_FIGURES,
        PRED_DIR,
    )


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    from cotescore.adapters import (
        boxes_to_gt_ssu_map,
        boxes_to_pred_masks,
        compute_canvas,
    )
    from cotescore.dataset import NCSEDataset
    from cotescore.layout import cote_score, f1, mean_iou
    from cotescore.types import GTBoxes
    from cotescore.visualisation import compute_cote_masks, visualize_cote_states

    return (
        GTBoxes,
        Image,
        NCSEDataset,
        Path,
        boxes_to_gt_ssu_map,
        boxes_to_pred_masks,
        compute_canvas,
        compute_cote_masks,
        cote_score,
        f1,
        mean_iou,
        np,
        pd,
        plt,
        visualize_cote_states,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Load the dataset

    `NCSEDataset` joins its paths with `pathlib`, so passing an **absolute** path as
    `csv_filename` or `images_subdir` overrides the root argument entirely. That is what lets the
    images folder and the ground-truth CSV live in unrelated places.

    Each sample is a dict: `{"image_path", "annotations", "filename"}`, where every annotation is
    `{"x", "y", "width", "height", "class", "ssu_id", "ssu_class", "confidence", "page_id"}` in
    **XYWH pixel** coordinates.
    """
    )
    return


@app.cell
def _(NCSE_GT_CSV, NCSE_IMAGES_DIR, NCSEDataset, pd):
    dataset = NCSEDataset(
        ".",
        split="test",
        csv_filename=NCSE_GT_CSV,
        images_subdir=NCSE_IMAGES_DIR,
    )
    dataset.load()

    # Coverage check. The loader silently skips any CSV filename with no matching image
    # file, so a mismatch would otherwise show up only as a quietly short results table.
    _csv_pages = pd.read_csv(NCSE_GT_CSV)["filename"].nunique()
    print(f"Pages resolved: {len(dataset)} / {_csv_pages} in the ground-truth CSV")
    if len(dataset) < _csv_pages:
        print(
            "  WARNING: some pages did not resolve. Image filenames must match the CSV\n"
            "  'filename' column exactly (including extension)."
        )
    return (dataset,)


@app.cell
def _(dataset, pd):
    # How much does prediction granularity differ from semantic granularity?
    _rows = [
        {
            "filename": dataset[_i]["filename"],
            "gt_boxes": len(dataset[_i]["annotations"]),
            "ssus": len({_a["ssu_id"] for _a in dataset[_i]["annotations"]}),
        }
        for _i in range(len(dataset))
    ]
    granularity_df = pd.DataFrame(_rows)
    print(
        f"Mean GT boxes per page: {granularity_df.gt_boxes.mean():.1f}\n"
        f"Mean SSUs per page:     {granularity_df.ssus.mean():.1f}"
    )
    granularity_df.head(10)
    return (granularity_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### What an SSU looks like

    Below, one page's ground-truth boxes are coloured **by `ssu_id`**. Boxes sharing a colour
    belong to the same semantic unit — typically one article running across several columns.
    This grouping is the whole point: COTe asks whether a prediction respects these units, not
    whether it reproduced the box count.
    """
    )
    return


@app.cell
def _(Image, dataset, plt):
    _sample = dataset[0]
    _img = Image.open(_sample["image_path"]).convert("RGB")

    _fig, _ax = plt.subplots(figsize=(7, 10))
    _ax.imshow(_img)
    _ssu_ids = sorted({_a["ssu_id"] for _a in _sample["annotations"]})
    _cmap = plt.get_cmap("tab10")
    for _ann in _sample["annotations"]:
        _c = _cmap(_ssu_ids.index(_ann["ssu_id"]) % 10)
        _ax.add_patch(
            plt.Rectangle(
                (_ann["x"], _ann["y"]),
                _ann["width"],
                _ann["height"],
                fill=False,
                edgecolor=_c,
                linewidth=2.5,
            )
        )
        _ax.text(
            _ann["x"] + 4,
            _ann["y"] + 22,
            f"SSU {_ann['ssu_id']}",
            color="white",
            fontsize=9,
            bbox={"facecolor": _c, "edgecolor": "none", "pad": 1.5},
        )
    _ax.set_title(
        f"{_sample['filename']}\n"
        f"{len(_sample['annotations'])} GT boxes, {len(_ssu_ids)} SSUs"
    )
    _ax.axis("off")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Load the layout models

    Both models implement the same `LayoutModel` contract, so everything downstream is identical:

    ```python
    model.predict_batch(paths) -> List[List[{"x","y","width","height","class","confidence"}]]
    ```

    - **Heron** — `docling-project/docling-layout-heron`, an RT-DETR detector fine-tuned for
      document layout, with a 17-class taxonomy.
    - **DocLayout-YOLO** — `juliozhao/DocLayout-YOLO-DocStructBench`, a YOLOv10 detector.

    Weights download from the Hugging Face hub on first `load()`. `doclayout_yolo` is imported
    lazily, so if it is not installed you will see the error here rather than at import time.
    """
    )
    return


@app.cell
def _(DEVICE):
    import sys
    from pathlib import Path as _P

    # The layout models live in `models/`, a top-level directory of the cotescore repo rather
    # than part of the installed package. Marimo puts the *notebook's* directory on sys.path,
    # not the repo root, so locate the root by walking up and add it explicitly.
    try:
        _start = _P(__file__).resolve().parent
    except NameError:
        _start = _P.cwd()
    for _cand in [_start, *_start.parents]:
        if (_cand / "models" / "docling_heron.py").exists():
            if str(_cand) not in sys.path:
                sys.path.insert(0, str(_cand))
            break
    else:
        raise RuntimeError(
            "Could not find the cotescore repo root (no models/docling_heron.py found "
            f"walking up from {_start}). Run this notebook from inside the repo."
        )

    from models.docling_heron import DoclingLayoutHeron
    from models.doclayout_yolo import DocLayoutYOLO

    heron = DoclingLayoutHeron(device=DEVICE)
    heron.load()

    yolo = DocLayoutYOLO(device=DEVICE)
    yolo.load()

    models = {"heron": heron, "yolo": yolo}
    print("Loaded:", ", ".join(models))
    return (models,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Predict and score

    We run one inference pass per model and score each page as we go.

    Ground truth is available as boxes, so we use COTe's **analytic bounding-box fast path**:
    wrap the GT in a `GTBoxes` (which also carries the image extent, because *excess* needs a
    background-area denominator) and pass predictions as an `(M, 4)` XYWH array. This avoids
    rasterising anything.

    ```python
    gt = GTBoxes(boxes=gt_xywh, ssu_ids=ssu_ids, image_width=w, image_height=h)
    cote, C, O, T, E = cote_score(gt, pred_xywh)
    ```

    `f1` and `mean_iou` take the annotation dicts directly and give us the instance-matching
    comparison.

    For production benchmarking, `benchmarks.runner.BenchmarkRunner.run_evaluation` does all of
    this with GPU/CPU pipelining and adds mAP. We write the loop out here because the loop is the
    thing being taught.
    """
    )
    return


@app.cell
def _(GTBoxes, Image, cote_score, f1, mean_iou, np):
    def score_page(sample, predictions):
        """Score one page. Returns (metrics_dict, image_width, image_height)."""
        with Image.open(sample["image_path"]) as _im:
            w, h = _im.size

        anns = sample["annotations"]
        gt = GTBoxes(
            boxes=np.array(
                [[a["x"], a["y"], a["width"], a["height"]] for a in anns], dtype=float
            ).reshape(-1, 4),
            ssu_ids=np.array([int(a["ssu_id"]) for a in anns], dtype=int),
            image_width=w,
            image_height=h,
        )
        pred_xywh = np.array(
            [[p["x"], p["y"], p["width"], p["height"]] for p in predictions], dtype=float
        ).reshape(-1, 4)

        cote, cov, ov, tr, ex = cote_score(gt, pred_xywh)

        return (
            {
                "filename": sample["filename"],
                "cote": cote,
                "coverage": cov,
                "overlap": ov,
                "trespass": tr,
                "excess": ex,
                "mean_iou": mean_iou(predictions, anns),
                "f1_50": f1(predictions, anns),
                "n_gt_boxes": len(anns),
                "n_ssus": len({a["ssu_id"] for a in anns}),
                "n_pred_boxes": len(predictions),
            },
            w,
            h,
        )

    return (score_page,)


@app.cell
def _(Image, np, pd, score_page):
    def run_model(model, model_name, dataset, batch_size=8):
        """One inference pass over the dataset. Returns (scores_df, predictions_df, preds_by_file)."""
        image_paths = [dataset[i]["image_path"] for i in range(len(dataset))]
        all_preds = model.predict_batch(image_paths, batch_size=batch_size)

        score_rows, flat_rows, preds_by_file = [], [], {}
        for idx, preds in enumerate(all_preds):
            sample = dataset[idx]
            metrics, w, h = score_page(sample, preds)
            metrics["model"] = model_name
            score_rows.append(metrics)
            preds_by_file[sample["filename"]] = preds

            # Flat interchange CSV: GT and predictions interleaved, one row per box.
            # Same schema as scripts/export_predictions.py, so notebook 2 can read it.
            base = {
                "filename": sample["filename"],
                "image_path": sample["image_path"],
                "image_width": w,
                "image_height": h,
                "model": model_name,
            }
            for a in sample["annotations"]:
                flat_rows.append(
                    {
                        **base,
                        "source": "gt",
                        "x": a["x"],
                        "y": a["y"],
                        "width": a["width"],
                        "height": a["height"],
                        "class": a["class"],
                        "confidence": a.get("confidence", 1.0),
                        "ssu_id": a["ssu_id"],
                    }
                )
            for p in preds:
                flat_rows.append(
                    {
                        **base,
                        "source": "pred",
                        "x": p["x"],
                        "y": p["y"],
                        "width": p["width"],
                        "height": p["height"],
                        "class": p.get("class"),
                        "confidence": p.get("confidence"),
                        "ssu_id": None,
                    }
                )

        return pd.DataFrame(score_rows), pd.DataFrame(flat_rows), preds_by_file

    return (run_model,)


@app.cell
def _(BATCH_SIZE, PRED_DIR, Path, dataset, models, pd, run_model):
    Path(PRED_DIR).mkdir(parents=True, exist_ok=True)

    scores, predictions, preds_by_model = {}, {}, {}
    for _name, _model in models.items():
        print(f"Running {_name} ...")
        _s, _p, _by_file = run_model(_model, _name, dataset, batch_size=BATCH_SIZE)
        scores[_name], predictions[_name], preds_by_model[_name] = _s, _p, _by_file
        _s.to_csv(f"{PRED_DIR}/ncse_{_name}_scores.csv", index=False)
        _p.to_csv(f"{PRED_DIR}/ncse_{_name}_predictions.csv", index=False)
        print(
            f"  {len(_s)} pages, {(_p.source == 'pred').sum()} predicted boxes "
            f"-> {PRED_DIR}/ncse_{_name}_predictions.csv"
        )

    all_scores = pd.concat(scores.values(), ignore_index=True)
    return all_scores, preds_by_model, scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Dataset-level results

    One row per model, averaged over pages.

    Read the row left to right. **coverage** near 1 means the model found the text. **overlap**
    is redundancy — the same SSU pixels claimed twice. **trespass** is the boundary error that
    punishes merging separate articles. **excess** is prediction spilling onto background.

    Then compare `cote` against `f1_50` and `mean_iou`. Where a model splits SSUs into columns —
    correct pixels, wrong box count — COTe stays high while F1 falls.
    """
    )
    return


@app.cell
def _(all_scores):
    _cols = [
        "cote",
        "coverage",
        "overlap",
        "trespass",
        "excess",
        "mean_iou",
        "f1_50",
        "n_pred_boxes",
    ]
    summary = all_scores.groupby("model")[_cols].mean().round(4)
    summary
    return (summary,)


@app.cell
def _(summary):
    # Direction of "better" per metric. Columns absent from both sets are descriptive
    # (e.g. box counts) and are never bolded — there is no better or worse box count.
    _higher_better = {"cote", "coverage", "mean_iou", "f1_50"}
    _lower_better = {"overlap", "trespass", "excess"}

    def df_to_markdown(df, caption):
        """Markdown table with the best value per scored column in bold."""
        lines = [f"**{caption}**", "", "| model | " + " | ".join(df.columns) + " |"]
        lines.append("|" + "---|" * (len(df.columns) + 1))
        best = {}
        for c in df.columns:
            if c in _higher_better:
                best[c] = df[c].max()
            elif c in _lower_better:
                best[c] = df[c].min()
        for name, row in df.iterrows():
            cells = [
                f"**{row[c]:.4f}**" if c in best and row[c] == best[c] else f"{row[c]:.4f}"
                for c in df.columns
            ]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        return "\n".join(lines)

    print(df_to_markdown(summary, "COTe on the NCSE test set"))
    return (df_to_markdown,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### COTe against F1, page by page

    Each point is a page. When the cloud sits above the diagonal, COTe is rewarding pixel-correct
    parsing that F1 penalises purely for disagreeing about how regions are subdivided.
    """
    )
    return


@app.cell
def _(all_scores, plt):
    _fig, _ax = plt.subplots(figsize=(6, 6))
    for _name, _grp in all_scores.groupby("model"):
        _ax.scatter(_grp["f1_50"], _grp["cote"], label=_name, alpha=0.75, s=45)
    _ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="y = x")
    _ax.set_xlabel("F1 @ IoU 0.5")
    _ax.set_ylabel("COTe")
    _ax.set_title("COTe vs F1, one point per page")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Visualise where the pixels went

    A score tells you *how much* was wrong; the pixel-state map tells you *what* was wrong.
    `compute_cote_masks` returns one mutually exclusive binary mask per state, so the colours
    partition the page:

    | colour | state | meaning |
    |---|---|---|
    | green | coverage | GT pixels correctly predicted |
    | amber | overlap | GT pixels claimed more than once |
    | red | trespass | prediction pulled in pixels from another SSU |
    | purple | overlap + trespass | both at once |
    | grey | missing | GT pixels no prediction reached |
    | blue | excess | prediction landed on background |

    Masks are computed on a canvas capped at 2000px on the long side, so the image is resized to
    match before drawing. Colours are fixed in `COTE_COLORS`, so figures are directly comparable
    across models. Output goes to one folder per model.
    """
    )
    return


@app.cell
def _(
    FIG_DIR,
    Image,
    Path,
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    compute_canvas,
    compute_cote_masks,
    np,
    plt,
    visualize_cote_states,
):
    def render_page(sample, predictions, title, out_path=None):
        """Render the COTe pixel-state overlay for one page."""
        with Image.open(sample["image_path"]) as _im:
            img = _im.convert("RGB")
            w, h = img.size

        canvas_w, canvas_h = compute_canvas(w, h, 2000)
        gt_map = boxes_to_gt_ssu_map(sample["annotations"], w, h, canvas_w, canvas_h)
        pred_masks = boxes_to_pred_masks(predictions, w, h, canvas_w, canvas_h)
        masks = compute_cote_masks(gt_map, pred_masks)

        # _draw_overlays does not resize: the image must match the mask canvas.
        img_arr = np.array(img.resize((canvas_w, canvas_h)))

        fig, ax = plt.subplots(figsize=(9, 12))
        patches = visualize_cote_states(img_arr, masks, ax=ax, show_missing=True)
        ax.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.9)
        ax.set_title(title)
        ax.axis("off")

        if out_path is not None:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
        return fig

    def render_model(model_name, dataset, preds_by_file, scores_df, limit=None):
        """Render every page for one model into FIG_DIR/<model_name>/."""
        out_dir = Path(FIG_DIR) / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        by_filename = scores_df.set_index("filename")["cote"].to_dict()

        n = len(dataset) if limit is None else min(limit, len(dataset))
        figs = []
        for i in range(n):
            sample = dataset[i]
            stem = Path(sample["filename"]).stem
            title = (
                f"{model_name} — {sample['filename']}\n"
                f"COTe = {by_filename.get(sample['filename'], float('nan')):.3f}"
            )
            fig = render_page(
                sample,
                preds_by_file[sample["filename"]],
                title,
                out_path=out_dir / f"{stem}_cote.png",
            )
            figs.append(fig)
            plt.close(fig)
        print(f"{model_name}: wrote {n} figures to {out_dir}")
        return figs

    return render_model, render_page


@app.cell
def _(FIG_DIR, N_FIGURES, dataset, models, preds_by_model, render_model, scores):
    for _name in models:
        render_model(
            _name,
            dataset,
            preds_by_model[_name],
            scores[_name],
            limit=N_FIGURES,
        )
    print(f"\nOne folder per model under {FIG_DIR}/")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Inspect a single page side by side

    The same page under both models. Differences in the colour mix are the differences in parsing
    behaviour: more red means more boundary-crossing merges, more grey means missed text, more
    amber means the model predicted the same region twice.
    """
    )
    return


@app.cell
def _(dataset, models, preds_by_model, render_page):
    _idx = 0
    _sample = dataset[_idx]
    [
        render_page(
            _sample,
            preds_by_model[_name][_sample["filename"]],
            f"{_name} — {_sample['filename']}",
        )
        for _name in models
    ]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## What we produced

    In `outputs/predictions/`:

    - `ncse_{model}_scores.csv` — per page: `cote, coverage, overlap, trespass, excess,
      mean_iou, f1_50` and box counts.
    - `ncse_{model}_predictions.csv` — flat box table, GT and predictions interleaved
      (`source ∈ {gt, pred}`). **Notebook 2 reads this file.**

    In `outputs/figures/{model}/` — one COTe pixel-state image per page.

    ---

    COTe has told us how well each model recovered the *geometry* of the page. It says nothing
    about whether the text inside those regions was read correctly. That is the CEV's job, and
    the two interact: bad parsing sends bad crops to the OCR engine, so a poor final transcription
    does not by itself tell you which stage to fix.

    Continue to **`ncse_cev_tutorial.py`**, which loads these predictions, runs two OCR engines,
    and uses COTe together with SpACER to attribute the error.
    """
    )
    return


if __name__ == "__main__":
    app.run()
