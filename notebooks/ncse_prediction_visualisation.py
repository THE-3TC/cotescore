import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image
    import marimo as mo

    from cotescore.visualisation import compute_cote_masks, visualize_cote_states
    from cotescore.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks
    from cotescore.metrics import cote_score, coverage, overlap, trespass, excess, mean_iou, f1

    _data_dir = Path("data/file_predictions")
    image_dir = Path("data/ncse_images")

    yolo_df = pd.read_csv(_data_dir / "yolo_predictions.csv")
    heron_df = pd.read_csv(_data_dir / "heron_predictions.csv")
    ppdoc_l = pd.read_csv(_data_dir / "ppdoc_l_predictions.csv")
    return (
        Image,
        Path,
        boxes_to_gt_ssu_map,
        boxes_to_pred_masks,
        compute_cote_masks,
        cote_score,
        coverage,
        excess,
        f1,
        heron_df,
        image_dir,
        mean_iou,
        mo,
        np,
        overlap,
        pd,
        plt,
        ppdoc_l,
        trespass,
        visualize_cote_states,
        yolo_df,
    )


@app.cell
def _(heron_df, yolo_df):
    """Diagnostic: check pred/gt row counts per model."""
    for _name, _df in [("YOLO", yolo_df), ("Heron", heron_df)]:
        _gt = (_df.source == "gt").sum()
        _pred = (_df.source == "pred").sum()
        print(f"{_name}: {_gt} GT rows, {_pred} pred rows across {_df.filename.nunique()} images")
    return


@app.cell
def _():
    return


@app.cell
def _(
    Image,
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    compute_cote_masks,
    coverage,
    excess,
    image_dir,
    np,
    overlap,
    plt,
    trespass,
    visualize_cote_states,
):
    def make_cote_figure(df, filename, figsize=(12, 8), title=None):
        img_df = df[df.filename == filename]
        gt_boxes = img_df[img_df.source == "gt"].to_dict("records")
        pred_boxes = img_df[img_df.source == "pred"].to_dict("records")

        image_array = np.array(Image.open(image_dir / filename).convert("RGB"))
        actual_h, actual_w = image_array.shape[:2]
        csv_w = int(img_df.image_width.iloc[0])
        csv_h = round(actual_h * csv_w / actual_w)

        gt_ssu_map = boxes_to_gt_ssu_map(gt_boxes, csv_w, csv_h, actual_w, actual_h)
        pred_masks = boxes_to_pred_masks(pred_boxes, csv_w, csv_h, actual_w, actual_h)

        masks = compute_cote_masks(gt_ssu_map, pred_masks)

        cov = coverage(gt_ssu_map, pred_masks)
        ovl = overlap(gt_ssu_map, pred_masks)
        tres = trespass(gt_ssu_map, pred_masks)
        exc = excess(gt_ssu_map, pred_masks)

        fig, ax = plt.subplots(figsize=figsize, dpi=72)
        patches = visualize_cote_states(image_array, masks, ax)
        fig.legend(
            handles=patches,
            loc="lower center",
            ncol=max(len(patches), 1),
            framealpha=0.9,
            fontsize=20,
        )

        metrics_text = (
            f"Coverage: {cov:.3f}   Overlap: {ovl:.3f}   "
            f"Trespass: {tres:.3f}   Excess: {exc:.3f}   "
            f"GT boxes: {len(gt_boxes)}   Pred boxes: {len(pred_boxes)}"
        )
        _full_title = f"{title}\n{metrics_text}" if title else metrics_text
        fig.suptitle(_full_title, fontsize=30, y=1.02)
        plt.tight_layout()
        plt.close(fig)
        return fig

    return


@app.cell
def _(
    Image,
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    compute_cote_masks,
    cote_score,
    image_dir,
    np,
    plt,
    visualize_cote_states,
):
    def make_multi_model_figure(
        dfs, filename, panel_width=10, metrics_col_width=1.8, dpi=72, show_metrics=True
    ):
        """Render one image with all models side-by-side in a single figure.

        Args:
            dfs: List of (model_name, dataframe) tuples.
            filename: Image filename to visualise.
            panel_width: Width in inches of each image panel.
            metrics_col_width: Width in inches of the metrics column beside each panel.
            dpi: Figure resolution.
            show_metrics: If False, omit the metrics text columns (improves H:W ratio for publication).

        Returns:
            matplotlib Figure with one panel per model, shared legend and title.
        """
        # Peek at image to compute aspect ratio so panels have no whitespace.
        _img_peek = Image.open(image_dir / filename)
        _iw, _ih = _img_peek.size
        panel_height = panel_width * (_ih / _iw)

        n = len(dfs)
        if show_metrics:
            fig_width = n * (panel_width + metrics_col_width)
        else:
            fig_width = n * panel_width
        # constrained_layout automatically reserves space for suptitle and legend
        fig = plt.figure(figsize=(fig_width, panel_height), dpi=dpi, layout="constrained")

        if show_metrics:
            # Alternating columns: [image, metrics, image, metrics, ...]
            gs = fig.add_gridspec(
                1,
                n * 2,
                width_ratios=[panel_width, metrics_col_width] * n,
                wspace=0.02,
            )
        else:
            gs = fig.add_gridspec(1, n, wspace=0.02)

        best_patches = []
        for i, (model_name, df) in enumerate(dfs):
            if show_metrics:
                img_ax = fig.add_subplot(gs[0, i * 2])
                txt_ax = fig.add_subplot(gs[0, i * 2 + 1])
            else:
                img_ax = fig.add_subplot(gs[0, i])

            img_df = df[df.filename == filename]
            gt_boxes = img_df[img_df.source == "gt"].to_dict("records")
            pred_boxes = img_df[img_df.source == "pred"].to_dict("records")

            image_array = np.array(Image.open(image_dir / filename).convert("RGB"))
            actual_h, actual_w = image_array.shape[:2]
            csv_w = int(img_df.image_width.iloc[0])
            csv_h = round(actual_h * csv_w / actual_w)

            gt_ssu_map = boxes_to_gt_ssu_map(gt_boxes, csv_w, csv_h, actual_w, actual_h)
            pred_masks = boxes_to_pred_masks(pred_boxes, csv_w, csv_h, actual_w, actual_h)

            masks = compute_cote_masks(gt_ssu_map, pred_masks)

            cot, cov, ovl, tres, exc = cote_score(gt_ssu_map, pred_masks)

            patches = visualize_cote_states(image_array, masks, img_ax)
            if len(patches) > len(best_patches):
                best_patches = patches

            img_ax.set_title(model_name, fontsize=20, pad=4)

            if show_metrics:
                txt_ax.axis("off")
                metrics_text = (
                    f"CoTE: {cot:.3f}\n\n"
                    f"──────\n\n"
                    f"Cov:  {cov:.3f}\n\n"
                    f"Ovl:  {ovl:.3f}\n\n"
                    f"Tres: {tres:.3f}\n\n"
                    f"Exc:  {exc:.3f}\n\n"
                    f"──────\n\n"
                    f"Bboxes\n\n"
                    f"GT:   {len(gt_boxes)}\n\n"
                    f"Pred: {len(pred_boxes)}"
                )
                txt_ax.text(
                    0.1,
                    0.5,
                    metrics_text,
                    transform=txt_ax.transAxes,
                    fontsize=20,
                    verticalalignment="center",
                    horizontalalignment="left",
                    fontfamily="monospace",
                )

        fig.suptitle(f"Model parsing performance on {filename[0:-4]}", fontsize=25)
        fig.legend(
            handles=best_patches,
            loc="lower center",
            ncol=max(len(best_patches), 1),
            framealpha=0.9,
            fontsize=18,
        )
        plt.close(fig)
        return fig

    return (make_multi_model_figure,)


@app.cell
def _(heron_df):
    heron_df["filename"].unique()
    return


@app.cell
def _(heron_df, make_multi_model_figure, mo, ppdoc_l, yolo_df):

    _filename = "TTW_1868-05-16_page_5.png"
    _fig = make_multi_model_figure(
        [("YOLO", yolo_df), ("Heron", heron_df), ("PPDoc-L", ppdoc_l)],
        _filename,
        show_metrics=False,
    )
    mo.as_html(_fig)
    return


@app.cell
def _(
    Image,
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    cote_score,
    f1,
    heron_df,
    image_dir,
    mean_iou,
    mo,
    np,
    pd,
    ppdoc_l,
    yolo_df,
):

    _filename = "TTW_1868-05-16_page_5.png"
    _dfs = [("YOLO", yolo_df), ("Heron", heron_df), ("PPDoc-L", ppdoc_l)]

    _rows = []
    for _model_name, _df in _dfs:
        _img_df = _df[_df.filename == _filename]
        _gt_boxes = _img_df[_img_df.source == "gt"].to_dict("records")
        _pred_boxes = _img_df[_img_df.source == "pred"].to_dict("records")

        _image_array = np.array(Image.open(image_dir / _filename).convert("RGB"))
        _actual_h, _actual_w = _image_array.shape[:2]
        _csv_w = int(_img_df.image_width.iloc[0])
        _csv_h = round(_actual_h * _csv_w / _actual_w)

        _gt_ssu_map = boxes_to_gt_ssu_map(_gt_boxes, _csv_w, _csv_h, _actual_w, _actual_h)
        _pred_masks = boxes_to_pred_masks(_pred_boxes, _csv_w, _csv_h, _actual_w, _actual_h)

        _cot, _cov, _ovl, _tres, _exc = cote_score(_gt_ssu_map, _pred_masks)
        _miou = mean_iou(_pred_boxes, _gt_boxes)
        _f1 = f1(_pred_boxes, _gt_boxes)

        _rows.append(
            {
                "Model": _model_name,
                "CoTE": _cot,
                "Coverage": _cov,
                "Overlap": _ovl,
                "Trespass": _tres,
                "Excess": _exc,
                "mIoU": _miou,
                "F1": _f1,
            }
        )

    _metrics_df = pd.DataFrame(_rows).set_index("Model")

    import latex_tables as _lt

    _lt.latex_table(
        _lt.bold_best_cols(_metrics_df, lower_cols=["Overlap", "Trespass", "Excess"], fmt="{:.2f}"),
        caption=f"Per-model metrics for \\texttt{{{_filename[:-4]}}}, bold is best in column",
        label="tab:model-metrics",
    )
    mo.as_html(_metrics_df.style.format("{:.2f}"))
    return


@app.cell
def _(Path, heron_df, make_multi_model_figure, plt, ppdoc_l, yolo_df):
    _output_dir = Path("data/combined_predictions")
    _output_dir.mkdir(parents=True, exist_ok=True)

    _filenames = (
        set(yolo_df["filename"].unique())
        & set(heron_df["filename"].unique())
        & set(ppdoc_l["filename"].unique())
    )

    for _filename in sorted(_filenames):
        _fig = make_multi_model_figure(
            [("YOLO", yolo_df), ("Heron", heron_df), ("PPDoc-L", ppdoc_l)],
            _filename,
            show_metrics=False,
            dpi=72,
        )
        _fig.savefig(_output_dir / _filename, bbox_inches="tight", dpi=72, facecolor="white")
        plt.close(_fig)
        print(f"Saved {_filename}")

    print(f"Done. {len(_filenames)} images saved to {_output_dir}")
    return


if __name__ == "__main__":
    app.run()
