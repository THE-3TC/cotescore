import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # COTe on natural scenes

    Although COTe was developed for document understanding pipelines, there is nothing that prevents it being used in general machine vision tasks. In this notebook we compare the performance of COTe to Panoptic Quality (PQ), on two images from COCO panoptic. We apply two models YOLO26-instance and Mask2Former panoptic. In Panoptic segmentation each pixel can only have a single class, meaning that Overlap will become zero, in instance segmentation, Overlap is likely low but will still be present.

    **Panoptic Quality (PQ)**, the standard metric for this task.

    PQ matches each predicted segment to at most one ground-truth segment of the same class at
    IoU > 0.5. Its components:

    - **SQ** (segmentation quality): mean IoU of the matched pairs.
    - **RQ** (recognition quality): an F1 score over segments.

    $$\text{PQ} = \text{SQ} \times \text{RQ}$$

    PQ is an instance-matching metric, so it inherits F1's sensitivity to granularity. It also
    requires non-overlapping predictions, so it cannot say anything about overlap. COTe is
    computed over pixels and has neither restriction.


    This notebook is a simple demonstration of COTe on natural scenes deeper analysis is beyond it's scope/
    """)
    return


@app.cell
def _():
    # --- Point these at your data -------------------------------------------------
    DATA_DIR = "data/coco_data"  # <name>_<image_id>.jpg and <image_id>.png panoptic masks
    GT_JSON = "data/coco_data/coco_panoptic_gt.json"  # scripts/extract_coco_panoptic_gt.py
    PRED_DIR = "outputs/coco_demo"  # scripts/run_coco_panoptic_models.py

    # Mock predictions from perturbed GT, shown alongside any real model outputs
    USE_MOCKS = True
    return DATA_DIR, GT_JSON, PRED_DIR, USE_MOCKS


@app.cell
def _():
    import json
    from collections import defaultdict
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    from cotescore.layout import cote_score
    from cotescore.types import MaskInstance
    from cotescore.visualisation import compute_cote_masks, visualize_cote_states
    return (
        Image,
        MaskInstance,
        Path,
        compute_cote_masks,
        cote_score,
        defaultdict,
        json,
        np,
        pd,
        plt,
        visualize_cote_states,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Ground truth

    COCO panoptic stores each image's segmentation as an RGB PNG. The segment id is
    `R + 256·G + 256²·B`, and id `0` is **void** (unlabelled). The JSON's `segments_info`
    maps each segment id to a category, marks it as a *thing* (countable: person, elephant)
    or *stuff* (amorphous: grass, sky), and flags `iscrowd` regions.
    """)
    return


@app.cell
def _(DATA_DIR, GT_JSON, Image, Path, json, np):
    def rgb2id(png):
        png = png.astype(np.uint32)
        return png[..., 0] + 256 * png[..., 1] + 256 * 256 * png[..., 2]

    with open(GT_JSON) as _f:
        _gt = json.load(_f)

    categories = {c["id"]: c for c in _gt["categories"]}
    name_to_cat = {c["name"]: c["id"] for c in _gt["categories"]}

    samples = {}
    for _ann in _gt["annotations"]:
        _id = _ann["image_id"]
        _jpg = next(Path(DATA_DIR).glob(f"*_{_id}.jpg"))
        _png = np.array(Image.open(Path(DATA_DIR) / _ann["file_name"]).convert("RGB"))
        samples[_id] = {
            "image_path": _jpg,
            "image": np.array(Image.open(_jpg).convert("RGB")),
            "pan_map": rgb2id(_png),
            "segments": _ann["segments_info"],
        }
    print(f"Loaded {len(samples)} images: {sorted(samples)}")
    return categories, name_to_cat, samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Alignment check

    The mask and the image must have the same shape. Every id in the PNG must also appear in
    `segments_info`, and each segment's pixel count must match its recorded `area`. A failure
    here means the wrong PNG or a rescaled image.
    """)
    return


@app.cell
def _(np, pd, samples):
    _rows = []
    for _id, _s in samples.items():
        _ids_in_png = set(np.unique(_s["pan_map"]).tolist()) - {0}
        _ids_in_json = {seg["id"] for seg in _s["segments"]}
        _area_err = max(
            abs(int((_s["pan_map"] == seg["id"]).sum()) - seg["area"]) for seg in _s["segments"]
        )
        _rows.append(
            {
                "image_id": _id,
                "image_hw": _s["image"].shape[:2],
                "mask_hw": _s["pan_map"].shape,
                "shape_match": _s["image"].shape[:2] == _s["pan_map"].shape,
                "n_segments": len(_s["segments"]),
                "ids_match": _ids_in_png == _ids_in_json,
                "max_area_error_px": _area_err,
                "void_fraction": round(float((_s["pan_map"] == 0).mean()), 3),
            }
        )
    alignment_df = pd.DataFrame(_rows)
    assert alignment_df.shape_match.all(), "Image and panoptic mask shapes differ"
    assert alignment_df.ids_match.all(), "Segment ids in PNG and JSON differ"
    alignment_df
    return


@app.cell
def _(categories, np):
    GREY = (128, 128, 128)

    def cat_colour(cat_id):
        # The release's categories carry no colours, so seed one from the id: stable
        # across images and models.
        if cat_id not in categories:
            return GREY
        return tuple(np.random.default_rng(cat_id).integers(40, 255, size=3))

    def overlay_masks(image, masks, colours, alpha=0.6):
        """Blend coloured masks over an image and draw white segment boundaries."""
        out = image.astype(float).copy()
        label = np.zeros(image.shape[:2], dtype=np.int32)
        for _i, (_m, _c) in enumerate(zip(masks, colours), start=1):
            out[_m] = (1 - alpha) * out[_m] + alpha * np.asarray(_c, dtype=float)
            label[_m] = _i
        edges = np.zeros(label.shape, dtype=bool)
        edges[:-1] |= label[:-1] != label[1:]
        edges[:, :-1] |= label[:, :-1] != label[:, 1:]
        out[edges] = 255
        return out.astype(np.uint8)

    def label_segments(ax, masks, names, min_frac=0.005):
        """Write each segment's name on the mask pixel nearest its centroid."""
        for _m, _name in zip(masks, names):
            if _m.mean() < min_frac:
                continue
            _ys, _xs = np.nonzero(_m)
            _k = np.argmin((_ys - _ys.mean()) ** 2 + (_xs - _xs.mean()) ** 2)
            ax.text(
                _xs[_k],
                _ys[_k],
                _name,
                fontsize=8,
                ha="center",
                va="center",
                color="white",
                bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none", "pad": 1},
            )

    def cat_name(cat_id):
        return categories[cat_id]["name"] if cat_id in categories else "unmapped"
    return cat_colour, cat_name, label_segments, overlay_masks


@app.cell
def _(cat_colour, cat_name, label_segments, overlay_masks, plt, samples):
    _fig, _axes = plt.subplots(len(samples), 2, figsize=(12, 7 * len(samples)), squeeze=False)
    for (_id, _s), (_ax_img, _ax_gt) in zip(samples.items(), _axes):
        _masks = [_s["pan_map"] == seg["id"] for seg in _s["segments"]]
        _cats = [seg["category_id"] for seg in _s["segments"]]
        _ax_img.imshow(_s["image"])
        _ax_img.set_title(f"{_s['image_path'].name}  {_s['image'].shape[1]}×{_s['image'].shape[0]}")
        _ax_gt.imshow(overlay_masks(_s["image"], _masks, [cat_colour(c) for c in _cats]))
        label_segments(_ax_gt, _masks, [cat_name(c) for c in _cats])
        _ax_gt.set_title(f"Panoptic GT: {len(_masks)} segments (void left unpainted)")
        for _ax in (_ax_img, _ax_gt):
            _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Predictions

    Every prediction is a dict `{"mask", "category_id", "score"}` at the image's native
    resolution. Real model outputs are loaded from `PRED_DIR/<model>/<image_id>.npz`. Class
    names are mapped to COCO panoptic category ids by name, and anything unmapped gets `-1`.

    The **mock** predictions start from a perfect copy of the ground truth and apply one
    perturbation each:

    | mock | perturbation | what COTe should do | what PQ should do |
    |---|---|---|---|
    | `mock_perfect` | none | 1.0 | 1.0 |
    | `mock_shifted` | every segment translated by 5% of the width, 3% of the height | lose coverage, gain trespass | lower SQ, some segments drop below IoU 0.5 |
    | `mock_merged` | largest thing merged with a touching thing (or, failing that, the most-adjacent segment) | trespass | a TP becomes an FN |
    | `mock_split` | largest thing cut into three vertical strips of equal area | unchanged: same pixels, same unit | the TP becomes an FN and three FPs |
    | `mock_duplicated` | every thing predicted twice, the copy dilated by 5 px | overlap | nothing: PQ drops overlapping duplicates before scoring |
    | `mock_dropped` | largest thing removed | lose coverage | one FN |
    """)
    return


@app.cell
def _(categories, np):
    def gt_segments(sample):
        return [
            {
                "mask": sample["pan_map"] == seg["id"],
                "category_id": seg["category_id"],
                "isthing": bool(categories[seg["category_id"]]["isthing"]),
                "iscrowd": bool(seg["iscrowd"]),
            }
            for seg in sample["segments"]
        ]

    def dilate(mask, r):
        out = mask.copy()
        for _ in range(r):
            grown = out.copy()
            grown[1:] |= out[:-1]
            grown[:-1] |= out[1:]
            grown[:, 1:] |= out[:, :-1]
            grown[:, :-1] |= out[:, 1:]
            out = grown
        return out

    def mock_predictions(segs):
        h, w = segs[0]["mask"].shape
        base = [
            {"mask": s["mask"], "category_id": s["category_id"], "score": 1.0}
            for s in segs
            if not s["iscrowd"]
        ]
        things = sorted(
            [p for p, s in zip(base, [s for s in segs if not s["iscrowd"]]) if s["isthing"]],
            key=lambda p: -p["mask"].sum(),
        )
        largest = things[0]
        rest = [p for p in base if p is not largest]

        dx, dy = int(0.05 * w), int(0.03 * h)

        def shift(m):
            out = np.zeros_like(m)
            out[dy:, dx:] = m[: h - dy, : w - dx]
            return out

        # Prefer a touching thing (e.g. the rider on the elephant) so the merge is also
        # visible at the things-only scope; fall back to whichever segment touches most.
        ring = dilate(largest["mask"], 3) & ~largest["mask"]
        touching_things = [p for p in things[1:] if (p["mask"] & ring).any()]
        neighbour = max(touching_things or rest, key=lambda p: (p["mask"] & ring).sum())

        # Three vertical strips of equal area, so no strip can reach IoU > 0.5.
        _ys, _xs = np.nonzero(largest["mask"])
        _order = np.argsort(_xs, kind="stable")
        strips = []
        for _part in np.array_split(_order, 3):
            _m = np.zeros_like(largest["mask"])
            _m[_ys[_part], _xs[_part]] = True
            strips.append(dict(largest, mask=_m))

        return {
            "mock_perfect": base,
            "mock_shifted": [dict(p, mask=shift(p["mask"])) for p in base],
            "mock_merged": [p for p in rest if p is not neighbour]
            + [dict(largest, mask=largest["mask"] | neighbour["mask"])],
            "mock_split": rest + strips,
            "mock_duplicated": base
            + [dict(p, mask=dilate(p["mask"], 5), score=0.9) for p in things],
            "mock_dropped": rest,
        }
    return gt_segments, mock_predictions


@app.cell
def _(
    PRED_DIR,
    Path,
    USE_MOCKS,
    gt_segments,
    mock_predictions,
    name_to_cat,
    np,
    samples,
):
    predictions = {}  # model -> image_id -> list of prediction dicts

    if USE_MOCKS:
        for _id, _s in samples.items():
            for _model, _preds in mock_predictions(gt_segments(_s)).items():
                predictions.setdefault(_model, {})[_id] = _preds

    _unmapped = set()
    _root = Path(PRED_DIR)
    for _model_dir in sorted(_root.iterdir()) if _root.exists() else []:
        for _f in sorted(_model_dir.glob("*.npz")):
            _d = np.load(_f)
            _preds = []
            for _m, _label, _score in zip(_d["masks"], _d["labels"], _d["scores"]):
                _cat = name_to_cat.get(str(_label), -1)
                if _cat == -1:
                    _unmapped.add(str(_label))
                _preds.append({"mask": _m, "category_id": _cat, "score": float(_score)})
            predictions.setdefault(_model_dir.name, {})[int(_f.stem)] = _preds

    if _unmapped:
        print(f"WARNING: labels with no COCO panoptic category: {sorted(_unmapped)}")
    print("Models:", ", ".join(predictions))
    return (predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Scoring

    Both metrics are computed at two scopes:

    - `all`: things and stuff. This is the natural scope for a panoptic model.
    - `things`: stuff ground truth becomes background and stuff predictions are dropped.
      This is the fair scope for an instance model like YOLO, which never predicts stuff.

    At the `all` scope the only background is the void, a small fraction of the image. Excess
    is normalised by background area, so it swings sharply there.
    """)
    return


@app.cell
def _(defaultdict, np):
    def panoptic_quality(gt_segs, pred_segs, void):
        """COCO panoptic quality for one image. Returns (PQ, SQ, RQ)."""
        taken = np.zeros(void.shape, dtype=bool)
        preds = []
        for p in sorted(pred_segs, key=lambda p: -p["score"]):
            m = p["mask"] & ~taken
            if m.sum() == 0 or m.sum() < 0.5 * p["mask"].sum():
                continue
            taken |= m
            preds.append(dict(p, mask=m))

        stats = defaultdict(lambda: {"iou": 0.0, "tp": 0, "fp": 0, "fn": 0})
        crowd = defaultdict(lambda: np.zeros(void.shape, dtype=bool))
        matched = set()
        for g in gt_segs:
            if g["iscrowd"]:
                crowd[g["category_id"]] |= g["mask"]
                continue
            for pi, p in enumerate(preds):
                if pi in matched or p["category_id"] != g["category_id"]:
                    continue
                inter = (g["mask"] & p["mask"]).sum()
                union = g["mask"].sum() + p["mask"].sum() - inter - (p["mask"] & void).sum()
                iou = inter / union if union else 0.0
                if iou > 0.5:
                    stats[g["category_id"]]["tp"] += 1
                    stats[g["category_id"]]["iou"] += iou
                    matched.add(pi)
                    break
            else:
                stats[g["category_id"]]["fn"] += 1

        for pi, p in enumerate(preds):
            if pi in matched:
                continue
            ignored = (p["mask"] & void).sum() + (p["mask"] & crowd[p["category_id"]]).sum()
            if ignored / p["mask"].sum() > 0.5:
                continue
            stats[p["category_id"]]["fp"] += 1

        pq, sq, rq = [], [], []
        for st in stats.values():
            denom = st["tp"] + 0.5 * st["fp"] + 0.5 * st["fn"]
            if denom == 0:
                continue
            pq.append(st["iou"] / denom)
            sq.append(st["iou"] / st["tp"] if st["tp"] else 0.0)
            rq.append(st["tp"] / denom)
        if not pq:
            return 1.0, 1.0, 1.0
        return float(np.mean(pq)), float(np.mean(sq)), float(np.mean(rq))
    return (panoptic_quality,)


@app.cell
def _(
    MaskInstance,
    categories,
    cote_score,
    gt_segments,
    np,
    panoptic_quality,
    pd,
    predictions,
    samples,
):
    SCOPES = ("all", "things")

    def in_scope(cat_id, scope):
        if scope == "all":
            return True
        return cat_id in categories and bool(categories[cat_id]["isthing"])

    def scoped_inputs(sample, preds, scope):
        """GT SSU map, GT segments and predictions restricted to one scope."""
        gt = [g for g in gt_segments(sample) if in_scope(g["category_id"], scope)]
        gt_map = np.zeros(sample["pan_map"].shape, dtype=np.int32)
        for _ssu, g in enumerate(gt, start=1):
            gt_map[g["mask"]] = _ssu
        return gt_map, gt, [p for p in preds if in_scope(p["category_id"], scope)]

    _rows = []
    for _model, _by_image in predictions.items():
        for _id, _preds in _by_image.items():
            _s = samples[_id]
            _void = _s["pan_map"] == 0
            for _scope in SCOPES:
                _gt_map, _gt, _p = scoped_inputs(_s, _preds, _scope)
                _masks = [MaskInstance(mask=p["mask"], pred_id=i) for i, p in enumerate(_p)]
                if not _masks:
                    _masks = [MaskInstance(mask=np.zeros_like(_void), pred_id=0)]
                _cote, _c, _o, _t, _e = cote_score(_gt_map, _masks)
                _pq, _sq, _rq = panoptic_quality(_gt, _p, _void)
                _rows.append(
                    {
                        "model": _model,
                        "image_id": _id,
                        "scope": _scope,
                        "cote": _cote,
                        "coverage": _c,
                        "overlap": _o,
                        "trespass": _t,
                        "excess": _e,
                        "pq": _pq,
                        "sq": _sq,
                        "rq": _rq,
                        "n_gt": len(_gt),
                        "n_pred": len(_p),
                    }
                )
    scores_df = pd.DataFrame(_rows)
    scores_df.round(3)
    return SCOPES, scoped_inputs, scores_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### COTe against PQ

    One point per image. Points above the dashed line are cases where COTe rates the
    prediction more highly than PQ, typically a granularity disagreement. Points below it are
    cases where COTe is penalising overlap or trespass that PQ cannot see.
    """)
    return


@app.cell
def _(SCOPES, mo):
    scope_picker = mo.ui.dropdown(options=list(SCOPES), value="all", label="Scope")
    scope_picker
    return (scope_picker,)


@app.cell
def _(plt, scope_picker, scores_df):
    _df = scores_df[scores_df.scope == scope_picker.value]
    _models = list(dict.fromkeys(scores_df.model))  # fixed order, so colours never shift
    _markers = dict(zip(sorted(_df.image_id.unique()), ["o", "s", "^", "D"]))
    _cmap = plt.get_cmap("tab10")

    _fig, _ax = plt.subplots(figsize=(7, 7))
    for _i, _model in enumerate(_models):
        for _id, _grp in _df[_df.model == _model].groupby("image_id"):
            _ax.scatter(
                _grp.pq,
                _grp.cote,
                color=_cmap(_i % 10),
                marker=_markers[_id],
                s=70,
                edgecolor="white",
                linewidth=1,
                label=_model if _id == min(_markers) else None,
            )
    for _id, _mk in _markers.items():
        _ax.scatter([], [], marker=_mk, color="grey", label=f"image {_id}")
    #_ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    #_ax.set_xlim(-0.05, 1.05)
    _ax.set_xlabel("PQ")
    _ax.set_ylabel("COTe")
    _ax.set_title(f"COTe vs PQ, scope = {scope_picker.value}")
    _ax.grid(alpha=0.3)
    _ax.legend(fontsize=8, loc="lower right")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Where the pixels went

    Pick a model, an image and a scope. The left panel shows the ground truth at that scope,
    the middle panel shows the predictions coloured by class, and the right panel shows the
    COTe pixel states:

    | colour | state |
    |---|---|
    | green | coverage |
    | amber | overlap |
    | red | trespass |
    | purple | overlap + trespass |
    | grey | missing |
    | blue | excess |
    """)
    return


@app.cell
def _(SCOPES, mo, predictions, samples):
    model_picker = mo.ui.dropdown(
        options=list(predictions), value=next(iter(predictions)), label="Model"
    )
    image_picker = mo.ui.dropdown(
        options=[str(i) for i in samples], value=str(next(iter(samples))), label="Image"
    )
    view_scope_picker = mo.ui.dropdown(options=list(SCOPES), value="all", label="Scope")
    mo.hstack([model_picker, image_picker, view_scope_picker], justify="start")
    return image_picker, model_picker, view_scope_picker


@app.cell
def _(
    MaskInstance,
    cat_colour,
    cat_name,
    compute_cote_masks,
    image_picker,
    label_segments,
    model_picker,
    np,
    overlay_masks,
    plt,
    predictions,
    samples,
    scoped_inputs,
    scores_df,
    view_scope_picker,
    visualize_cote_states,
):
    _id = int(image_picker.value)
    _model = model_picker.value
    _scope = view_scope_picker.value
    _s = samples[_id]
    _gt_map, _gt, _p = scoped_inputs(_s, predictions[_model].get(_id, []), _scope)
    _row = scores_df[
        (scores_df.model == _model) & (scores_df.image_id == _id) & (scores_df.scope == _scope)
    ].iloc[0]

    _fig, (_ax_gt, _ax_pred, _ax_cote) = plt.subplots(1, 3, figsize=(20, 9))

    _gt_masks = [g["mask"] for g in _gt]
    _ax_gt.imshow(
        overlay_masks(_s["image"], _gt_masks, [cat_colour(g["category_id"]) for g in _gt])
    )
    label_segments(_ax_gt, _gt_masks, [cat_name(g["category_id"]) for g in _gt])
    _ax_gt.set_title(f"GT ({_scope}): {len(_gt)} segments")

    _pred_masks = [p["mask"] for p in _p]
    _ax_pred.imshow(
        overlay_masks(_s["image"], _pred_masks, [cat_colour(p["category_id"]) for p in _p])
    )
    label_segments(_ax_pred, _pred_masks, [cat_name(p["category_id"]) for p in _p])
    _ax_pred.set_title(f"{_model}: {len(_p)} segments")

    _masks = [MaskInstance(mask=p["mask"], pred_id=i) for i, p in enumerate(_p)] or [
        MaskInstance(mask=np.zeros(_gt_map.shape, dtype=bool), pred_id=0)
    ]
    _patches = visualize_cote_states(
        _s["image"], compute_cote_masks(_gt_map, _masks), ax=_ax_cote, show_missing=True
    )
    _ax_cote.legend(handles=_patches, loc="upper right", fontsize=8, framealpha=0.9)
    _ax_cote.set_title(
        f"COTe {_row.cote:.3f} (C {_row.coverage:.2f}, O {_row.overlap:.2f}, "
        f"T {_row.trespass:.2f}, E {_row.excess:.2f})\n"
        f"PQ {_row.pq:.3f} (SQ {_row.sq:.2f}, RQ {_row.rq:.2f})"
    )
    for _ax in (_ax_gt, _ax_pred, _ax_cote):
        _ax.axis("off")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
