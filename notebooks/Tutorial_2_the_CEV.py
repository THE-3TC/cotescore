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
    # The CEV on NCSE: is OCR or parsing the bottleneck?

    This is the second of two notebooks. It assumes you have run **`ncse_cote_tutorial.py`**,
    which wrote layout predictions and per-page COTe scores to `outputs/predictions/`.

    ## The problem

    You run a document pipeline and the transcription comes out bad. Two things could be at
    fault, and the final text looks the same either way:

    1. **Parsing** — the layout model carved the page up wrongly, so the OCR engine was handed
       crops that were cut off, merged, or missed entirely.
    2. **OCR** — the regions were fine, but the engine misread the characters.

    Character Error Rate cannot separate these. CER needs an alignment between predicted and
    reference text, and when parsing is wrong the reading order is wrong too, so the alignment
    itself becomes meaningless — CER collapses precisely when you most need a diagnosis.

    ## The Character Error Vector

    The CEV drops alignment entirely and compares **bags of characters**. Four distributions:

    | symbol | key | what it is |
    |---|---|---|
    | $Q$ | `gt` | all ground-truth characters on the page |
    | $R$ | `parsing` | ground-truth characters that fall inside **predicted** regions |
    | $S^*$ | `ocr` | OCR run on **ground-truth** regions |
    | $S$ | `total` | OCR run on **predicted** regions |

    Four comparisons follow, each isolating one stage:

    $$d_{pars} = m(R, Q) \qquad d_{ocr} = m(S^*, Q) \qquad d_{int} = m(S, R) \qquad d_{total} = m(S, Q)$$

    - $d_{ocr}$ is the error the OCR engine makes when handed **perfect** regions.
    - $d_{total}$ is the error of the **whole pipeline**.
    - $d_{pars}$ is the error the parser introduces on its own.

    Two instantiations share this structure: **SpACER**, a count-based metric analogous to CER
    ($(D + \hat{E}) / 2C$), and **CDD**, a distribution-based metric using the square root of the
    Jensen–Shannon divergence, bounded in $[0, 1]$.

    ## What NCSE can and cannot give us

    Building $R$ means asking *which predicted region does each ground-truth character fall
    inside* — a spatial join of per-character midpoints against predicted boxes. The NCSE ground
    truth has the text of each region but **no character-level positions**, so $R$ cannot be
    built.

    That rules out $d_{pars}$ and $d_{int}$. `spacer_decomp` and `cdd_decomp` return `None` for
    any component whose inputs are absent, so this is visible rather than silent. We are left
    with:

    $$d_{ocr} \quad \text{and} \quad d_{total}$$

    This is less of a loss than it sounds, and it is the point of this notebook. Those two terms
    are exactly what the triage rule needs, and both are cheap — no character positions, just
    page text plus two OCR runs. **If you had character boxes** (as the Spiritualist, HierText and
    DocBank experiments do) you would call `spacer_decomp_spatial` / `cdd_decomp_spatial` instead
    and get the full four-component vector.
    """
    )
    return


@app.cell
def _():
    # --- Point these at your data (same as notebook 1) ----------------------------
    NCSE_IMAGES_DIR = "/teamspace/studios/this_studio/ncse/images"
    NCSE_GT_CSV = "/teamspace/studios/this_studio/ncse/ncse_testset_bboxes.csv"

    # --- Written by notebook 1 ----------------------------------------------------
    PRED_DIR = "outputs/predictions"

    # --- Written by this notebook -------------------------------------------------
    CEV_DIR = "outputs/cev"

    LAYOUT_MODELS = ["heron", "yolo"]
    OCR_MODELS = ["easyocr", "tesseract"]
    USE_GPU = True

    # Triage thresholds (section 5)
    COTE_THRESHOLD = 0.5
    RATIO_THRESHOLD = 0.5
    return (
        CEV_DIR,
        COTE_THRESHOLD,
        LAYOUT_MODELS,
        NCSE_GT_CSV,
        NCSE_IMAGES_DIR,
        OCR_MODELS,
        PRED_DIR,
        RATIO_THRESHOLD,
        USE_GPU,
    )


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from PIL import Image

    from cotescore.ocr import cdd_decomp, spacer_decomp

    return Image, Path, cdd_decomp, np, pd, plt, spacer_decomp


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Load ground truth and predictions

    Two sources:

    - **Ground-truth regions with their text** come from the GT CSV. Its `x1, y1, x2, y2` may
      have been recorded at a different resolution than the images on disk, so we rescale by
      `image_width / actual_width` — the same correction `NCSEDataset` applies internally.
    - **Predicted regions** come from notebook 1's flat CSVs, already in on-disk pixel
      coordinates.
    """
    )
    return


@app.cell
def _(Image, NCSE_GT_CSV, NCSE_IMAGES_DIR, Path, pd):
    _gt = pd.read_csv(NCSE_GT_CSV)
    _gt["text"] = _gt["text"].fillna("")

    # Rescale GT boxes to on-disk image pixels, and record each page's true size.
    _sizes = {}
    for _fn in _gt["filename"].unique():
        _p = Path(NCSE_IMAGES_DIR) / _fn
        if _p.exists():
            with Image.open(_p) as _im:
                _sizes[_fn] = _im.size

    _gt = _gt[_gt["filename"].isin(_sizes)].copy()
    _gt["img_w"] = _gt["filename"].map(lambda f: _sizes[f][0])
    _gt["img_h"] = _gt["filename"].map(lambda f: _sizes[f][1])
    _sx = _gt["img_w"] / _gt["image_width"]
    _sy = _gt["img_h"] / _gt["image_height"]
    gt_df = _gt.assign(
        x=_gt["x1"] * _sx,
        y=_gt["y1"] * _sy,
        width=(_gt["x2"] - _gt["x1"]) * _sx,
        height=(_gt["y2"] - _gt["y1"]) * _sy,
    )[
        [
            "filename",
            "x",
            "y",
            "width",
            "height",
            "ssu_id",
            "text",
            "img_w",
            "img_h",
        ]
    ]

    pages = sorted(gt_df["filename"].unique())
    print(f"{len(pages)} pages, {len(gt_df)} ground-truth regions")
    print(f"Mean GT characters per page: {gt_df.groupby('filename').text.apply(lambda s: sum(map(len, s))).mean():.0f}")
    return gt_df, pages


@app.cell
def _(LAYOUT_MODELS, PRED_DIR, pd):
    pred_df = {}
    cote_df = {}
    for _m in LAYOUT_MODELS:
        _p = pd.read_csv(f"{PRED_DIR}/ncse_{_m}_predictions.csv")
        pred_df[_m] = _p[_p["source"] == "pred"].reset_index(drop=True)
        cote_df[_m] = pd.read_csv(f"{PRED_DIR}/ncse_{_m}_scores.csv")
        print(
            f"{_m}: {len(pred_df[_m])} predicted regions across "
            f"{pred_df[_m].filename.nunique()} pages"
        )
    return cote_df, pred_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. OCR engines

    Two small adapters with a `run(crop) -> str` contract, mirroring the `OCRModel` base class in
    the SpACER repository. They are inlined here so this notebook stands alone; SpACER's
    `ocr_models/` package has the same interface plus PaddleOCR and TrOCR backends if you want to
    extend the comparison.

    - **EasyOCR** — neural detector + recogniser, benefits from a GPU.
    - **Tesseract** — classical engine. `--psm 6` tells it to treat each crop as a single uniform
      block of text, which suits region crops.
    """
    )
    return


@app.cell
def _(USE_GPU, np):
    class EasyOCREngine:
        name = "easyocr"

        def __init__(self, gpu=USE_GPU):
            import easyocr

            self.reader = easyocr.Reader(["en"], gpu=gpu)

        def run(self, crop):
            return " ".join(self.reader.readtext(np.array(crop), detail=0))

    class TesseractEngine:
        name = "tesseract"

        def __init__(self, psm=6):
            import pytesseract

            self.pytesseract = pytesseract
            self.config = f"--psm {psm}"

        def run(self, crop):
            return self.pytesseract.image_to_string(crop, config=self.config)

    def build_engine(name):
        return {"easyocr": EasyOCREngine, "tesseract": TesseractEngine}[name]()

    return (build_engine,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Crop and transcribe

    Six passes in total:

    - **$S^*$ — 2 passes.** OCR of the **ground-truth** crops, once per engine. This depends only
      on the OCR engine, so it is computed once and reused by both layout models. That is why
      $d_{ocr}$ will be *identical* down each engine's pair of rows in the results table — a
      useful check that the decomposition really is isolating the OCR stage.
    - **$S$ — 4 passes.** OCR of the **predicted** crops, one per (layout model × engine).

    Results are cached to CSV, so re-running the notebook does not re-transcribe.
    """
    )
    return


@app.cell
def _(CEV_DIR, Image, NCSE_IMAGES_DIR, Path, pd):
    def crop_regions(image, boxes):
        """Crop [x, y, width, height] boxes, clamped to the image. Degenerate boxes -> None."""
        W, H = image.size
        crops = []
        for x, y, w, h in boxes:
            x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
            x1, y1 = min(W, int(round(x + w))), min(H, int(round(y + h)))
            crops.append(image.crop((x0, y0, x1, y1)) if x1 - x0 >= 2 and y1 - y0 >= 2 else None)
        return crops

    OCR_COLUMNS = ["filename", "region_idx", "ocr_text"]

    def ocr_regions(engine, region_df, tag, pages):
        """OCR every region in region_df. Cached to CEV_DIR/ocr_<tag>.csv."""
        cache = Path(CEV_DIR) / f"ocr_{tag}.csv"
        if cache.exists():
            print(f"  [cached] {cache}")
            cached = pd.read_csv(cache)
            # A model that predicted nothing leaves a header-only file; keep the schema
            # so downstream lookups by column name still work.
            if cached.empty:
                cached = pd.DataFrame(columns=OCR_COLUMNS)
            return cached.fillna({"ocr_text": ""})

        cache.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for n, page in enumerate(pages, 1):
            sub = region_df[region_df["filename"] == page]
            if sub.empty:
                continue
            with Image.open(Path(NCSE_IMAGES_DIR) / page) as im:
                img = im.convert("RGB")
                crops = crop_regions(
                    img, sub[["x", "y", "width", "height"]].to_numpy(float)
                )
            for (_, r), crop in zip(sub.iterrows(), crops):
                rows.append(
                    {
                        "filename": page,
                        "region_idx": int(r.name),
                        "ocr_text": "" if crop is None else engine.run(crop),
                    }
                )
            print(f"  {tag}: {n}/{len(pages)} pages", end="\r")

        # columns= keeps the schema when a model produced no regions at all.
        out = pd.DataFrame(rows, columns=OCR_COLUMNS)
        out.to_csv(cache, index=False)
        print(f"  {tag}: {len(out)} regions -> {cache}")
        if out.empty:
            print(f"  WARNING: {tag} produced no regions on these pages.")
        return out.fillna({"ocr_text": ""})

    return (ocr_regions,)


@app.cell
def _(
    LAYOUT_MODELS,
    OCR_MODELS,
    build_engine,
    gt_df,
    ocr_regions,
    pages,
    pred_df,
):
    ocr_gt = {}  # S*  : ocr_model            -> DataFrame
    ocr_pred = {}  # S  : (layout, ocr_model) -> DataFrame

    for _ocr_name in OCR_MODELS:
        print(f"Loading {_ocr_name} ...")
        _engine = build_engine(_ocr_name)

        # S* — OCR of ground-truth regions (shared across layout models)
        ocr_gt[_ocr_name] = ocr_regions(_engine, gt_df, f"gt_{_ocr_name}", pages)

        # S — OCR of predicted regions, per layout model
        for _lm in LAYOUT_MODELS:
            ocr_pred[(_lm, _ocr_name)] = ocr_regions(
                _engine, pred_df[_lm], f"{_lm}_{_ocr_name}", pages
            )

        del _engine
    print("OCR complete.")
    return ocr_gt, ocr_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Score

    For each page we build three character bags and hand the same dict to both metrics:

    ```python
    bags = {"gt": gt_text, "ocr": ocr_of_gt_regions, "total": ocr_of_predicted_regions}
    spacer_decomp(bags)   # -> d_ocr_macro, d_total_macro   (d_pars, d_int are None)
    cdd_decomp(bags)      # -> d_ocr, d_total               (d_pars, d_int are None)
    ```

    **Pass page-level strings, not per-box lists.** `spacer_decomp` compares the reference and
    prediction box lists pairwise with `zip`, so a 12-box ground truth scored against a 40-box
    prediction would be silently truncated to the first 12. There is no meaningful pairing
    between ground-truth and predicted regions anyway — that is exactly what is unknown without
    character positions — so the page is the right unit. With single strings the macro and micro
    variants coincide, and we report the macro values.
    """
    )
    return


@app.cell
def _(
    LAYOUT_MODELS,
    OCR_MODELS,
    cdd_decomp,
    gt_df,
    ocr_gt,
    ocr_pred,
    pages,
    pd,
    spacer_decomp,
):
    def join_text(df, page, col="ocr_text"):
        if col not in df.columns or "filename" not in df.columns:
            return ""
        return " ".join(df.loc[df["filename"] == page, col].fillna("").astype(str))

    _rows = []
    for _page in pages:
        _q = join_text(gt_df, _page, "text")  # Q
        if not _q:
            continue
        for _ocr_name in OCR_MODELS:
            _s_star = join_text(ocr_gt[_ocr_name], _page)  # S*
            for _lm in LAYOUT_MODELS:
                _s = join_text(ocr_pred[(_lm, _ocr_name)], _page)  # S
                _bags = {"gt": _q, "ocr": _s_star, "total": _s}
                _sp = spacer_decomp(_bags)
                _cd = cdd_decomp(_bags)
                _rows.append(
                    {
                        "filename": _page,
                        "layout_model": _lm,
                        "ocr_model": _ocr_name,
                        "n_gt_chars": len(_q),
                        "spacer_d_ocr": _sp.d_ocr_macro,
                        "spacer_d_total": _sp.d_total_macro,
                        "spacer_d_pars": _sp.d_pars_macro,
                        "spacer_d_int": _sp.d_int_macro,
                        "cdd_d_ocr": _cd.d_ocr,
                        "cdd_d_total": _cd.d_total,
                        "cdd_d_pars": _cd.d_pars,
                        "cdd_d_int": _cd.d_int,
                    }
                )

    cev_df = pd.DataFrame(_rows)
    print(f"{len(cev_df)} page x layout x ocr rows")
    print(
        "d_pars / d_int all None (no character positions): "
        f"{cev_df.spacer_d_pars.isna().all()} / {cev_df.spacer_d_int.isna().all()}"
    )
    cev_df.head()
    return (cev_df,)


@app.cell
def _(CEV_DIR, Path, cev_df):
    Path(CEV_DIR).mkdir(parents=True, exist_ok=True)
    cev_df.to_csv(f"{CEV_DIR}/ncse_cev_scores.csv", index=False)
    print(f"Wrote {CEV_DIR}/ncse_cev_scores.csv")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### Results

    One row per (layout model, OCR engine), averaged over pages.

    Two things to look for:

    1. **`d_ocr` repeats down each engine's pair of rows.** It is measured on ground-truth
       regions, so the layout model cannot influence it. Seeing it hold constant confirms the
       decomposition is isolating the stages rather than smearing them together.
    2. **The gap between `d_ocr` and `d_total`** is everything the parsing step added. A small gap
       means the layout model handed over essentially the same characters the ground truth would
       have; a large gap means the parser lost or duplicated text before OCR ever ran. The gap can
       come out slightly *negative*: these are bags of characters, not aligned strings, so a
       predicted region that happens to crop more cleanly than the ground-truth box can transcribe
       marginally better. Treat a small negative gap as "parsing cost nothing here", not as an
       error.

    SpACER is count-based and unbounded above; CDD is a distribution distance bounded in
    $[0, 1]$. They rank the same way but are not on the same scale — compare within a table, not
    across.
    """
    )
    return


@app.cell
def _(cev_df):
    spacer_table = (
        cev_df.groupby(["layout_model", "ocr_model"])[["spacer_d_ocr", "spacer_d_total"]]
        .mean()
        .rename(columns={"spacer_d_ocr": "d_ocr", "spacer_d_total": "d_total"})
        .round(4)
    )
    spacer_table["gap (d_total - d_ocr)"] = (
        spacer_table["d_total"] - spacer_table["d_ocr"]
    ).round(4)
    spacer_table
    return (spacer_table,)


@app.cell
def _(cev_df):
    cdd_table = (
        cev_df.groupby(["layout_model", "ocr_model"])[["cdd_d_ocr", "cdd_d_total"]]
        .mean()
        .rename(columns={"cdd_d_ocr": "d_ocr", "cdd_d_total": "d_total"})
        .round(4)
    )
    cdd_table["gap (d_total - d_ocr)"] = (
        cdd_table["d_total"] - cdd_table["d_ocr"]
    ).round(4)
    cdd_table
    return (cdd_table,)


@app.cell
def _(cdd_table, spacer_table):
    print("SpACER (count-based, unbounded)\n")
    print(spacer_table.to_string())
    print("\n\nCDD (sqrt-JSD, bounded [0, 1])\n")
    print(cdd_table.to_string())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Which stage is the bottleneck?

    Now combine the two notebooks. We have, per page:

    - **COTe** — how good the geometry is (notebook 1).
    - **$d_{ocr} / d_{total}$** — what share of the pipeline's total error was already present
      when OCR ran on *perfect* regions.

    Neither term needs character-level ground truth, which is the whole point: this diagnosis is
    available on datasets like NCSE where the full CEV is not.

    ```python
    ocr_is_bottleneck = (cote > 0.5) & (d_ocr / d_total > 0.5)
    ```

    Reading the two terms:

    - **High COTe** — the regions are geometrically sound, so we can trust that the OCR engine
      was given a fair chance.
    - **High ratio** — most of the error survives even with perfect regions, so it is the
      transcription engine at fault.

    Fail either test and the finger points at parsing: either the geometry is visibly wrong, or
    the pipeline error is much larger than what OCR alone produces, meaning the region carve-up
    introduced it.

    A caveat worth stating plainly. **We cannot validate this rule on NCSE.** Validation means
    checking the prediction against the true label $d_{ocr} \geq d_{pars}$, and $d_{pars}$ is
    exactly the term the missing character positions deny us. The 0.5/0.5 defaults come from a
    threshold sweep on the Spiritualist dataset, where character boxes make the ground-truth
    label computable — see `cote_conditioned_validation.py` and the F1 heatmap in
    `spiritualist_decomposition.py` in the SpACER repository. Here we are *applying* a
    pre-validated rule, not establishing one.
    """
    )
    return


@app.cell
def _(COTE_THRESHOLD, LAYOUT_MODELS, RATIO_THRESHOLD, cev_df, cote_df, pd):
    _cote = pd.concat(
        [cote_df[_m][["filename", "cote"]].assign(layout_model=_m) for _m in LAYOUT_MODELS],
        ignore_index=True,
    )
    triage_df = cev_df.merge(_cote, on=["filename", "layout_model"], how="left")

    triage_df["ocr_over_total"] = (
        triage_df["spacer_d_ocr"] / triage_df["spacer_d_total"]
    ).replace([float("inf"), -float("inf")], float("nan"))

    triage_df["ocr_is_bottleneck"] = (triage_df["cote"] > COTE_THRESHOLD) & (
        triage_df["ocr_over_total"] > RATIO_THRESHOLD
    )
    triage_df["verdict"] = triage_df["ocr_is_bottleneck"].map(
        {True: "OCR", False: "parsing"}
    )
    print(f"Rule: cote > {COTE_THRESHOLD} AND d_ocr/d_total > {RATIO_THRESHOLD}\n")
    print(
        triage_df.groupby(["layout_model", "ocr_model"])["verdict"]
        .value_counts()
        .unstack(fill_value=0)
        .to_string()
    )
    return (triage_df,)


@app.cell
def _(COTE_THRESHOLD, RATIO_THRESHOLD, plt, triage_df):
    _combos = sorted(triage_df.groupby(["layout_model", "ocr_model"]).groups)
    _fig, _axes = plt.subplots(
        1, len(_combos), figsize=(4.2 * len(_combos), 4.4), sharex=True, sharey=True
    )
    _axes = [_axes] if len(_combos) == 1 else list(_axes)

    for _ax, (_lm, _om) in zip(_axes, _combos):
        _sub = triage_df[
            (triage_df.layout_model == _lm) & (triage_df.ocr_model == _om)
        ]
        for _verdict, _colour in [("OCR", "tab:red"), ("parsing", "tab:blue")]:
            _s = _sub[_sub.verdict == _verdict]
            _ax.scatter(
                _s["cote"], _s["ocr_over_total"], s=42, alpha=0.8,
                c=_colour, label=f"{_verdict} ({len(_s)})",
            )
        _ax.axvline(COTE_THRESHOLD, color="k", ls="--", lw=1)
        _ax.axhline(RATIO_THRESHOLD, color="k", ls="--", lw=1)
        _ax.set_title(f"{_lm} + {_om}")
        _ax.set_xlabel("COTe")
        _ax.legend(fontsize=8, loc="lower left")
    _axes[0].set_ylabel("SpACER  $d_{ocr}$ / $d_{total}$")
    _fig.suptitle(
        "Bottleneck triage: top-right quadrant = OCR-limited, elsewhere = parsing-limited"
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(CEV_DIR, triage_df):
    triage_df.to_csv(f"{CEV_DIR}/ncse_triage.csv", index=False)
    print(f"Wrote {CEV_DIR}/ncse_triage.csv")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Summary

    Across the two notebooks we evaluated a two-stage document pipeline, keeping the stages
    separable throughout:

    - **COTe** scored the layout stage on pixels rather than box matching, so it stayed
      meaningful even though the models and the ground truth disagree about how a page divides
      into regions.
    - **The CEV** scored the OCR stage on character bags rather than aligned strings, so it stayed
      meaningful even when parsing errors destroyed the reading order.
    - Combining $COTe$ with $d_{ocr}/d_{total}$ attributed each page's error to a stage — using
      only quantities that a dataset without character positions can supply.

    **To go further.** If your ground truth includes character boxes, use `spacer_decomp_spatial`
    and `cdd_decomp_spatial` instead of the dict interface. They build $R$ by joining character
    midpoints against predicted regions, which recovers $d_{pars}$ and $d_{int}$ — the full
    four-component vector, and the ground-truth label needed to validate a triage rule rather
    than merely apply one.

    Outputs from this notebook, in `outputs/cev/`:

    - `ocr_*.csv` — cached transcriptions, one file per OCR pass.
    - `ncse_cev_scores.csv` — per page × layout model × OCR engine CEV components.
    - `ncse_triage.csv` — the above plus COTe, the ratio, and the verdict.
    """
    )
    return


if __name__ == "__main__":
    app.run()
