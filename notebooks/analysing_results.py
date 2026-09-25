import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import json
    import marimo as mo
    import patchworklib as pw
    from plotnine import (
        aes,
        geom_hline,
        geom_line,
        ggplot,
        labs,
        theme,
        theme_minimal,
        geom_vline,
        geom_point,
        scale_x_log10,
        facet_wrap,
        element_text,
        element_blank,
    )
    return Path, aes, geom_point, ggplot, json, mo, pd


@app.cell
def _():
    return


@app.cell
def _(Path):
    results_folder = Path("data/results")
    ppdoc_results_folder = results_folder / "ppdoc"
    torch_results_folder = results_folder / "torch"

    doclaynet_json = "doclaynet/benchmark_all_results.json"
    HNLA_json = "hnla2013/benchmark_all_results.json"
    NCSE_json = "ncse/benchmark_all_results.json"
    HNLA_nosssu_json = "hnla2013_nossu/benchmark_all_results.json"
    NCSE_nossu_json = "ncse_nossu/benchmark_all_results.json"
    return (
        HNLA_json,
        HNLA_nosssu_json,
        NCSE_json,
        NCSE_nossu_json,
        doclaynet_json,
        ppdoc_results_folder,
        torch_results_folder,
    )


@app.cell
def _(json, pd, ppdoc_results_folder, torch_results_folder):

    def create_results_table(filename):
        MODEL_NAME_MAP = {
            "DocLayout-YOLO": "YOLO",
            "DoclingLayoutHeron": "Heron",
            "PPDocLayout-L": "PPDoc-L",
            "PPDocLayout-M": "PPDoc-M",
            "PPDocLayout-S": "PPDoc-S",
        }

        out_df = []
        for _folder in [ppdoc_results_folder, torch_results_folder]:
            with open(_folder / filename, "r") as f:
                _data = json.load(f)
            _df = pd.DataFrame.from_dict(_data["models"], orient="index")
            out_df.append(_df)
        out_df = pd.concat(out_df, ignore_index=False)
        out_df = out_df.loc[
            :, ["cot_score", "coverage", "overlap", "trespass", "excess", "mean_iou", "map"]
        ]
        out_df.index = out_df.index.map(lambda x: MODEL_NAME_MAP.get(x, x))
        out_df = out_df.sort_index()
        return out_df

    def df_to_latex_md(df, caption, label):
        col_rename = {
            "cot_score": "COTe",
            "coverage": "Coverage",
            "overlap": "Overlap",
            "trespass": "Trespass",
            "excess": "Excess",
            "mean_iou": "IoU",
            "map": "mAP",
        }

        col_best = {
            "cot_score": "high",
            "coverage": "high",
            "overlap": "low",
            "trespass": "low",
            "excess": "low",
            "mean_iou": "high",
            "map": "high",
        }

        df_fmt = df.copy().astype(float)

        for col, direction in col_best.items():
            if col in df_fmt.columns:
                best_val = df_fmt[col].max() if direction == "high" else df_fmt[col].min()
                df_fmt[col] = df_fmt[col].apply(
                    lambda x: f"\\textbf{{{x:.2f}}}" if x == best_val else f"{x:.2f}"
                )

        latex = df_fmt.rename(columns=col_rename).to_latex(
            index=True, escape=False, caption=caption, label=label
        )

        return f"### LaTeX Table Output\n```latex\n{latex}\n```"
    return create_results_table, df_to_latex_md


@app.cell
def _(
    HNLA_json,
    HNLA_nosssu_json,
    NCSE_json,
    NCSE_nossu_json,
    create_results_table,
    doclaynet_json,
):
    doclaynet_df = create_results_table(doclaynet_json)
    HNLA2013_df = create_results_table(HNLA_json)
    NCSE_df = create_results_table(NCSE_json)

    HNLA2013_nossu_df = create_results_table(HNLA_nosssu_json)
    NCSE_nossu_df = create_results_table(NCSE_nossu_json)
    return HNLA2013_df, HNLA2013_nossu_df, NCSE_df, NCSE_nossu_df, doclaynet_df


@app.cell
def _(HNLA2013_nossu_df):
    HNLA2013_nossu_df
    return


@app.cell
def _(NCSE_nossu_df):
    NCSE_nossu_df
    return


@app.cell
def _(HNLA2013_df, NCSE_df, doclaynet_df, pd):
    all_datasets = pd.concat(
        [
            doclaynet_df.assign(dataset="doclaynet"),
            HNLA2013_df.assign(dataset="HNLA2013"),
            NCSE_df.assign(dataset="NCSE"),
        ],
        ignore_index=False,
    )
    all_datasets = all_datasets.reset_index()
    return (all_datasets,)


@app.cell
def _(aes, all_datasets, geom_point, ggplot):
    (
        ggplot(all_datasets, aes(x="cot_score", y="mean_iou", colour="index", shape="dataset"))
        + geom_point()
    )
    return


@app.cell
def _(doclaynet_df):
    doclaynet_df.index
    return


@app.cell
def _():
    return


@app.cell
def _(df_to_latex_md, doclaynet_df, mo):
    mo.md(
        df_to_latex_md(
            doclaynet_df, caption="Model results for DocLayNet", label="tab:res_doclaynet"
        )
    )
    return


@app.cell
def _(HNLA2013_df, df_to_latex_md, mo):
    mo.md(
        df_to_latex_md(HNLA2013_df, caption="Model results for HNLA2013", label="tab:res_HNLA2013")
    )
    return


@app.cell
def _(NCSE_df, df_to_latex_md, mo):
    mo.md(df_to_latex_md(NCSE_df, caption="Model results for NCSE", label="tab:res_ncse"))
    return


@app.cell
def _(Path, json, pd):

    def load_results_to_dataframe(base_path: str) -> pd.DataFrame:
        """
        Load all JSON result files from a directory and return a combined DataFrame.

        Each row contains the model name, filename, and per-image metrics.
        Files without a 'model' field are skipped.

        Args:
            base_path: Path to directory containing JSON result files

        Returns:
            DataFrame with columns: model, filename, mean_iou, coverage,
            overlap, trespass, excess, cot_score
        """
        records = []
        path = Path(base_path)

        for json_file in path.glob("*.json"):
            with open(json_file, "r") as f:
                data = json.load(f)

            model = data.get("model")
            if not model:
                continue

            for image_result in data.get("per_image_results", []):
                record = {
                    "model": model,
                    "filename": image_result.get("filename"),
                    **image_result.get("metrics", {}),
                }
                records.append(record)

        return pd.DataFrame(records)
    return (load_results_to_dataframe,)


@app.cell
def _(load_results_to_dataframe, pd):
    MODEL_NAME_MAP = {
        "juliozhao/DocLayout-YOLO-DocStructBench": "YOLO",
        "docling-project/docling-layout-heron": "Heron",
        "PP-DocLayout-L": "PPDoc-L",
        "PP-DocLayout-M": "PPDoc-M",
        "PP-DocLayout-S": "PPDoc-S",
    }

    NCSE_comparison_df = pd.concat(
        [
            load_results_to_dataframe("data/results/ppdoc/ncse"),
            load_results_to_dataframe("data/results/torch/ncse"),
        ]
    )

    NCSE_comparison_df["model"] = NCSE_comparison_df["model"].map(
        lambda x: MODEL_NAME_MAP.get(x, x)
    )

    HNLA_comparison_df = pd.concat(
        [
            load_results_to_dataframe("data/results/ppdoc/hnla2013"),
            load_results_to_dataframe("data/results/torch/hnla2013"),
        ]
    )

    HNLA_comparison_df["model"] = HNLA_comparison_df["model"].map(
        lambda x: MODEL_NAME_MAP.get(x, x)
    )

    doclaynet_comparison_df = pd.concat(
        [
            load_results_to_dataframe("data/results/ppdoc/doclaynet"),
            load_results_to_dataframe("data/results/torch/doclaynet"),
        ]
    )

    doclaynet_comparison_df["model"] = doclaynet_comparison_df["model"].map(
        lambda x: MODEL_NAME_MAP.get(x, x)
    )
    return (
        HNLA_comparison_df,
        MODEL_NAME_MAP,
        NCSE_comparison_df,
        doclaynet_comparison_df,
    )


@app.cell
def _(NCSE_comparison_df, aes, geom_point, ggplot):
    ggplot(NCSE_comparison_df, aes(y="coverage", x="overlap", color="model")) + geom_point()
    return


@app.cell
def _(NCSE_comparison_df, aes, geom_point, ggplot):
    ggplot(NCSE_comparison_df, aes(y="coverage", x="trespass", color="model")) + geom_point()
    return


@app.cell
def _(NCSE_comparison_df):
    NCSE_comparison_df.drop(columns=["filename", "model"]).corr()
    return


@app.cell
def _(NCSE_comparison_df):
    NCSE_comparison_df.drop(columns="filename").groupby("model").apply(
        lambda x: x.drop(columns="model").corr()
    )
    return


@app.cell
def _(NCSE_comparison_df):
    NCSE_comparison_df.drop(columns=["filename", "model"]).corr(method="spearman")
    return


@app.cell
def _(HNLA_comparison_df):
    HNLA_comparison_df.drop(columns=["filename", "model"]).corr(method="spearman")
    return


@app.cell
def _(doclaynet_comparison_df):
    doclaynet_comparison_df.drop(columns=["filename", "model"]).corr(method="spearman")
    return


@app.cell
def _(NCSE_comparison_df, aes, geom_point, ggplot):
    ggplot(NCSE_comparison_df, aes(x="cot_score", y="mean_iou", colour="model")) + geom_point()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # No ssu
    """)
    return


@app.cell
def _(MODEL_NAME_MAP, load_results_to_dataframe, pd):
    NCSE_nossu_comparison_df = pd.concat(
        [
            load_results_to_dataframe("data/results/ppdoc/ncse_nossu"),
            load_results_to_dataframe("data/results/torch/ncse_nossu"),
        ]
    )

    NCSE_nossu_comparison_df["model"] = NCSE_nossu_comparison_df["model"].map(
        lambda x: MODEL_NAME_MAP.get(x, x)
    )
    return (NCSE_nossu_comparison_df,)


@app.cell
def _(NCSE_nossu_comparison_df):
    NCSE_nossu_comparison_df.drop(columns=["filename", "model"]).corr()
    return


@app.cell
def _(MODEL_NAME_MAP, load_results_to_dataframe, pd):
    HNLA_nossu_comparison_df = pd.concat(
        [
            load_results_to_dataframe("data/results/ppdoc/hnla2013_nossu"),
            load_results_to_dataframe("data/results/torch/hnla2013_nossu"),
        ]
    )

    HNLA_nossu_comparison_df["model"] = HNLA_nossu_comparison_df["model"].map(
        lambda x: MODEL_NAME_MAP.get(x, x)
    )

    HNLA_nossu_comparison_df.drop(columns=["filename", "model"]).corr()
    return


if __name__ == "__main__":
    app.run()
