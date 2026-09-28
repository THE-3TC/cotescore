"""LaTeX formatting helpers for paper tables (booktabs, bold headers, bold-best).

Copied from SpACER/spacer_analysis/tables.py and extended with ``gt_pred_table``.
The same file lives in fakenewslib/src/fakenewslib/tables.py -- keep them in sync by hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Raw index keys -> header labels, so every notebook prints the same row header
# without each call site having to set ``df.index.name``.
INDEX_LABELS = {
    "parsing_model": "Parsing Model",
    "ocr_model": "OCR Model",
    "dataset": "Dataset",
    "model": "Model",
}

# Printed for undefined values (e.g. a correlation over a constant column).
NAN_LABEL = "--"

# Header corner for tables scoring one labelling against another:
# rows are the ground truth, columns the prediction.
GT_PRED_CORNER = r"GT $\downarrow$ / Pred $\rightarrow$"


def bold_best_cols(df: pd.DataFrame, lower_cols=None, higher_cols=None,
                   fmt: str = "{:.3f}") -> pd.DataFrame:
    """Bold the best value per column. Returns a string-valued frame for escape=False output.

    lower_cols: columns where lower is better; every other column is treated as
    higher-is-better (``higher_cols`` is accepted for readability at the call site).
    """
    lower_cols = set(lower_cols or [])
    result = df.copy().astype(object)
    for col in df.columns:
        best = df[col].min() if col in lower_cols else df[col].max()  # pandas skips NaN
        for idx in df.index:
            val = df.loc[idx, col]
            if pd.isna(val):
                result.loc[idx, col] = NAN_LABEL
                continue
            s = fmt.format(val)
            result.loc[idx, col] = f"\\textbf{{{s}}}" if val == best else s
    return result


def bold_best_pivot(df: pd.DataFrame, lower_is_better: bool = True,
                    fmt: str = "{:.3f}") -> pd.DataFrame:
    """Bold column-best values; bold + $^*$ for the overall table best.

    Intended for pivot tables (parsing_model rows x ocr_model columns).
    """
    fn = "min" if lower_is_better else "max"
    col_best = getattr(df, fn)(axis=0)
    table_best = float(getattr(np, f"nan{fn}")(df.values))
    result = df.copy().astype(object)
    for col in df.columns:
        for idx in df.index:
            val = df.loc[idx, col]
            if pd.isna(val):
                result.loc[idx, col] = NAN_LABEL
                continue
            s = fmt.format(val)
            if val == table_best:
                s = f"\\textbf{{{s}}}$^{{*}}$"
            elif val == col_best[col]:
                s = f"\\textbf{{{s}}}"
            result.loc[idx, col] = s
    return result


def latex_table(df: pd.DataFrame, caption: str, label: str, col_fmt: str | None = None,
                echo: bool = True) -> str:
    """Return (and by default print) a centered, bold-header booktabs table (position=htbp).

    The index is folded into the header row as a plain leading column rather
    than pandas's separate index-name row: ``\\textbf{Row} & \\textbf{Col1} & ...``.
    Raw index keys listed in :data:`INDEX_LABELS` are relabelled; an unnamed
    index gets an empty header cell.
    Cell values are emitted verbatim (escape=False), so pre-format with
    :func:`bold_best_cols` / :func:`bold_best_pivot` or pass strings.
    """
    index_label = INDEX_LABELS.get(df.index.name, df.index.name)
    _df = df.rename_axis("" if index_label is None else index_label).reset_index()
    col_fmt = col_fmt or ("l" + "c" * (_df.shape[1] - 1))
    # pandas runs each header through str.format, so braces must be doubled --
    # both the \textbf{} wrapper and any inside a math-mode column label.
    headers = [
        r"\textbf{{" + str(c).replace("{", "{{").replace("}", "}}") + "}}"
        for c in _df.columns
    ]
    raw = _df.to_latex(
        index=False,
        header=headers,
        caption=caption,
        label=label,
        escape=False,
        position="htbp",
        column_format=col_fmt,
        float_format="%.3f",
    )
    # Replace \hline with booktabs rules (\toprule, \midrule, \bottomrule)
    hline_count = 0
    lines = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if stripped.startswith(r"\begin{table}"):
            lines.append(line)
            lines.append(r"\centering")
        elif stripped == r"\hline":
            hline_count += 1
            lines.append(
                r"\toprule" if hline_count == 1
                else r"\midrule" if hline_count == 2
                else r"\bottomrule"
            )
        else:
            lines.append(line)
    out = "\n".join(lines)
    if echo:
        print(out)
    return out


def gt_pred_table(df: pd.DataFrame, caption: str, label: str, fmt: str = "{:.2f}",
                  echo: bool = True) -> str:
    """A matrix scoring one labelling against another: rows GT, columns prediction.

    The header corner reads ``GT ↓ / Pred →`` so the axes need no extra
    header row. Numeric cells are formatted with ``fmt``; strings pass through.
    """
    body = df.map(lambda v: v if isinstance(v, str) else NAN_LABEL if pd.isna(v) else fmt.format(v))
    return latex_table(body.rename_axis(GT_PRED_CORNER), caption, label, echo=echo)
