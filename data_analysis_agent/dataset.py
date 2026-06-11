"""Dataset wrapper + the safe-pandas operations the agent's tools call.

Named `dataset.py` (not `data.py`) so it doesn't collide with the
`data_analysis_agent/data/` package directory.
"""
from __future__ import annotations

import importlib.resources as resources
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


SUPPORTED_AGGS = ("mean", "median", "sum", "min", "max", "count", "std", "nunique")


def bundled_penguins_path() -> Path:
    with resources.as_file(
        resources.files("data_analysis_agent.data").joinpath("penguins.csv")
    ) as p:
        return Path(p)


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    n_null: int
    n_unique: int
    summary: Dict[str, Any]


class Dataset:
    """A read-only pandas DataFrame wrapper with bounded query results.

    The wrapper enforces three things:
      1. Column names from the LLM are validated against the schema.
      2. `query` runs a safe pandas `df.query()` expression — no eval()
         on raw Python, no DataFrame mutation, results capped at 50 rows.
      3. Aggregations only use a closed set of named operations
         (mean / median / sum / min / max / count / std / nunique) —
         never `agg(lambda)` or arbitrary callables.
    """

    DEFAULT_ROW_CAP = 50

    def __init__(self, df: pd.DataFrame, row_cap: int = DEFAULT_ROW_CAP):
        if df.empty:
            raise ValueError("Dataset must contain at least one row")
        self.df = df.reset_index(drop=True)
        self.row_cap = row_cap

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def n_cols(self) -> int:
        return len(self.df.columns)

    def columns(self) -> List[ColumnSummary]:
        out: List[ColumnSummary] = []
        for name in self.df.columns:
            series = self.df[name]
            dtype = str(series.dtype)
            n_null = int(series.isna().sum())
            n_unique = int(series.nunique(dropna=True))
            summary: Dict[str, Any] = {}
            if pd.api.types.is_numeric_dtype(series):
                # Drop NaNs to compute, JSON-friendly rounding.
                desc = series.describe(percentiles=[]).to_dict()
                for k, v in desc.items():
                    try:
                        summary[k] = round(float(v), 4)
                    except (TypeError, ValueError):
                        summary[k] = v
            else:
                vc = series.value_counts(dropna=True).head(5)
                summary = {str(k): int(v) for k, v in vc.items()}
            out.append(ColumnSummary(
                name=name, dtype=dtype, n_null=n_null,
                n_unique=n_unique, summary=summary,
            ))
        return out

    def _require_column(self, name: str) -> None:
        if name not in self.df.columns:
            raise ValueError(
                f"no column named {name!r}; available: {list(self.df.columns)}"
            )

    def describe_column(self, name: str) -> Dict[str, Any]:
        self._require_column(name)
        col = self.df[name]
        result: Dict[str, Any] = {
            "name": name,
            "dtype": str(col.dtype),
            "n_null": int(col.isna().sum()),
            "n_unique": int(col.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(col):
            d = col.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
            for k, v in d.items():
                try:
                    result[k] = round(float(v), 4)
                except (TypeError, ValueError):
                    result[k] = v
        else:
            vc = col.value_counts(dropna=True).head(20)
            result["value_counts"] = {str(k): int(v) for k, v in vc.items()}
        return result

    def query(self, expr: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Run a pandas `df.query` expression and return up to `max_rows`."""
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError("'expr' must be a non-empty string")
        try:
            result = self.df.query(expr)
        except Exception as e:
            raise ValueError(f"bad query expression: {e}") from None
        cap = self.row_cap if max_rows is None else max(1, min(int(max_rows), self.row_cap))
        return result.head(cap)

    def aggregate(
        self,
        group_by: List[str],
        value_col: str,
        agg: str,
    ) -> pd.DataFrame:
        """Groupby + named aggregation. `agg` is one of SUPPORTED_AGGS."""
        if not group_by:
            raise ValueError("group_by must contain at least one column")
        for g in group_by:
            self._require_column(g)
        if agg != "count":
            # `count` works on any column; the others require a numeric value_col.
            self._require_column(value_col)
            if not pd.api.types.is_numeric_dtype(self.df[value_col]):
                raise ValueError(
                    f"agg {agg!r} requires a numeric value_col; {value_col!r} is "
                    f"{self.df[value_col].dtype}"
                )
        else:
            self._require_column(value_col)
        if agg not in SUPPORTED_AGGS:
            raise ValueError(
                f"agg must be one of {SUPPORTED_AGGS}, got {agg!r}"
            )
        grouped = self.df.groupby(group_by, dropna=False, observed=True)[value_col]
        out = grouped.agg(agg).reset_index()
        out = out.rename(columns={value_col: f"{value_col}_{agg}"})
        # Sort by the aggregate result for predictable ordering.
        out = out.sort_values(out.columns[-1], ascending=False, kind="stable")
        return out.head(self.row_cap)

    def correlation(self, col1: str, col2: str) -> Dict[str, Any]:
        """Pearson correlation between two numeric columns."""
        for col in (col1, col2):
            self._require_column(col)
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                raise ValueError(
                    f"correlation requires numeric columns; {col!r} is "
                    f"{self.df[col].dtype}"
                )
        pair = self.df[[col1, col2]].dropna()
        n_used = len(pair)
        if n_used < 2:
            raise ValueError(f"not enough non-null pairs (got {n_used}) for correlation")
        r = float(pair[col1].corr(pair[col2]))
        return {
            "col1": col1,
            "col2": col2,
            "method": "pearson",
            "n_used": n_used,
            "r": round(r, 4),
        }


def format_dataframe(df: pd.DataFrame, max_chars: int = 4000) -> str:
    """Render a DataFrame as a compact pipe-separated text table for the LLM."""
    if df.empty:
        return "(empty)"
    cols = [str(c) for c in df.columns]
    header = " | ".join(cols)
    sep = "-+-".join("-" * len(c) for c in cols)
    body_lines: List[str] = []
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if pd.isna(v):
                cells.append("NA")
            elif isinstance(v, float):
                cells.append(f"{v:.3f}" if abs(v) < 1e6 else f"{v:g}")
            else:
                cells.append(str(v))
        body_lines.append(" | ".join(cells))
    text = "\n".join([header, sep, *body_lines])
    if len(text) > max_chars:
        text = text[: max_chars - 50] + f"\n... [{len(text) - max_chars} chars elided]"
    return text


def load_penguins(path: Optional[Path | str] = None) -> Dataset:
    """Load the bundled Palmer Penguins CSV into a `Dataset`."""
    p = Path(path) if path else bundled_penguins_path()
    df = pd.read_csv(p)
    return Dataset(df=df)
