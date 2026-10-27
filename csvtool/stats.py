"""Quick descriptive stats for a DataFrame."""
import re

import pandas as pd


def summary(df: pd.DataFrame) -> dict:
    """Return a compact per-column summary (count, nulls, mean/min/max for numeric).

    Every column gets ``nulls`` and its pandas ``dtype``; numeric columns
    additionally get rounded ``mean``/``min``/``max`` (``None`` for an empty
    frame). Output is JSON-serialisable — the CLI prints it with
    ``json.dumps(..., indent=2)``.
    """
    out = {}
    for col in df.columns:
        s = df[col]
        entry = {"nulls": int(s.isnull().sum()), "dtype": str(s.dtype)}
        if pd.api.types.is_numeric_dtype(s):
            entry["mean"] = round(float(s.mean()), 4) if len(s) else None
            entry["min"] = float(s.min()) if len(s) else None
            entry["max"] = float(s.max()) if len(s) else None
        out[col] = entry
    return out


def top_categories(df: pd.DataFrame, col: str, n: int = 5) -> list:
    """Return the top-n most frequent categories in a column as (value, count).

    Ties are broken by pandas' ``value_counts`` ordering (first-seen among
    equal counts). Useful for a quick look at a categorical column without
    pulling up a notebook.
    """
    return df[col].value_counts().head(n).reset_index().values.tolist()


def _is_iso8601(value: str) -> bool:
    """Return True when ``value`` parses as an ISO 8601 date or datetime.

    Accepts the common profiles: ``YYYY-MM-DD``, optional time with 'T' or a
    space separator, an optional fractional second, and a trailing zone
    (``Z`` or ``+HH:MM``/``-HH:MM``). Full validation is done with a regex so
    we don't pay for datetime parsing on every value of a large column.
    """
    return bool(
        re.fullmatch(
            r"\d{4}-\d{2}-\d{2}"                      # date
            r"(?:[T ]\d{2}:\d{2}:\d{2}"                # time, T or space separator
            r"(?:\.\d+)?"                              # optional fractional seconds
            r"(?:Z|[+-]\d{2}:?\d{2})?)?",              # optional zone
            value.strip(),
        )
    )


def infer_type(s: pd.Series) -> str:
    """Best-effort type label for a column: int, float, bool, date, or string.

    Native dtypes are used directly; object columns are sniffed (first 20
    non-null values) for numeric, boolean, or ISO 8601 date content. All-
    numeric columns with missing values arrive as float dtype — reported as
    int if every present value is whole, since that's usually the intended
    type.
    """
    if s.dtype == bool:
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        v = s.dropna()
        return "int" if len(v) and (v % 1).abs().max() == 0 else "float"
    # object columns: sniff a sample of the values for numeric/bool/date content
    vals = s.dropna().astype(str).head(20)
    if len(vals):
        numeric = vals.str.replace(r"[\$,]", "", regex=True).str.strip()
        parsed = pd.to_numeric(numeric, errors="coerce")
        if parsed.notna().mean() >= 0.9:
            return "float" if (parsed.dropna() % 1).abs().max() > 0 else "int"
        lowered = vals.str.lower()
        if lowered.isin(["true", "false"]).mean() >= 0.9:
            return "bool"
        if vals.map(_is_iso8601).mean() >= 0.9:
            return "date"
    return "string"


def profile(df: pd.DataFrame) -> dict:
    """Per-column profile: inferred type, null count/rate, and cardinality."""
    n = len(df)
    out = {}
    for col in df.columns:
        s = df[col]
        nulls = int(s.isnull().sum())
        out[col] = {
            "type": infer_type(s),
            "nulls": nulls,
            "null_rate": round(nulls / n, 4) if n else None,
            "cardinality": int(s.nunique(dropna=True)),
        }
    return out


def flag_sparse_columns(df: pd.DataFrame, null_frac_threshold: float = 0.5) -> list:
    """Flag columns whose null rate exceeds ``null_frac_threshold``.

    Returns a list of dicts (sorted by null rate, highest first) with the
    column name, its null count/rate, and a suggested action: ``drop`` when
    more than half the rows are null (the column carries little signal),
    otherwise ``impute`` — worth filling rather than dropping.
    """
    n = len(df)
    flags = []
    for col in df.columns:
        s = df[col]
        nulls = int(s.isnull().sum())
        rate = nulls / n if n else 0.0
        if rate > null_frac_threshold:
            action = "drop" if rate > 0.5 else "impute"
            flags.append(
                {
                    "column": col,
                    "nulls": nulls,
                    "null_rate": round(rate, 4),
                    "suggestion": action,
                }
            )
    flags.sort(key=lambda f: f["null_rate"], reverse=True)
    return flags
