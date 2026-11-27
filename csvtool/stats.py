"""Quick descriptive stats for a DataFrame."""
import math
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


def top(df: pd.DataFrame, col: str, n: int = 5) -> list:
    """Top-n most frequent non-null values of ``col`` as ``{"value", "count"}`` dicts.

    Same ranking as :func:`top_categories` (pandas ``value_counts`` ordering,
    first-seen among equal counts), but nulls are dropped explicitly and the
    output is JSON-serialisable — numpy scalars are converted to their Python
    equivalents, so the CLI can pretty-print it directly. ``n`` caps the
    list; a column with fewer distinct non-null values than that simply
    reports them all, and an all-null (or empty) column reports nothing.
    """
    vc = df[col].dropna().value_counts().head(n)
    out = []
    for value, count in vc.items():
        v = value.item() if hasattr(value, "item") else value
        out.append({"value": v, "count": int(count)})
    return out


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


def drop_const_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Return ``(df minus constant columns, dropped column names)``.

    A constant column is one with zero variance: at most one distinct value
    counting non-null entries (``nunique(dropna=True) <= 1``). An all-null
    column counts too — it carries no signal either — and an empty frame
    drops every column, so callers should check ``len(df)`` first if that
    would surprise them. Column order of the kept frame is unchanged and the
    input is not mutated (``drop`` returns a copy).
    """
    const = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    return df.drop(columns=const), list(const)


def spread(df: pd.DataFrame, columns: list | None = None) -> dict:
    """Per-numeric-column spread report: min, median, max, q25 and q75.

    Only numeric columns appear (pass ``columns`` to restrict which ones are
    reported). Quartiles are linear-interpolated via ``Series.quantile``; an
    empty frame reports ``None`` for every stat so the JSON stays stable.
    Output is JSON-serialisable, in the shape of :func:`summary`.
    """
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    out = {}
    for col in columns:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        entry = {
            "min": float(s.min()) if len(s) else None,
            "q25": float(s.quantile(0.25)) if len(s) else None,
            "median": float(s.median()) if len(s) else None,
            "q75": float(s.quantile(0.75)) if len(s) else None,
            "max": float(s.max()) if len(s) else None,
        }
        out[col] = entry
    return out


def outlier(df: pd.DataFrame, columns: list | None = None, k: float = 1.5) -> dict:
    """Per-numeric-column Tukey outlier report: IQR fences and flagged counts.

    Fences are ``q1 - k * iqr`` / ``q3 + k * iqr`` (Tukey's rule with
    ``k=1.5`` by default); a value outside the fences counts as flagged.
    Only numeric columns appear — pass ``columns`` to restrict which ones
    are reported, matching :func:`spread`. Nulls are skipped when computing
    fences and flags alike, so they inflate neither count nor rate (the
    rate is over non-null values). An empty frame reports ``None`` for the
    fences and zeros for the counts. Output is JSON-serialisable.
    """
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    out = {}
    for col in columns:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        v = s.dropna()
        if len(v):
            q1 = float(v.quantile(0.25))
            q3 = float(v.quantile(0.75))
            iqr = q3 - q1
            lower, upper = q1 - k * iqr, q3 + k * iqr
            flagged = int(((v < lower) | (v > upper)).sum())
        else:
            lower, upper, flagged = None, None, 0
        out[col] = {
            "lower_fence": round(lower, 4) if lower is not None else None,
            "upper_fence": round(upper, 4) if upper is not None else None,
            "flagged": flagged,
            "flag_rate": round(flagged / len(v), 4) if len(v) else 0.0,
        }
    return out


def correlate(df: pd.DataFrame, columns: list | None = None) -> dict:
    """Pairwise Pearson correlations for the numeric columns of ``df``.

    Returns a symmetric matrix as nested dicts — ``{a: {b: r}}`` for every
    ordered pair of distinct columns — rounded to 4 decimals so the JSON is
    stable. Pairs are compared in column order with each column once, so the
    upper triangle drives the values; the lower triangle mirrors it and each
    column's own diagonal entry is ``1.0``. Nulls are handled pairwise (each
    pair uses the rows where both columns have a value), so a sparse column
    doesn't drag its correlations to null. A constant or all-null column has
    an undefined correlation and reports ``None`` in its row.

    Only numeric columns appear: by default that is every column pandas sees
    as numeric (booleans included, since they carry a 0/1 signal); pass
    ``columns`` to restrict the matrix to named columns, matching the
    ``--columns`` behaviour of :func:`spread` and :func:`outlier`. Non-numeric
    names in that list are silently skipped.
    """
    if columns is None:
        columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        columns = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    # constant (or all-null) columns have an undefined correlation: pearson
    # divides by zero and reports NaN, so mark them up front
    const = {col for col in columns if df[col].dropna().nunique(dropna=True) <= 1}

    out = {col: {other: (1.0 if other == col else None) for other in columns} for col in columns}
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            if a in const or b in const:
                r = None
            else:
                v = df[[a, b]].dropna()
                r = float(v[a].corr(v[b])) if len(v) > 1 else float("nan")
                if r is not None and not math.isnan(r):
                    r = round(r, 4)
                else:
                    r = None
            out[a][b] = r
            out[b][a] = r
    return out


def categories(df: pd.DataFrame, col: str, max_values: int | None = None) -> list:
    """Sorted unique non-null values of a column.

    Values are sorted by (type name, value) so mixed-type columns order
    deterministically instead of raising on cross-type comparison; numpy
    scalars are converted to their Python equivalents for stable JSON output.
    ``max_values`` caps the list; when it does, a trailing ``"…"`` marker
    makes the truncation visible in JSON output.
    """
    vals = df[col].dropna().unique()
    vals = sorted(vals, key=lambda v: (str(type(v)), str(v)))
    if max_values is not None and len(vals) > max_values:
        vals = list(vals[:max_values]) + ["…"]
    return [v.item() if hasattr(v, "item") else v for v in vals]


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
