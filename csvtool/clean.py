"""CSV cleaning helpers built on pandas."""
import io
import sys

import pandas as pd


def load(path: str, **kw) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with sensible defaults.

    ``path`` may be the string "-" to read from stdin (useful in pipelines,
    e.g. ``zcat data.csv.zst | csvtool summary -``).
    """
    if path == "-":
        text = sys.stdin.read()
        return pd.read_csv(io.StringIO(text), **kw)
    return pd.read_csv(path, **kw)


def sample(df: pd.DataFrame, n: int, seed: int = 0) -> pd.DataFrame:
    """Return an i.i.d. row sample of at most `n` rows (the whole df if smaller).

    The sample is reproducible for a given `seed`, and the index is reset so
    downstream stats don't carry sampling artifacts.
    """
    if len(df) <= n:
        return df.reset_index(drop=True)
    out = df.sample(n, random_state=seed).sort_index()
    return out.reset_index(drop=True)


def drop_null_rows(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop rows where at least `threshold` fraction of values are null.

    A row is dropped only when the fraction of its null cells meets or
    exceeds `threshold`; with the default 0.5 that keeps every row that has
    at least one populated cell. The index is reset on the result so callers
    can treat it as a fresh table.
    """
    null_frac = df.isnull().mean(axis=1)
    return df[null_frac < threshold].reset_index(drop=True)


def coerce_numbers(df: pd.DataFrame, cols=None):
    """Coerce object columns to numeric where possible (e.g. '$1,234' -> 1234.0).

    Strips currency symbols, thousands separators and surrounding whitespace,
    then parses with `pd.to_numeric(errors="coerce")`. A column is only
    replaced when at least half of its rows became numeric, so a mostly-text
    column keeps its original strings instead of turning into NaN soup.
    """
    if cols is None:
        cols = df.select_dtypes(include=["object"]).columns
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = out[c].astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
        coerced = pd.to_numeric(s, errors="coerce")
        # only replace if at least half the values became numeric
        if coerced.notna().sum() >= max(1, len(out) // 2):
            out[c] = coerced
    return out


def dedupe(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """Remove duplicate rows (optionally on a subset of columns).

    Keeps the first occurrence of each duplicated row and resets the index.
    `subset` restricts what counts as "the same row" — e.g. pass an id column
    to collapse rows that share it, regardless of their other values.
    """
    return df.drop_duplicates(subset=subset).reset_index(drop=True)
