"""CSV cleaning helpers built on pandas."""
import pandas as pd


def load(path: str, **kw) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with sensible defaults."""
    return pd.read_csv(path, **kw)


def drop_null_rows(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Drop rows where more than `threshold` fraction of values are null."""
    null_frac = df.isnull().mean(axis=1)
    return df[null_frac <= threshold].reset_index(drop=True)


def coerce_numbers(df: pd.DataFrame, cols=None):
    """Coerce object columns to numeric where possible (e.g. '$1,234' -> 1234.0)."""
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
    """Remove duplicate rows (optionally on a subset of columns)."""
    return df.drop_duplicates(subset=subset).reset_index(drop=True)
