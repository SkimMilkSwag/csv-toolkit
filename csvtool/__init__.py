"""csv-toolkit — small pandas-based CSV cleaning and analysis toolkit.

Handles the annoying real-world CSV cases with a tiny, dependency-light API:
dropping mostly-null rows, coercing strings like ``$1,234`` into numbers,
deduping, and printing compact per-column summaries or profiles.

Sub-modules
-----------
csvtool.clean
    Data loading and cleaning: :func:`load` (file or stdin via ``-``),
    :func:`sample`, :func:`drop_null_rows`, :func:`coerce_numbers`,
    :func:`dedupe`.
csvtool.stats
    Descriptive statistics: :func:`summary`, :func:`top_categories`,
    :func:`infer_type`, :func:`profile`, :func:`flag_sparse_columns`.
csvtool.cli
    The ``csvtool`` command-line entry point (``summary``, ``profile``,
    ``flag-nulls``).

Example
-------
>>> from csvtool.clean import load, coerce_numbers, drop_null_rows
>>> from csvtool.stats import profile
>>> df = coerce_numbers(drop_null_rows(load("data.csv")))   # doctest: +SKIP
>>> profile(df)                                             # doctest: +SKIP
"""
__version__ = "0.1.0"


def __getattr__(name):  # noqa: D401 - PEP 562 lazy submodule re-export
    """Lazily expose the sub-modules (``csvtool.clean``, ``csvtool.stats``,
    ``csvtool.cli``) on the package without importing them eagerly — keeps
    ``import csvtool`` cheap and pandas-free until a sub-module is requested."""
    import importlib

    if name in ("clean", "stats", "cli"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["clean", "stats", "cli", "__version__"]
