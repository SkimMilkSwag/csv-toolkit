# csv-toolkit

A small toolkit for cleaning and summarising CSV files with pandas. Handles the
annoying real-world cases: dropping mostly-null rows, coercing strings like
`$1,234` into numbers, deduping, per-column type profiling, and printing a
compact summary, spread (min/q25/median/q75/max), or Tukey IQR outlier report.

## Install

```bash
pip install -e .
```

## Usage (CLI)

```bash
csvtool summary data.csv
csvtool summary data.csv --max-null-frac 0.3

# Per-column type inference, null rate, and cardinality
# (infers int/float/bool/date/string; ISO 8601 date & datetime columns -> "date")
csvtool profile data.csv
csvtool profile data.csv --coerce   # coerce numeric strings before inferring

# Suppress zero-variance columns (cardinality <= 1, all-null included) from
# the report — handy for dropping env/build-tag style constants at a glance
csvtool summary data.csv --drop-const
csvtool profile data.csv --drop-const

# Write the report to a file instead of printing it (.json pretty-prints;
# other extensions pass the JSON through verbatim)
csvtool summary data.csv --out reports/summary.json
csvtool profile data.csv --coerce --out profile.json

# Per-numeric-column min / q25 / median / q75 / max as JSON (non-numeric
# columns are omitted; --columns restricts the report to named columns)
csvtool spread data.csv
csvtool spread data.csv --columns price qty --out reports/spread.json

# Sorted unique values of a column as JSON (nulls dropped, deduplicated;
# --max caps the list and appends an … marker when it truncates)
csvtool categories data.csv --column region
csvtool categories data.csv --column region --max 10

# Per-numeric-column Tukey IQR outlier report: q1 - k*iqr / q3 + k*iqr fences,
# count and rate of values outside the fences (nulls skipped; --columns
# restricts the report, --k loosens/tightens the fences)
csvtool outlier data.csv
csvtool outlier data.csv --columns price qty --k 3

# Flag columns above a null-rate threshold (default 0.5) with drop/impute suggestions
csvtool flag-nulls data.csv
csvtool flag-nulls data.csv --null-frac 0.3

# Remove duplicate rows (whole-row, or by the columns named with --subset)
csvtool dedupe data.csv
csvtool dedupe data.csv --subset id
csvtool dedupe data.csv --count    # print how many duplicate rows were removed
csvtool dedupe data.csv --out clean.csv   # file the result; --count still prints to stdout
```

## Usage (library)

```python
from csvtool.clean import load, drop_null_rows, coerce_numbers
from csvtool.stats import summary, top_categories, profile

df = load("data.csv")
df = coerce_numbers(drop_null_rows(df))
print(summary(df))
print(top_categories(df, "category"))
```

## Tests

```bash
python -m pytest tests/ -v
```

## Docs

Module docstrings are Sphinx-ready; build the API reference with:

```bash
pip install -e .[docs]
sphinx-build docs docs/_build/html
```

## License

MIT — see [LICENSE](LICENSE).
