# csv-toolkit

A small toolkit for cleaning and summarising CSV files with pandas. Handles the
annoying real-world cases: dropping mostly-null rows, coercing strings like
`$1,234` into numbers, deduping, and printing a compact per-column summary.

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

# Write the report to a file instead of printing it (.json pretty-prints;
# other extensions pass the JSON through verbatim)
csvtool summary data.csv --out reports/summary.json
csvtool profile data.csv --coerce --out profile.json

# Flag columns above a null-rate threshold (default 0.5) with drop/impute suggestions
csvtool flag-nulls data.csv
csvtool flag-nulls data.csv --null-frac 0.3

# Remove duplicate rows (whole-row, or by the columns named with --subset)
csvtool dedupe data.csv
csvtool dedupe data.csv --subset id
csvtool dedupe data.csv --count    # print how many duplicate rows were removed
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
