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
```

## Usage (library)

```python
from csvtool.clean import load, drop_null_rows, coerce_numbers
from csvtool.stats import summary, top_categories

df = load("data.csv")
df = coerce_numbers(drop_null_rows(df))
print(summary(df))
print(top_categories(df, "category"))
```

## Tests

```bash
python -m pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
