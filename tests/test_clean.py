import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from csvtool.clean import drop_null_rows, coerce_numbers, dedupe


def test_drop_null_rows():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 9]})
    out = drop_null_rows(df, threshold=0.5)
    assert len(out) == 1  # only the fully-populated row survives


def test_coerce_numbers():
    df = pd.DataFrame({"price": ["$1,234", "$567", "890"]})
    out = coerce_numbers(df)
    assert out["price"].iloc[0] == 1234.0
    assert out["price"].dtype != object


def test_dedupe():
    df = pd.DataFrame({"x": [1, 1, 2], "y": ["a", "a", "b"]})
    out = dedupe(df)
    assert len(out) == 2
