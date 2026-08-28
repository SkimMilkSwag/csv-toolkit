import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
import pandas as pd
from csvtool.clean import drop_null_rows, coerce_numbers, dedupe
from csvtool.stats import profile, infer_type


def test_drop_null_rows():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 9]})
    out = drop_null_rows(df, threshold=0.5)
    assert len(out) == 1  # only the fully-populated row survives


def test_drop_null_rows_boundary():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 9]})
    # threshold 1.0 drops only rows that are entirely null; none qualify
    assert len(drop_null_rows(df, threshold=1.0)) == 2


def test_profile_types_nulls_cardinality():
    df = pd.DataFrame({
        "id": [1, 2, None],
        "price": [10.5, 3.25, 1.0],
        "ok": [True, False, True],
        "name": ["a", "a", "b"],
        "tag": ["x", None, None],
    })
    p = profile(df)
    assert p["id"]["type"] == "int"
    assert p["price"]["type"] == "float"
    assert p["ok"]["type"] == "bool"
    assert p["name"]["type"] == "string"
    assert p["id"]["nulls"] == 1 and abs(p["id"]["null_rate"] - 1 / 3) < 1e-4
    assert p["tag"]["null_rate"] == 0.6667  # 2 of 3 rows null
    assert p["name"]["cardinality"] == 2
    assert p["ok"]["cardinality"] == 2


def test_infer_type_numeric_strings():
    s = pd.Series(["$1,234", "890", "$567"])
    assert infer_type(s) == "int"
    s2 = pd.Series(["$1,234.50", "890", "$567.25"])
    assert infer_type(s2) == "float"
    assert infer_type(pd.Series(["True", "False"])) == "bool"
    assert infer_type(pd.Series([None])) == "string"


def test_profile_cli(tmp_path, capsys):
    import json as _json
    from csvtool.cli import main
    f = tmp_path / "data.csv"
    f.write_text("id,amount,note\n1,\"$10\",hi\n2,,\n3,\"$20.5\",yo\n")
    main(["profile", str(f)])
    out = _json.loads(capsys.readouterr().out)
    assert out["id"]["type"] == "int"
    assert out["amount"]["nulls"] == 1  # the empty cell in row 2
    assert out["note"]["cardinality"] == 2


def test_profile_cli_coerce(tmp_path, capsys):
    import json as _json
    from csvtool.cli import main
    f = tmp_path / "data.csv"
    f.write_text("id,amount\n1,\"$10\"\n2,\"$20\"\n")
    main(["profile", str(f), "--coerce"])
    out = _json.loads(capsys.readouterr().out)
    assert out["amount"]["type"] == "int"


def test_coerce_numbers():
    df = pd.DataFrame({"price": ["$1,234", "$567", "890"]})
    out = coerce_numbers(df)
    assert out["price"].iloc[0] == 1234.0
    assert out["price"].dtype != object


def test_dedupe():
    df = pd.DataFrame({"x": [1, 1, 2], "y": ["a", "a", "b"]})
    out = dedupe(df)
    assert len(out) == 2
