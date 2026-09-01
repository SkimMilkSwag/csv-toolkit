import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
import pandas as pd
from csvtool.clean import drop_null_rows, coerce_numbers, dedupe, sample
from csvtool.stats import profile, infer_type, summary


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


def test_sample_size_and_reproducible():
    df = pd.DataFrame({"v": range(100)})
    out = sample(df, 5)
    assert len(out) == 5
    assert list(out.index) == list(range(5))  # index reset
    assert (sample(df, 5) == out).all().all()  # same seed -> same rows
    other = sample(df, 5, seed=42)
    assert not (other == out).all().all()      # different seed -> different rows


def test_sample_smaller_df_returns_all():
    df = pd.DataFrame({"v": [1, 2, 3]})
    out = sample(df, 10)
    assert len(out) == 3 and list(out.index) == [0, 1, 2]


def test_load_from_stdin(monkeypatch):
    import io
    from csvtool.clean import load

    monkeypatch.setattr("sys.stdin", io.StringIO("id,note\n1,hi\n2,lo\n"))
    df = load("-")
    assert list(df.columns) == ["id", "note"]
    assert len(df) == 2
    assert df["id"].tolist() == [1, 2]


def test_load_stdin_via_cli(tmp_path, capsys, monkeypatch):
    import io
    import json as _json
    from csvtool.cli import main

    monkeypatch.setattr("sys.stdin", io.StringIO("id,value\n1,10\n2,20\n3,30\n"))
    main(["summary", "-"])
    out = _json.loads(capsys.readouterr().out)
    assert out["id"]["nulls"] == 0 and out["value"]["max"] == 30.0


def test_summary_cli_sample(tmp_path, capsys):
    import json as _json
    from csvtool.cli import main
    f = tmp_path / "big.csv"
    f.write_text("id,value\n" + "\n".join(f"{i},{i * 2}" for i in range(100)) + "\n")
    main(["summary", str(f)])
    full = _json.loads(capsys.readouterr().out)
    assert full["id"]["nulls"] == 0 and full["id"]["max"] == 99.0

    # --sample 5 must summarize only the reproducible seed-0 draw of 5 rows:
    # sample() uses df.sample(n, random_state=seed), so the exact rows are
    # knowable and the summary's max id pins the draw.
    import pandas as _pd
    idx = _pd.DataFrame({"id": range(100)}).sample(5, random_state=0)["id"].sort_values()
    main(["summary", str(f), "--sample", "5"])
    s = _json.loads(capsys.readouterr().out)
    assert int(s["id"]["max"]) == int(idx.max())
    assert abs(s["value"]["mean"] - 2 * idx.mean()) < 1e-4
