import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
import pandas as pd
from csvtool.clean import drop_null_rows, coerce_numbers, dedupe, sample
from csvtool.stats import profile, infer_type, summary, flag_sparse_columns


def test_drop_null_rows():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 9]})
    out = drop_null_rows(df, threshold=0.5)
    assert len(out) == 1  # only the fully-populated row survives


def test_drop_null_rows_boundary():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 9]})
    # threshold 1.0 drops only rows that are entirely null; none qualify
    assert len(drop_null_rows(df, threshold=1.0)) == 2


def test_drop_null_rows_exactly_at_threshold_is_dropped():
    # drop_null_rows drops a row when its null fraction MEETS the threshold
    # (null_frac < threshold keeps the row), so at threshold 0.5 any row that
    # is exactly half-null goes away, while a fully-populated row survives.
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 9]})
    # row1 (1, None) -> 0.5 null frac; row2 (None, None) -> 1.0; row3 (3, 9) -> 0.0
    assert list(drop_null_rows(df, threshold=0.5).itertuples(index=False)) == [(3, 9)]
    # one step below the threshold: only the fully-null row (1.0) is dropped,
    # so both half-null-free rows are kept
    assert len(drop_null_rows(df, threshold=0.6)) == 2
    # a threshold just above every row's null frac keeps everything
    assert len(drop_null_rows(df, threshold=1.1)) == 3


def test_drop_null_rows_empty_df():
    out = drop_null_rows(pd.DataFrame({"a": []}), threshold=0.5)
    assert len(out) == 0 and list(out.columns) == ["a"]


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


def test_infer_type_iso8601_dates():
    # plain dates
    assert infer_type(pd.Series(["2026-10-01", "2026-10-02", "2025-01-31"])) == "date"
    # full datetimes with Z or numeric zones, fractional seconds, space separator
    assert infer_type(pd.Series([
        "2026-10-27T14:30:00Z",
        "2026-10-27T14:30:05.123+04:00",
        "2026-10-27 14:30:00",
    ])) == "date"


def test_infer_type_iso8601_mixed_or_junk_stays_string():
    # below the 90% sniff threshold -> falls through to string
    assert infer_type(pd.Series(["2026-10-01", "not-a-date"])) == "string"
    assert infer_type(pd.Series(["October 27, 2026", "yesterday"])) == "string"
    # wrong shapes must not match: month-first, slashes, bare times
    assert infer_type(pd.Series(["27/10/2026", "14:30:00", "2026-10-01T9"])) == "string"


def test_profile_cli_reports_date_column(tmp_path, capsys):
    import json as _json
    from csvtool.cli import main
    f = tmp_path / "events.csv"
    f.write_text("event_id,created_at\n1,2026-10-01T08:00:00Z\n2,2026-10-02T09:30:00Z\n")
    main(["profile", str(f)])
    out = _json.loads(capsys.readouterr().out)
    assert out["created_at"]["type"] == "date"


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


def test_flag_sparse_columns_default_threshold():
    # 10 rows; 'sparse' is null in 8 (0.8), 'borderline' in 6 (0.6) — both
    # above the default 0.5 flag threshold AND majority-null, so both drop
    df = pd.DataFrame({
        "sparse": [None] * 8 + [1, 2],
        "borderline": [None] * 6 + [1, 2, 3, 4],
        "ok": [1, None, 3, 4, 5, 6, 7, 8, 9, 10],
    })
    flags = flag_sparse_columns(df)
    assert [f["column"] for f in flags] == ["sparse", "borderline"]  # sorted by null rate
    assert flags[0]["null_rate"] == 0.8 and flags[0]["suggestion"] == "drop"
    assert flags[1]["null_rate"] == 0.6 and flags[1]["suggestion"] == "drop"

    # 'soft' is 4 of 10 null (0.4): below the default threshold, not flagged;
    # at a lower threshold it qualifies but is under 50% null -> impute
    df2 = pd.concat(
        [df, pd.DataFrame({"soft": [None] * 4 + list(range(6))})], axis=1
    )
    assert all(f["column"] != "soft" for f in flag_sparse_columns(df2))
    soft = [f for f in flag_sparse_columns(df2, null_frac_threshold=0.3) if f["column"] == "soft"]
    assert len(soft) == 1 and soft[0]["suggestion"] == "impute"


def test_flag_sparse_columns_custom_threshold():
    df = pd.DataFrame({
        "a": [None, None, 1, 2],          # 0.5 nulls
        "b": [None, None, None, 2],       # 0.75 nulls
    })
    # default 0.5: only 'b' exceeds the threshold (strictly greater); majority missing -> drop
    out = flag_sparse_columns(df)
    assert [f["column"] for f in out] == ["b"]
    assert out[0]["suggestion"] == "drop"
    # lower threshold to 0.25: both columns qualify; 'a' is exactly half null -> impute
    out = flag_sparse_columns(df, null_frac_threshold=0.25)
    assert [f["column"] for f in out] == ["b", "a"]  # still sorted by null rate
    assert {f["suggestion"] for f in out} == {"drop", "impute"}


def test_flag_sparse_columns_empty_df():
    df = pd.DataFrame({"a": []})
    assert flag_sparse_columns(df) == []


def test_flag_nulls_cli(tmp_path, capsys):
    import json as _json
    from csvtool.cli import main
    f = tmp_path / "data.csv"
    # 'rare' is null in 3 of 4 rows (majority) -> drop; 'main' is never null
    f.write_text("main,rare\n1,\n2,7\n3,\n4,\n")
    main(["flag-nulls", str(f)])
    out = _json.loads(capsys.readouterr().out)
    assert [o["column"] for o in out] == ["rare"]
    assert out[0]["null_rate"] == 0.75 and out[0]["suggestion"] == "drop"


def test_summary_out_writes_json_file(tmp_path, capsys):
    import json as _json
    from csvtool.cli import main
    f = tmp_path / "data.csv"
    f.write_text("id,value\n1,10\n2,20\n3,30\n")
    out_path = str(tmp_path / "reports" / "summary.json")
    main(["summary", str(f), "--out", out_path])
    # nothing on stdout; the file holds the same JSON a print would
    assert capsys.readouterr().out == ""
    payload = _json.loads(open(out_path).read())
    assert payload["id"]["nulls"] == 0 and payload["value"]["max"] == 30.0


def test_summary_out_non_json_extension_passes_through(tmp_path, capsys):
    import json as _json
    from csvtool.cli import main
    f = tmp_path / "data.csv"
    f.write_text("id,value\n1,10\n2,20\n")
    out_path = str(tmp_path / "summary.txt")
    main(["summary", str(f), "--out", out_path])
    # a .txt still gets the JSON document verbatim (no re-formatting)
    payload = _json.loads(open(out_path).read())
    assert payload["id"]["nulls"] == 0


def test_profile_out_writes_file(tmp_path, capsys):
    import json as _json
    from csvtool.cli import main
    f = tmp_path / "data.csv"
    f.write_text("id,amount\n1,10.5\n2,20.25\n")
    out_path = str(tmp_path / "profile.json")
    main(["profile", str(f), "--out", out_path])
    assert capsys.readouterr().out == ""
    payload = _json.loads(open(out_path).read())
    assert payload["amount"]["type"] == "float"


def test_out_bad_path_exits_one(tmp_path, capsys):
    import pytest
    from csvtool.cli import main
    f = tmp_path / "data.csv"
    f.write_text("id\n1\n")
    # a path whose parent directory doesn't exist yet and can't be created
    # (we use the read-only root of tmp_path's grandparent): simplest is to
    # point at a file inside an existing file, so makedirs succeeds but open()
    # hits NotADirectoryError
    blocker = tmp_path / "blocker.txt"
    blocker.write_text("x")
    bad = str(blocker / "nested" / "out.json")
    with pytest.raises(SystemExit) as exc:
        main(["summary", str(f), "--out", bad])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "error:" in err and "out.json" in err


def test_dedupe_cli_prints_table(tmp_path, capsys):
    from csvtool.cli import main
    f = tmp_path / "dups.csv"
    # two fully-duplicate rows (1,a) and one unique row (2,b)
    f.write_text("id,label\n1,a\n1,a\n2,b\n")
    main(["dedupe", str(f)])
    out = _io_read(capsys)
    lines = out.strip().splitlines()
    assert lines[0] == "id,label"
    # exactly the two unique rows survive, in order
    assert [ln for ln in lines[1:]] == ["1,a", "2,b"]


def test_dedupe_cli_subset(tmp_path, capsys):
    from csvtool.cli import main
    f = tmp_path / "dups.csv"
    # rows share 'id' but differ in 'note'; --subset id collapses them
    f.write_text("id,note\n1,first\n1,second\n2,only\n")
    main(["dedupe", str(f), "--subset", "id"])
    out = _io_read(capsys).strip().splitlines()
    # first occurrence kept per id, so 1,first and 2,only survive
    assert [ln for ln in out[1:]] == ["1,first", "2,only"]


def test_dedupe_cli_count(tmp_path, capsys):
    from csvtool.cli import main
    f = tmp_path / "dups.csv"
    f.write_text("id,label\n1,a\n1,a\n1,a\n2,b\n")
    main(["dedupe", str(f), "--count"])
    assert _io_read(capsys).strip() == "2"  # 4 rows -> 2 unique, 2 removed


def test_dedupe_cli_no_duplicates_count_zero(tmp_path, capsys):
    from csvtool.cli import main
    f = tmp_path / "clean.csv"
    f.write_text("id\n1\n2\n3\n")
    main(["dedupe", str(f), "--count"])
    assert _io_read(capsys).strip() == "0"


def _io_read(capsys) -> str:
    """capsys.readouterr().out, isolated so tests above stay unaffected."""
    return capsys.readouterr().out
