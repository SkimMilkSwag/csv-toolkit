"""CLI for csv-toolkit."""
import argparse
import json
import os
import sys

from .clean import load, drop_null_rows, coerce_numbers, dedupe, sample
from .stats import summary, profile, flag_sparse_columns


def _emit(payload: str, out: str | None) -> None:
    """Write ``payload`` to a file when ``--out`` was given, else stdout.

    The format follows the extension: ``.json`` gets pretty-printed JSON,
    anything else (including ``.csv``) gets the payload as-is.  Parent
    directories are created on demand; a failed write reports the path and
    exits 1 instead of raising a traceback at the user.
    """
    if not out:
        sys.stdout.write(payload)
        return
    try:
        d = os.path.dirname(os.path.abspath(out))
        os.makedirs(d, exist_ok=True)
        with open(out, "w") as f:
            f.write(payload)
    except OSError as exc:
        print(f"error: writing {out}: {exc}", file=sys.stderr)
        sys.exit(1)


def _json_payload(obj) -> str:
    return json.dumps(obj, indent=2) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(prog="csvtool", description="Clean and summarise CSV files.")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("summary", help="print a compact column summary as JSON")
    s.add_argument("file")
    s.add_argument("--max-null-frac", type=float, default=0.5)
    s.add_argument("--sample", type=int, default=None, metavar="N",
                   help="summarize an i.i.d. sample of N rows instead of the full file (for large files)")
    s.add_argument("--out", metavar="PATH", default=None,
                   help="write the report to PATH (.json pretty-prints; other extensions pass through) "
                        "instead of printing it")

    pr = sub.add_parser("profile", help="per-column type inference + null rate + cardinality")
    pr.add_argument("file")
    pr.add_argument("--coerce", action="store_true", help="try to coerce numeric strings before inferring types")
    pr.add_argument("--out", metavar="PATH", default=None,
                    help="write the profile to PATH (.json pretty-prints; other extensions pass through) "
                         "instead of printing it")

    fl = sub.add_parser("flag-nulls", help="flag columns above a null-rate threshold with drop/impute suggestions")
    fl.add_argument("file")
    fl.add_argument("--null-frac", type=float, default=0.5, metavar="FRAC",
                    help="flag columns whose null rate exceeds this fraction (default 0.5)")

    dd = sub.add_parser("dedupe", help="remove duplicate rows and print the deduplicated table")
    dd.add_argument("file")
    dd.add_argument("--subset", nargs="+", metavar="COL",
                    help="treat rows as duplicates when these columns match (default: all columns)")
    dd.add_argument("--count", action="store_true",
                    help="print how many duplicate rows were removed instead of the table")

    args = p.parse_args(argv)
    if args.cmd == "summary":
        df = load(args.file)
        if args.sample is not None:
            df = sample(df, args.sample)
        df = drop_null_rows(df, args.max_null_frac)
        _emit(_json_payload(summary(df)), args.out)
    elif args.cmd == "profile":
        df = load(args.file)
        if args.coerce:
            df = coerce_numbers(df)
        _emit(_json_payload(profile(df)), args.out)
    elif args.cmd == "flag-nulls":
        df = load(args.file)
        print(json.dumps(flag_sparse_columns(df, args.null_frac), indent=2))
    elif args.cmd == "dedupe":
        df = load(args.file)
        out = dedupe(df, args.subset or None)
        if args.count:
            print(len(df) - len(out))  # how many duplicate rows were removed
        else:
            out.to_csv(sys.stdout, index=False)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
