"""CLI for csv-toolkit."""
import argparse
import json

from .clean import load, drop_null_rows, coerce_numbers
from .stats import summary


def main(argv=None):
    p = argparse.ArgumentParser(prog="csvtool", description="Clean and summarise CSV files.")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("summary", help="print a compact column summary as JSON")
    s.add_argument("file")
    s.add_argument("--max-null-frac", type=float, default=0.5)

    args = p.parse_args(argv)
    if args.cmd == "summary":
        df = load(args.file)
        df = drop_null_rows(df, args.max_null_frac)
        print(json.dumps(summary(df), indent=2))
    else:
        p.print_help()


if __name__ == "__main__":
    main()
