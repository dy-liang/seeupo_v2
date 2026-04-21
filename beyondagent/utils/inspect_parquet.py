#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any

# Usage: python inspect_parquet.py --rows 50
DEFAULT_PARQUET_PATH = "/Users/dyliang/Desktop/work-2025/project_code/seeupo_v2/external/appworld/data/dev.parquet"


def shorten(value: Any, max_len: int = 180) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = repr(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def summarize_record(record: dict[str, Any]) -> dict[str, str]:
    return {key: shorten(value) for key, value in record.items()}


def print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def inspect_with_pyarrow(path: str, sample_rows: int) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    print_section("Backend")
    print("pyarrow")

    print_section("Schema")
    print(table.schema)

    print_section("Basic Info")
    print(f"rows: {table.num_rows}")
    print(f"columns: {table.num_columns}")
    print(f"column_names: {table.column_names}")

    print_section("Sample Rows")
    rows = table.slice(0, sample_rows).to_pylist()
    for i, row in enumerate(rows):
        print(f"[row {i}]")
        print(json.dumps(summarize_record(row), ensure_ascii=False, indent=2))


def inspect_with_pandas(path: str, sample_rows: int) -> None:
    import pandas as pd

    df = pd.read_parquet(path)
    print_section("Backend")
    print("pandas")

    print_section("Basic Info")
    print(f"rows: {len(df)}")
    print(f"columns: {len(df.columns)}")
    print(f"column_names: {list(df.columns)}")
    print("\ndtypes:")
    print(df.dtypes)

    print_section("Sample Rows")
    records = df.head(sample_rows).to_dict(orient="records")
    for i, row in enumerate(records):
        print(f"[row {i}]")
        print(json.dumps(summarize_record(row), ensure_ascii=False, indent=2))


def inspect_with_duckdb(path: str, sample_rows: int) -> None:
    import duckdb

    con = duckdb.connect()
    schema_rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
    count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path}')").fetchone()[0]
    sample = con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT {sample_rows}").fetchdf()

    print_section("Backend")
    print("duckdb")

    print_section("Schema")
    for row in schema_rows:
        print(row)

    print_section("Basic Info")
    print(f"rows: {count}")
    print(f"columns: {len(sample.columns)}")
    print(f"column_names: {list(sample.columns)}")

    print_section("Sample Rows")
    records = sample.to_dict(orient="records")
    for i, row in enumerate(records):
        print(f"[row {i}]")
        print(json.dumps(summarize_record(row), ensure_ascii=False, indent=2))


def inspect_with_polars(path: str, sample_rows: int) -> None:
    import polars as pl

    df = pl.read_parquet(path)
    print_section("Backend")
    print("polars")

    print_section("Basic Info")
    print(f"rows: {df.height}")
    print(f"columns: {df.width}")
    print(f"column_names: {df.columns}")
    print("\ndtypes:")
    print(df.schema)

    print_section("Sample Rows")
    records = df.head(sample_rows).to_dicts()
    for i, row in enumerate(records):
        print(f"[row {i}]")
        print(json.dumps(summarize_record(row), ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect a parquet file: schema, shape, and sample rows."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=DEFAULT_PARQUET_PATH,
        help=f"Path to parquet file. Default: {DEFAULT_PARQUET_PATH}",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=3,
        help="Number of sample rows to print. Default: 3",
    )
    args = parser.parse_args()

    path = os.path.abspath(os.path.expanduser(args.path))
    if not os.path.exists(path):
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    backends = [
        ("pyarrow", inspect_with_pyarrow),
        ("pandas", inspect_with_pandas),
        ("duckdb", inspect_with_duckdb),
        ("polars", inspect_with_polars),
    ]

    errors = []
    for name, fn in backends:
        try:
            fn(path, args.rows)
            return 0
        except ModuleNotFoundError as exc:
            errors.append(f"{name}: {exc}")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    print("No usable parquet backend is installed.", file=sys.stderr)
    print("Tried:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    print(
        "\nInstall one of: pyarrow, pandas, duckdb, or polars, then rerun this script.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
