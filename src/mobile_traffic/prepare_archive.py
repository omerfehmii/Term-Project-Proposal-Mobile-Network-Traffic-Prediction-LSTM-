from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Telecom Italia Milan grid data into a single CSV with timestamp, cell_id, traffic."
    )
    parser.add_argument("--archive_dir", type=str, default="archive", help="Directory containing daily CSV files.")
    parser.add_argument("--output", type=str, default="data/milan.csv", help="Output path for merged CSV.")
    parser.add_argument(
        "--metric",
        type=str,
        default="internet",
        choices=["internet", "all"],
        help="Traffic metric to aggregate: 'internet' only or sum of all activity columns.",
    )
    return parser.parse_args()


def load_daily(path: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    dt_col = df["datetime"]
    # convert timestamp: accept ms since epoch or ISO strings
    if pd.api.types.is_numeric_dtype(dt_col):
        df["timestamp"] = pd.to_datetime(dt_col, unit="ms", errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(dt_col, errors="coerce", utc=False)
    df["cell_id"] = df["CellID"]

    value_cols = ["smsin", "smsout", "callin", "callout", "internet"]
    for col in value_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if metric == "internet":
        df["traffic"] = df.get("internet", 0.0)
    else:
        present = [c for c in value_cols if c in df.columns]
        df["traffic"] = df[present].sum(axis=1)
    grouped = df.groupby(["timestamp", "cell_id"], as_index=False)["traffic"].sum()
    return grouped[["timestamp", "cell_id", "traffic"]]


def main():
    args = parse_args()
    archive_dir = Path(args.archive_dir)
    files: List[Path] = sorted(archive_dir.glob("sms-call-internet-mi-*.csv"))
    if not files:
        raise SystemExit(f"No matching traffic CSV files found in {archive_dir} (expected sms-call-internet-mi-*.csv).")

    frames = [load_daily(f, args.metric) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.groupby(["timestamp", "cell_id"], as_index=False)["traffic"].sum()
    combined = combined.sort_values(["timestamp", "cell_id"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(
        f"Wrote {len(combined)} rows across {combined['cell_id'].nunique()} cells "
        f"from {combined['timestamp'].min()} to {combined['timestamp'].max()} -> {output_path}"
    )


if __name__ == "__main__":
    main()
