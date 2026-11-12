from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATA_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the public Jena Climate CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "jena_climate_2009_2016.csv"
    archive_path = args.output_dir / "jena_climate_2009_2016.csv.zip"
    if csv_path.exists() and not args.force:
        print(f"Dataset already exists at {csv_path}")
        return 0
    print(f"Downloading {DATA_URL}")
    with (
        urllib.request.urlopen(DATA_URL) as response,
        archive_path.open("wb") as output,
    ):
        shutil.copyfileobj(response, output)
    with zipfile.ZipFile(archive_path) as archive:
        member = next(name for name in archive.namelist() if name.endswith(".csv"))
        with archive.open(member) as source, csv_path.open("wb") as output:
            shutil.copyfileobj(source, output)
    archive_path.unlink()
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
