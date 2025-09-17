import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", str(Path.home() / ".local/share/m5_trader/data")))


def partition_path(symbol: str, date_str: str, stem: str, ext: str = "parquet") -> Path:
    base = DATA_DIR / f"symbol={symbol}" / f"date={date_str}"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{stem}.{ext}"

