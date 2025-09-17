from pathlib import Path
from typing import Any


def write_parquet_any(df: Any, path: Path) -> None:
    try:
        import pandas as pd  # type: ignore
        if isinstance(df, pd.DataFrame):
            df.to_parquet(path, index=False, engine="pyarrow")
            return
    except Exception as e:
        raise RuntimeError(f"Parquet write failed, pandas/pyarrow not usable: {e}")

