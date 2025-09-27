from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from wfo.runner import run_wfo
from wfo.data_access import MarketDataAccess


def test_wfo_supervised_smoke(monkeypatch, tmp_path):
    timestamps = pd.date_range(
        start=pd.Timestamp("2024-01-01 09:30", tz="America/New_York"),
        periods=520,
        freq="1min",
    )
    close = 4000 + np.cumsum(np.random.normal(scale=0.5, size=len(timestamps)))
    returns = pd.Series(close).pct_change().fillna(0.0)
    feature = np.sin(np.linspace(0, 6, len(timestamps)))
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "close": close,
            "feature": feature,
            "returns": returns,
        }
    )
    df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df["label"] = df["label"].fillna(0).astype(int)

    def fake_get_bars(self, symbol, start, end, tz="America/New_York", limit=None):
        return df.copy()

    monkeypatch.setattr(MarketDataAccess, "get_bars", fake_get_bars)

    result = run_wfo(
        symbols=["ES"],
        is_days=1,
        oos_days=1,
        step_days=1,
        cycles_min=1,
        embargo_days=0,
        label_lookahead_bars=1,
        cpcv_folds=2,
        config_path=None,
        dry_run=True,
        output_root=tmp_path / "wfo",
        strategies=[
            {
                "name": "ES_LogReg",
                "type": "supervised",
                "model": "logistic",
                "params": {"target_col": "label", "C": 0.5},
            }
        ],
        rl_fast_smoke=False,
        rl_fast_overrides=None,
        costs_bps=0.0,
    )

    output_dir = Path(result["output_dir"])
    assert output_dir.exists()

    dsr_path = output_dir / "dsr.json"
    assert dsr_path.exists(), "DSR output missing"

    per_cycle = pd.read_csv(output_dir / "per_cycle.csv")
    assert "ES_LogReg" in per_cycle["config"].values
