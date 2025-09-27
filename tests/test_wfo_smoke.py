import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from wfo.cpcv import CombinatorialPurgedCV, CPCVConfig
from wfo.runner import run_wfo
import wfo.runner as wfo_runner


def _synthetic_bars(days: int = 40) -> pd.DataFrame:
    start = datetime(2024, 1, 1)
    periods = days * 24 * 60
    index = pd.date_range(start=start, periods=periods, freq="1min", tz="America/New_York")
    price = 100 + np.sin(np.linspace(0, 3, len(index)))
    df = pd.DataFrame({
        "timestamp": index,
        "open": price,
        "high": price + 0.2,
        "low": price - 0.2,
        "close": price + 0.1,
        "volume": 1_000,
    })
    return df


def test_cpcv_purge_embargo():
    n = 120
    ts = pd.date_range("2024-01-01", periods=n, freq="T")
    cfg = CPCVConfig(n_groups=6, test_group_size=2, embargo=2, label_lookahead=2, max_splits=3, random_state=7)
    splitter = CombinatorialPurgedCV(cfg)

    for train_idx, test_idx in splitter.split(ts):
        assert set(train_idx).isdisjoint(set(test_idx))
        for ti in test_idx:
            for look in range(-cfg.label_lookahead, cfg.label_lookahead + 1):
                assert (ti + look) not in train_idx
            for emb in range(1, cfg.embargo + 1):
                assert (ti + emb) not in train_idx


def test_wfo_smoke(tmp_path, monkeypatch):
    def fake_get_bars(self, symbol, start, end, tz="America/New_York", limit=None):
        return _synthetic_bars(days=60)

    monkeypatch.setattr("wfo.data_access.MarketDataAccess.get_bars", fake_get_bars)

    bars = _synthetic_bars(days=60)
    raw_cycles = wfo_runner._build_cycles(bars, is_days=10, oos_days=4, step_days=4)
    assert raw_cycles, "expected at least one WFO cycle"
    raw_is, raw_oos = raw_cycles[0]
    label_lookahead = 1
    embargo_bars = int(1 * 390)
    expected_is_len = len(raw_is) - label_lookahead
    expected_oos_len = len(raw_oos) - embargo_bars
    assert expected_is_len > 0
    assert expected_oos_len > 0
    expected_oos_first = raw_oos.iloc[embargo_bars]["timestamp"] if expected_oos_len > 0 else None

    selection_capture = {}
    original_selection = wfo_runner._run_cpcv_selection

    def patched_selection(is_df, config, strategies, *args, **kwargs):
        selection_capture.setdefault("is_len", len(is_df))
        return original_selection(is_df, config, strategies, *args, **kwargs)

    monkeypatch.setattr("wfo.runner._run_cpcv_selection", patched_selection)

    original_strategy_returns = wfo_runner._strategy_returns

    def patched_strategy_returns(df, strategy):
        if selection_capture.get("oos_len") is None and "timestamp" in df.columns:
            if len(df) == expected_oos_len and expected_oos_first is not None:
                if not df.empty and df["timestamp"].iloc[0] == expected_oos_first:
                    selection_capture["oos_len"] = len(df)
        return original_strategy_returns(df, strategy)

    monkeypatch.setattr("wfo.runner._strategy_returns", patched_strategy_returns)

    out_dir = tmp_path / "artifacts"
    result = run_wfo(
        symbols=["ES"],
        is_days=10,
        oos_days=4,
        step_days=4,
        cycles_min=1,
        embargo_days=1,
        label_lookahead_bars=1,
        cpcv_folds=5,
        config_path=Path("wfo/wfo_config.yaml"),
        dry_run=True,
        output_root=out_dir,
        strategies=[
            {"name": "ma_fast", "type": "moving_average", "params": {"fast": 10, "slow": 40}}
        ],
    )

    artifacts_path = Path(result["output_dir"])
    assert (artifacts_path / "wfo_summary.json").exists()
    assert (artifacts_path / "per_cycle.csv").exists()
    assert (artifacts_path / "dsr.json").exists()
    assert selection_capture.get("is_len") == expected_is_len
    assert selection_capture.get("oos_len") == expected_oos_len
    assert result["summary"].get("effective_trials") == result["dsr"]["effective_trials"]
