"""Reporting helpers for WFO runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio


def summarise_cycles(per_cycle: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_cycle:
        return {"message": "no_cycles"}
    df = pd.DataFrame(per_cycle)
    summary = {
        "cycles": len(df),
        "sharpe_mean": float(df["sharpe"].mean()),
        "sharpe_median": float(df["sharpe"].median()),
        "sharpe_iqr": float(df["sharpe"].quantile(0.75) - df["sharpe"].quantile(0.25)),
        "sortino_mean": float(df.get("sortino", pd.Series([0])).mean()),
        "calmar_mean": float(df.get("calmar", pd.Series([0])).mean()),
        "max_drawdown_min": float(df.get("max_drawdown", pd.Series([0])).min()),
        "turnover_mean": float(df.get("turnover", pd.Series([0])).mean()),
        "expectancy_mean": float(df.get("expectancy", pd.Series([0])).mean()),
        "slippage_budget_used": float(df.get("slippage_usage", pd.Series([0])).mean()),
    }
    return summary


def write_reports(output_dir: Path, per_cycle: List[Dict[str, Any]], aggregates: Dict[str, Any], extras: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_cycle)
    df.to_csv(output_dir / "per_cycle.csv", index=False)
    (output_dir / "wfo_summary.json").write_text(json.dumps(aggregates, indent=2, default=float))
    for name, payload in extras.items():
        (output_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float))

    # Print concise table
    if not df.empty:
        print("\nPer-cycle metrics:")
        print(df[["cycle", "symbol", "sharpe", "sortino", "max_drawdown", "turnover", "expectancy"]])
        print("\nAggregated summary:")
        for key, value in aggregates.items():
            print(f"  {key}: {value}")


__all__ = ["summarise_cycles", "write_reports"]
