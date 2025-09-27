"""Main orchestration for the walk-forward optimisation framework."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import yaml

from .cpcv import CombinatorialPurgedCV, CPCVConfig
from .data_access import MarketDataAccess
from .metrics import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    turnover,
    expectancy,
    deflated_sharpe_ratio,
)
from .reporting import summarise_cycles, write_reports
from .reality_checks import white_reality_check, hansen_spa


@dataclass
class StrategyConfig:
    name: str
    type: str
    params: Dict[str, Any]


@dataclass
class RunnerConfig:
    strategies: List[StrategyConfig]
    trading_days_per_year: int = 252
    minutes_per_trading_day: int = 390
    label_lookahead_bars: int = 1
    embargo_days: int = 1
    cpcv_groups: int = 12
    cpcv_test_groups: int = 2
    cpcv_max_splits: int | None = None
    cpcv_random_state: int | None = 7
    dsr_threshold: float = 0.05
    go_no_go: Dict[str, Any] | None = None
    rc_bootstrap: int = 1000
    rc_block_len: int = 78


def load_config(path: Path | str | None) -> RunnerConfig:
    if path is None:
        defaults = _default_yaml()
    else:
        with open(path, "r", encoding="utf-8") as fh:
            defaults = yaml.safe_load(fh)
    strategies = [StrategyConfig(**entry) for entry in defaults.get("strategies", [])]
    return RunnerConfig(
        strategies=strategies,
        trading_days_per_year=defaults.get("trading_days_per_year", 252),
        minutes_per_trading_day=defaults.get("minutes_per_trading_day", 390),
        label_lookahead_bars=defaults.get("label_lookahead_bars", 1),
        embargo_days=defaults.get("embargo_days", 1),
        cpcv_groups=defaults.get("cpcv", {}).get("n_groups", 12),
        cpcv_test_groups=defaults.get("cpcv", {}).get("test_group_size", 2),
        cpcv_max_splits=defaults.get("cpcv", {}).get("max_splits"),
        cpcv_random_state=defaults.get("cpcv", {}).get("random_state"),
        dsr_threshold=defaults.get("dsr_threshold", 0.05),
        go_no_go=defaults.get("go_no_go"),
        rc_bootstrap=defaults.get("reality_checks", {}).get("n_bootstrap", 1000),
        rc_block_len=defaults.get("reality_checks", {}).get("block_len_bars", 78),
    )


def _default_yaml() -> Dict[str, Any]:
    return {
        "strategies": [
            {"name": "ma_fast", "type": "moving_average", "params": {"fast": 10, "slow": 40}},
            {"name": "ma_medium", "type": "moving_average", "params": {"fast": 20, "slow": 60}},
            {"name": "ma_slow", "type": "moving_average", "params": {"fast": 40, "slow": 120}},
        ],
        "trading_days_per_year": 252,
        "minutes_per_trading_day": 390,
        "label_lookahead_bars": 1,
        "embargo_days": 1,
        "cpcv": {
            "n_groups": 12,
            "test_group_size": 2,
            "random_state": 7,
            "max_splits": 20,
        },
        "dsr_threshold": 0.05,
        "go_no_go": {
            "sharpe_min": 0.8,
            "dsr_p_max": 0.05,
            "slippage_budget_max": 0.8,
        },
        "reality_checks": {
            "n_bootstrap": 1000,
            "block_len_bars": 78,
        },
    }


def run_wfo(
    symbols: Sequence[str],
    is_days: int,
    oos_days: int,
    step_days: int,
    cycles_min: int,
    embargo_days: int,
    label_lookahead_bars: int,
    cpcv_folds: int,
    config_path: Path | str | None,
    dry_run: bool = True,
    output_root: Path | str = Path("artifacts/wfo"),
) -> Dict[str, Any]:
    config = load_config(config_path)
    config.label_lookahead_bars = label_lookahead_bars or config.label_lookahead_bars
    config.embargo_days = embargo_days or config.embargo_days

    data_access = MarketDataAccess()
    now = datetime.utcnow()
    start_lookup = now - timedelta(days=max(is_days, 180) * 2)

    per_cycle_records: List[Dict[str, Any]] = []
    per_config_returns: Dict[str, List[np.ndarray]] = {s.name: [] for s in config.strategies}
    selected_returns: List[np.ndarray] = []

    for symbol in symbols:
        bars = data_access.get_bars(symbol, start_lookup, now)
        if bars.empty:
            print(f"[WFO] No data found for {symbol}, skipping")
            continue

        cycles = _build_cycles(bars, is_days, oos_days, step_days)
        if len(cycles) < cycles_min:
            print(f"[WFO] Not enough cycles for {symbol}: {len(cycles)} < {cycles_min}")
            continue

        for cycle_idx, (is_df, oos_df) in enumerate(cycles, start=1):
            is_df = is_df.copy()
            oos_df = oos_df.copy()
            embargo_bars = int(config.embargo_days * config.minutes_per_trading_day)
            if config.label_lookahead_bars > 0:
                if len(is_df) <= config.label_lookahead_bars:
                    print(f"[WFO] Skipping cycle {cycle_idx} for {symbol}: IS window too short after label purge")
                    continue
                is_df = is_df.iloc[:-config.label_lookahead_bars]
            if embargo_bars > 0:
                if len(oos_df) <= embargo_bars:
                    print(f"[WFO] Skipping cycle {cycle_idx} for {symbol}: OOS window too short after embargo")
                    continue
                oos_df = oos_df.iloc[embargo_bars:]
            if is_df.empty or oos_df.empty:
                print(f"[WFO] Skipping cycle {cycle_idx} for {symbol}: empty window after gap enforcement")
                continue
            selection = _run_cpcv_selection(
                is_df,
                config,
                cpcv_folds=cpcv_folds,
                label_lookahead=config.label_lookahead_bars,
                minutes_per_day=config.minutes_per_trading_day,
            )
            if not selection:
                print(f"[WFO] Skipping cycle {cycle_idx} for {symbol}: no strategies passed CPCV selection")
                continue
            best = selection[0]
            for candidate in selection:
                strat_returns = _strategy_returns(oos_df, candidate["strategy"])
                per_config_returns.setdefault(candidate["strategy"].name, []).append(strat_returns.values)

            equity = _strategy_equity(oos_df, best["strategy"])
            returns = _strategy_returns(oos_df, best["strategy"])

            record = {
                "cycle": cycle_idx,
                "symbol": symbol,
                "config": best["strategy"].name,
                "sharpe": sharpe_ratio(returns.values),
                "sortino": sortino_ratio(returns.values),
                "max_drawdown": max_drawdown(equity.values),
                "calmar": calmar_ratio(returns.values, equity.values),
                "turnover": turnover(_strategy_signals(oos_df, best["strategy"]).values),
                "expectancy": expectancy(returns.values),
                "slippage_usage": 0.0,
            }
            per_cycle_records.append(record)
            selected_returns.append(returns.values)

            if dry_run and cycle_idx >= cycles_min:
                break

    if not per_cycle_records:
        raise RuntimeError("No WFO cycles were executed")

    output_dir = Path(output_root) / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary = summarise_cycles(per_cycle_records)

    aggregated_selected = np.concatenate(selected_returns) if selected_returns else np.zeros(1)

    matrix_columns = []
    names_order = []
    for name, series_list in per_config_returns.items():
        if not series_list:
            continue
        names_order.append(name)
        matrix_columns.append(np.concatenate(series_list))
    if not matrix_columns:
        matrix_columns.append(aggregated_selected)
        names_order.append("selected")
    oos_matrix = np.column_stack(matrix_columns)
    m_eff = _effective_trials(oos_matrix)
    dsr = deflated_sharpe_ratio(aggregated_selected, trials_effective=m_eff)
    white = white_reality_check(oos_matrix, n_bootstrap=config.rc_bootstrap, block_len=config.rc_block_len)
    spa = hansen_spa(oos_matrix, n_bootstrap=config.rc_bootstrap, block_len=config.rc_block_len)

    extras = {
        "dsr": {"z_score": dsr.z_score, "p_value": dsr.p_value, "effective_trials": m_eff},
        "white_rc": {"p_value": white.p_value, "survivors": white.survivors, "strategies": names_order},
        "spa": {"p_value": spa.p_value, "survivors": spa.survivors, "strategies": names_order},
    }

    write_reports(output_dir, per_cycle_records, summary, extras)

    if dry_run:
        os.environ.setdefault("DRY_RUN", "1")
        os.environ.setdefault("ALLOW_ORDERS", "0")

    if config.go_no_go:
        summary["go_no_go"] = _evaluate_go_no_go(summary, dsr, config.go_no_go)

    return {
        "summary": summary,
        "output_dir": str(output_dir),
        "dsr": extras["dsr"],
    }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _build_cycles(bars: pd.DataFrame, is_days: int, oos_days: int, step_days: int) -> List[tuple[pd.DataFrame, pd.DataFrame]]:
    bars = bars.copy()
    bars = bars.sort_values("timestamp")
    bars["day"] = bars["timestamp"].dt.floor("D")
    unique_days = bars["day"].drop_duplicates().to_list()
    cycles: List[tuple[pd.DataFrame, pd.DataFrame]] = []
    pointer = is_days
    while pointer + oos_days <= len(unique_days):
        is_days_list = unique_days[pointer - is_days : pointer]
        oos_days_list = unique_days[pointer : pointer + oos_days]
        is_slice = bars[bars["day"].isin(is_days_list)].drop(columns=["day"]).copy()
        oos_slice = bars[bars["day"].isin(oos_days_list)].drop(columns=["day"]).copy()
        if not is_slice.empty and not oos_slice.empty:
            cycles.append((is_slice, oos_slice))
        pointer += step_days
    return cycles


def _run_cpcv_selection(
    is_df: pd.DataFrame,
    config: RunnerConfig,
    cpcv_folds: int,
    label_lookahead: int,
    minutes_per_day: int,
) -> List[Dict[str, Any]]:
    base_df = is_df.reset_index(drop=True)
    returns = base_df["close"].pct_change().dropna().reset_index(drop=True)
    timestamps = base_df["timestamp"].iloc[1:].reset_index(drop=True)

    groups = min(config.cpcv_groups, len(returns))
    embargo_bars = int(config.embargo_days * config.minutes_per_trading_day)
    cpcv_cfg = CPCVConfig(
        n_groups=groups,
        test_group_size=min(config.cpcv_test_groups, max(1, groups - 1)),
        embargo=embargo_bars,
        label_lookahead=int(label_lookahead),
        max_splits=cpcv_folds,
        random_state=config.cpcv_random_state,
    )
    splitter = CombinatorialPurgedCV(cpcv_cfg)

    results = []
    for strat in config.strategies:
        strat_returns = _strategy_returns(base_df, strat).iloc[1:].reset_index(drop=True)
        oos_sharpes = []
        oos_returns = []
        for train_idx, test_idx in splitter.split(timestamps):
            if len(test_idx) == 0:
                continue
            fold_returns = strat_returns.iloc[test_idx]
            oos_sharpes.append(sharpe_ratio(fold_returns.values))
            oos_returns.append(fold_returns.values)
        if not oos_returns:
            continue
        stacked = np.concatenate(oos_returns)
        dsr = deflated_sharpe_ratio(stacked, trials_effective=len(config.strategies))
        results.append(
            {
                "strategy": strat,
                "median_sharpe": float(np.median(oos_sharpes)),
                "sharpe_scores": oos_sharpes,
                "dsr": dsr,
            }
        )

    results.sort(key=lambda x: (x["median_sharpe"], -x["dsr"].p_value), reverse=True)
    return results


def _strategy_returns(df: pd.DataFrame, strategy: StrategyConfig) -> pd.Series:
    if strategy.type == "moving_average":
        fast = strategy.params.get("fast", 10)
        slow = strategy.params.get("slow", 40)
        tmp = df[["timestamp", "close"]].copy()
        tmp["fast"] = tmp["close"].rolling(fast, min_periods=fast).mean()
        tmp["slow"] = tmp["close"].rolling(slow, min_periods=slow).mean()
        tmp["signal"] = np.where(tmp["fast"] > tmp["slow"], 1.0, -1.0)
        tmp["signal"] = tmp["signal"].shift(1).fillna(0)
        tmp["returns"] = tmp["close"].pct_change().fillna(0)
        tmp["strategy"] = tmp["signal"] * tmp["returns"]
        return tmp["strategy"].fillna(0)
    raise ValueError(f"Unsupported strategy type: {strategy.type}")


def _strategy_equity(df: pd.DataFrame, strategy: StrategyConfig) -> pd.Series:
    returns = _strategy_returns(df, strategy)
    equity = (1 + returns).cumprod()
    return equity


def _strategy_signals(df: pd.DataFrame, strategy: StrategyConfig) -> pd.Series:
    if strategy.type == "moving_average":
        fast = strategy.params.get("fast", 10)
        slow = strategy.params.get("slow", 40)
        tmp = df[["close"]].copy()
        tmp["fast"] = tmp["close"].rolling(fast, min_periods=fast).mean()
        tmp["slow"] = tmp["close"].rolling(slow, min_periods=slow).mean()
        signal = np.where(tmp["fast"] > tmp["slow"], 1.0, -1.0)
        return pd.Series(signal, index=df.index).shift(1).fillna(0)
    raise ValueError(f"Unsupported strategy type: {strategy.type}")


def _effective_trials(matrix: np.ndarray) -> float:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        return 1.0
    if matrix.shape[1] == 1:
        return 1.0
    c = np.nan_to_num(np.corrcoef(matrix.T), nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, 1.0)
    eig = np.linalg.eigvalsh(c)
    denom = np.sum(eig ** 2) + 1e-12
    return float(max(1.0, (eig.sum() ** 2) / denom))


def _evaluate_go_no_go(summary: Dict[str, Any], dsr, thresholds: Dict[str, Any]) -> bool:
    sharpe_min = thresholds.get("sharpe_min")
    dsr_p_max = thresholds.get("dsr_p_max")
    slippage_max = thresholds.get("slippage_budget_max")
    sharpe_ok = sharpe_min is None or summary.get("sharpe_median", 0) >= sharpe_min
    dsr_ok = dsr_p_max is None or dsr.p_value <= dsr_p_max
    slippage_ok = slippage_max is None or summary.get("slippage_budget_used", 0) <= slippage_max
    return bool(sharpe_ok and dsr_ok and slippage_ok)


__all__ = ["run_wfo", "load_config"]
