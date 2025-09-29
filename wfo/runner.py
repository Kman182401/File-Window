"""Main orchestration for the walk-forward optimisation framework."""

from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, UTC
from importlib import metadata
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from .cpcv import CombinatorialPurgedCV, CPCVConfig
from .data_access import DataAccessConfig, MarketDataAccess
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
from .rl_adapter import RLAdapter, RLSpec, SB3_AVAILABLE
from .labeling import (
    TripleBarrierConfig,
    ensure_forward_label,
    triple_barrier_labels,
)
from .supervised_baselines import logistic_positions
from .utils import enable_determinism, resolve_session_minutes
_TB_CACHE_LIMIT = 64
_TB_LABEL_CACHE: OrderedDict[Tuple[Any, ...], pd.DataFrame] = OrderedDict()


@dataclass
class StrategyConfig:
    name: str
    type: str
    params: Optional[Dict[str, Any]] = None
    algo: Optional[str] = None
    policy: Optional[str] = None
    rl: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    reward: Optional[Dict[str, Any]] = None


@dataclass
class RunnerConfig:
    strategies: List[StrategyConfig]
    trading_days_per_year: int = 252
    minutes_per_trading_day: int = 390
    session_minutes: Dict[str, int] = field(default_factory=dict)
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
    rl_fast_smoke: bool = False
    rl_fast_overrides: Optional[Dict[str, Any]] = None
    trading_costs_bps: float = 0.0
    deterministic_debug: bool = False
    deterministic_seed: int = 42
    labeling: Optional[Dict[str, Any]] = None
    event_bars: Optional[Dict[str, Any]] = None
    fracdiff: Optional[Dict[str, Any]] = None


def _tb_cache_key(
    symbol: str,
    cycle_idx: int,
    df: pd.DataFrame,
    config: TripleBarrierConfig,
    use_meta: bool,
    data_hash: Optional[str],
) -> Tuple[Any, ...]:
    if df.empty:
        start = end = None
    else:
        start = df.index[0]
        end = df.index[-1]
    return (
        symbol,
        cycle_idx,
        data_hash,
        len(df),
        str(start),
        str(end),
        float(config.pt_multiplier),
        float(config.sl_multiplier),
        int(config.max_holding),
        int(config.volatility_span),
        bool(use_meta),
        config.side_col,
    )


def _tb_cache_get(key: Tuple[Any, ...]) -> Optional[pd.DataFrame]:
    cached = _TB_LABEL_CACHE.get(key)
    if cached is None:
        return None
    _TB_LABEL_CACHE.move_to_end(key)
    return cached.copy()


def _tb_cache_set(key: Tuple[Any, ...], value: pd.DataFrame) -> None:
    _TB_LABEL_CACHE[key] = value.copy()
    _TB_LABEL_CACHE.move_to_end(key)
    while len(_TB_LABEL_CACHE) > _TB_CACHE_LIMIT:
        _TB_LABEL_CACHE.popitem(last=False)


def load_config(path: Path | str | None) -> RunnerConfig:
    if path is None:
        defaults = _default_yaml()
    else:
        with open(path, "r", encoding="utf-8") as fh:
            defaults = yaml.safe_load(fh)
    strategies = [_build_strategy(entry) for entry in defaults.get("strategies", [])]
    return RunnerConfig(
        strategies=strategies,
        trading_days_per_year=defaults.get("trading_days_per_year", 252),
        minutes_per_trading_day=defaults.get("minutes_per_trading_day", 390),
        session_minutes=defaults.get("session_minutes", {}),
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
        rl_fast_smoke=bool(defaults.get("rl_fast_smoke", False)),
        rl_fast_overrides=defaults.get("rl_fast_overrides"),
        trading_costs_bps=float(defaults.get("trading_costs_bps", 0.0)),
        deterministic_debug=bool(defaults.get("deterministic_debug", False)),
        deterministic_seed=int(defaults.get("deterministic_seed", 42)),
        labeling=defaults.get("labeling"),
        event_bars=defaults.get("event_bars"),
        fracdiff=defaults.get("fracdiff"),
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
        "session_minutes": {
            "default": 390,
        },
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
        "deterministic_debug": False,
        "deterministic_seed": 42,
    }


def _build_strategy(entry: Dict[str, Any]) -> StrategyConfig:
    return StrategyConfig(
        name=entry["name"],
        type=entry["type"],
        params=entry.get("params"),
        algo=entry.get("algo"),
        policy=entry.get("policy"),
        rl=entry.get("rl"),
        model=entry.get("model"),
        reward=entry.get("reward"),
    )


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
    strategies: Optional[Sequence[Any]] = None,
    rl_fast_smoke: Optional[bool] = None,
    rl_fast_overrides: Optional[Dict[str, Any]] = None,
    costs_bps: Optional[float] = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    config.label_lookahead_bars = label_lookahead_bars or config.label_lookahead_bars
    config.embargo_days = embargo_days or config.embargo_days
    if rl_fast_smoke is not None:
        config.rl_fast_smoke = bool(rl_fast_smoke)
    if rl_fast_overrides is not None:
        config.rl_fast_overrides = rl_fast_overrides
    if costs_bps is not None:
        config.trading_costs_bps = costs_bps

    if config.rc_bootstrap < 1000:
        print(
            "[WFO] Increasing n_bootstrap to 1000 to stabilise RC/SPA p-values."
        )
        config.rc_bootstrap = 1000
    if config.rc_block_len < 60:
        print(
            "[WFO] Increasing block_len_bars to 60 to respect SPA/RC dependence assumptions."
        )
        config.rc_block_len = 60

    if config.deterministic_debug:
        print(
            "[WFO] Deterministic debug mode enabled (may reduce performance)."
        )
        enable_determinism(config.deterministic_seed)

    data_cfg = DataAccessConfig(event_bars=config.event_bars, fracdiff=config.fracdiff)
    data_access = MarketDataAccess(data_cfg)
    now = datetime.now(UTC)
    start_lookup = now - timedelta(days=max(is_days, 180) * 2)

    strategy_entries = strategies if strategies is not None else config.strategies
    strategy_objs: List[StrategyConfig] = [
        s if isinstance(s, StrategyConfig) else _build_strategy(s)
        for s in strategy_entries
    ]

    label_cfg = config.labeling or {}
    label_mode = str(label_cfg.get("mode", "forward")).lower()
    use_meta_labels = bool(label_cfg.get("use_meta", False))
    tb_config_template: TripleBarrierConfig | None = None
    if label_mode == "triple_barrier":
        tb_config_template = TripleBarrierConfig(
            pt_multiplier=float(label_cfg.get("pt_multiplier", 1.0)),
            sl_multiplier=float(label_cfg.get("sl_multiplier", 1.0)),
            max_holding=int(label_cfg.get("max_holding_bars", max(1, config.label_lookahead_bars))),
            volatility_span=int(label_cfg.get("volatility_span", 50)),
            price_col=str(label_cfg.get("price_col", "close")),
            volatility_col=label_cfg.get("volatility_col"),
            side_col=label_cfg.get("side_col"),
            label_col=str(label_cfg.get("label_col", "label")),
            meta_label_col=str(label_cfg.get("meta_label_col", "meta_label")),
            ret_col=str(label_cfg.get("ret_col", "ret")),
            t1_col=str(label_cfg.get("t1_col", "t1")),
            sample_weight_col=str(label_cfg.get("sample_weight_col", "sample_weight")),
        )

    per_cycle_records: List[Dict[str, Any]] = []
    per_config_returns: Dict[str, List[np.ndarray]] = {s.name: [] for s in strategy_objs}
    selected_returns: List[np.ndarray] = []
    data_hashes: Dict[str, str] = {}
    rl_vecnorm_paths: Dict[str, List[str]] = defaultdict(list)
    resolved_minutes: Dict[str, int] = {}

    for symbol in symbols:
        minutes_lookup = {k.upper(): v for k, v in (config.session_minutes or {}).items()}
        symbol_minutes = minutes_lookup.get(symbol.upper())
        if symbol_minutes is None:
            symbol_minutes = resolve_session_minutes(symbol, config.minutes_per_trading_day)
        if symbol_minutes is None:
            symbol_minutes = config.minutes_per_trading_day
        else:
            symbol_minutes = int(symbol_minutes)
        resolved_minutes[symbol] = symbol_minutes
        bars = data_access.get_bars(symbol, start_lookup, now)
        if bars.empty:
            print(f"[WFO] No data found for {symbol}, skipping")
            continue
        data_hashes[symbol] = _hash_dataframe(bars)

        cycles = _build_cycles(bars, is_days, oos_days, step_days)
        if len(cycles) < cycles_min:
            print(f"[WFO] Not enough cycles for {symbol}: {len(cycles)} < {cycles_min}")
            continue

        for cycle_idx, (is_df, oos_df) in enumerate(cycles, start=1):
            is_df = is_df.copy()
            oos_df = oos_df.copy()
            embargo_bars = int(config.embargo_days * symbol_minutes)
            if config.label_lookahead_bars > 0:
                if len(is_df) <= config.label_lookahead_bars:
                    print(f"[WFO] Skipping cycle {cycle_idx} for {symbol}: IS window too short after label purge")
                    continue
                is_df = is_df.iloc[:-config.label_lookahead_bars]
            effective_embargo = embargo_bars
            if embargo_bars > 0:
                if len(oos_df) <= embargo_bars:
                    fallback = max(0, len(oos_df) - 1)
                    if fallback < embargo_bars:
                        print(
                            f"[WFO] Reducing embargo bars for {symbol} cycle {cycle_idx} "
                            f"from {embargo_bars} to {fallback} to preserve OOS window"
                        )
                    effective_embargo = fallback
                if effective_embargo > 0:
                    if len(oos_df) <= effective_embargo:
                        print(
                            f"[WFO] Skipping cycle {cycle_idx} for {symbol}: OOS window too short after embargo "
                            f"(effective={effective_embargo})"
                        )
                        continue
                    oos_df = oos_df.iloc[effective_embargo:]
            if is_df.empty or oos_df.empty:
                print(f"[WFO] Skipping cycle {cycle_idx} for {symbol}: empty window after gap enforcement")
                continue

            lookahead = max(1, config.label_lookahead_bars)
            if label_mode == "triple_barrier" and tb_config_template is not None:
                cache_key = _tb_cache_key(
                    symbol,
                    cycle_idx,
                    is_df,
                    tb_config_template,
                    use_meta_labels,
                    data_hashes.get(symbol),
                )
                tb_labels = _tb_cache_get(cache_key)
                if tb_labels is None:
                    tb_labels = triple_barrier_labels(is_df, tb_config_template)
                    _tb_cache_set(cache_key, tb_labels)
                tb_labels = tb_labels.reindex(is_df.index)
                is_df = is_df.join(tb_labels)
                if "exit_index" in is_df.columns:
                    is_df = is_df.drop(columns=["exit_index"])
                target_col = tb_config_template.meta_label_col if use_meta_labels else tb_config_template.label_col
                if target_col in is_df.columns:
                    is_df = is_df.dropna(subset=[target_col])
            else:
                ensure_forward_label(is_df, horizon=lookahead)

            ensure_forward_label(oos_df, horizon=lookahead)

            non_rl = [s for s in strategy_objs if s.type not in {"rl_policy", "supervised"}]
            selection: List[Dict[str, Any]] = []
            if non_rl:
                selection = _run_cpcv_selection(
                    is_df,
                    config,
                    non_rl,
                    cpcv_folds=cpcv_folds,
                    label_lookahead=config.label_lookahead_bars,
                )
                if selection:
                    best = selection[0]
                    for candidate in selection:
                        candidate_returns = _strategy_returns(oos_df, candidate["strategy"])
                        per_config_returns.setdefault(candidate["strategy"].name, []).append(candidate_returns.values)
                        if candidate["strategy"].name == best["strategy"].name:
                            record = _build_cycle_record(
                                cycle_idx,
                                symbol,
                                candidate["strategy"].name,
                                candidate_returns.values,
                                _strategy_signals(oos_df, candidate["strategy"]).values,
                            )
                            per_cycle_records.append(record)
                            selected_returns.append(candidate_returns.values)
                else:
                    print(f"[WFO] CPCV produced no results for cycle {cycle_idx} {symbol}")

            rl_strats = [s for s in strategy_objs if s.type == "rl_policy"]
            for rl_strat in rl_strats:
                if not SB3_AVAILABLE:
                    print(
                        f"[WFO] Skipping RL strategy {rl_strat.name}: RL libraries unavailable. "
                        "Install gymnasium, stable-baselines3, sb3-contrib to enable RL paths."
                    )
                    continue
                try:
                    from wfo_rl.rl_env_builder import make_env_from_df  # lazy import to avoid hard dependency
                except ImportError as exc:  # pragma: no cover - optional dependency guard
                    print(f"[WFO] Skipping RL strategy {rl_strat.name}: {exc}")
                    continue
                rl_cfg = dict(rl_strat.rl or {})
                if config.rl_fast_smoke and config.rl_fast_overrides:
                    rl_cfg.update(config.rl_fast_overrides)
                spec = RLSpec(
                    algo=rl_strat.algo or "RecurrentPPO",
                    policy=rl_strat.policy or ("MlpLstmPolicy" if (rl_strat.algo or "").lower().startswith("recurrent") else "MlpPolicy"),
                    train_timesteps=int(rl_cfg.get("train_timesteps", 50_000)),
                    n_envs=int(rl_cfg.get("n_envs", 1)),
                    seed=int(rl_cfg.get("seed", 42)),
                    vecnormalize_obs=bool(rl_cfg.get("vecnormalize_obs", True)),
                    vecnormalize_reward=bool(rl_cfg.get("vecnormalize_reward", True)),
                    policy_kwargs=rl_cfg.get("policy_kwargs") or {},
                    algo_kwargs=rl_cfg.get("algo_kwargs") or {},
                    use_imitation_warmstart=bool(rl_cfg.get("use_imitation_warmstart", False)),
                    imitation_kwargs=rl_cfg.get("imitation_kwargs"),
                    warmstart_epochs=int(rl_cfg.get("warmstart_epochs", 5)),
                    device=str(rl_cfg.get("device", "auto")),
                    compile_policy=bool(rl_cfg.get("compile_policy", False)),
                )
                adapter = RLAdapter(spec, fast_smoke=config.rl_fast_smoke)
                reward_kwargs = rl_strat.reward or {}
                is_env_fn = make_env_from_df(
                    is_df,
                    costs_bps=config.trading_costs_bps,
                    reward_kwargs=reward_kwargs,
                    eval_mode=False,
                )
                oos_env_fn = make_env_from_df(
                    oos_df,
                    costs_bps=config.trading_costs_bps,
                    reward_kwargs=reward_kwargs,
                    eval_mode=True,
                )
                setattr(is_env_fn, "_len", len(is_df))
                setattr(oos_env_fn, "_len", len(oos_df))
                strategy_dir = Path(output_root) / symbol / rl_strat.name / f"cycle_{cycle_idx}"
                strategy_dir.mkdir(parents=True, exist_ok=True)
                try:
                    model, vecnorm_path = adapter.fit_on_is(is_env_fn, str(strategy_dir))
                    rl_returns = adapter.score_on_oos(model, oos_env_fn, vecnorm_path)
                except RuntimeError as exc:  # pragma: no cover - dependency guard
                    print(f"[WFO] Skipping RL strategy {rl_strat.name}: {exc}")
                    continue
                per_config_returns.setdefault(rl_strat.name, []).append(rl_returns)
                per_cycle_records.append(
                    _build_cycle_record(
                        cycle_idx,
                        symbol,
                        rl_strat.name,
                        rl_returns,
                        None,
                    )
                )
                if not selection:
                    selected_returns.append(rl_returns)
                if vecnorm_path:
                    rl_vecnorm_paths[rl_strat.name].append(str(Path(vecnorm_path)))

            sup_strats = [s for s in strategy_objs if s.type == "supervised" and s.model == "logistic"]
            for sup_strat in sup_strats:
                params = sup_strat.params or {}
                if use_meta_labels and "target_col" not in params and tb_config_template is not None:
                    params["target_col"] = tb_config_template.meta_label_col
                target_col = params.get("target_col", "label")
                if "use_meta" not in params:
                    params["use_meta"] = use_meta_labels
                if params.get("use_meta") and "side_col" not in params and tb_config_template is not None:
                    params["side_col"] = tb_config_template.side_col or "side"
                if params.get("use_meta") and "meta_label_col" not in params and tb_config_template is not None:
                    params["meta_label_col"] = tb_config_template.meta_label_col
                if "sample_weight_col" not in params and tb_config_template is not None:
                    params["sample_weight_col"] = tb_config_template.sample_weight_col
                if target_col not in is_df.columns:
                    print(f"[WFO] Logistic baseline skipped (missing target '{target_col}' in IS data)")
                    continue
                try:
                    positions = logistic_positions(is_df, oos_df, **params)
                except Exception as exc:  # pragma: no cover - guard path
                    print(f"[WFO] Logistic baseline failed: {exc}")
                    continue
                returns_series = oos_df.get("returns")
                if returns_series is None:
                    returns_series = oos_df["close"].pct_change().fillna(0.0)
                logistic_returns = positions * returns_series.to_numpy(dtype=float)
                per_config_returns.setdefault(sup_strat.name, []).append(logistic_returns)
                per_cycle_records.append(
                    _build_cycle_record(
                        cycle_idx,
                        symbol,
                        sup_strat.name,
                        logistic_returns,
                        positions,
                    )
                )
                if not selection and not rl_strats:
                    selected_returns.append(logistic_returns)

            if dry_run and cycle_idx >= cycles_min:
                break

    if not per_cycle_records:
        raise RuntimeError("No WFO cycles were executed")

    output_dir = Path(output_root) / datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summary = summarise_cycles(per_cycle_records)

    if selected_returns:
        aggregated_selected = np.concatenate(selected_returns)
    else:
        first_non_empty = next((vals for vals in per_config_returns.values() if vals), [np.zeros(1)])
        aggregated_selected = np.concatenate(first_non_empty) if first_non_empty else np.zeros(1)

    if aggregated_selected.size == 0:
        aggregated_selected = np.zeros(1)

    total_return = float(np.prod(1 + aggregated_selected) - 1)
    summary["total_return"] = total_return
    summary["equity_final"] = float(1 + total_return)

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
    
    # Length guard: make all columns the same length before stacking
    min_len = min(col.shape[0] for col in matrix_columns)
    if any(col.shape[0] != min_len for col in matrix_columns):
        print(f"[WFO] Normalising column lengths to {min_len} for aggregation")
        matrix_columns = [col[:min_len] for col in matrix_columns]
    oos_matrix = np.column_stack(matrix_columns)

    m_eff = _effective_trials(oos_matrix)
    dsr = deflated_sharpe_ratio(aggregated_selected, trials_effective=m_eff)
    white = white_reality_check(oos_matrix, n_bootstrap=config.rc_bootstrap, block_len=config.rc_block_len)
    spa = hansen_spa(oos_matrix, n_bootstrap=config.rc_bootstrap, block_len=config.rc_block_len)

    summary["effective_trials"] = m_eff
    summary["white_rc_p"] = white.p_value
    summary["spa_p"] = spa.p_value

    extras = {
        "dsr": {"z_score": dsr.z_score, "p_value": dsr.p_value, "effective_trials": m_eff},
        "white_rc": {"p_value": white.p_value, "survivors": white.survivors, "strategies": names_order},
        "spa": {"p_value": spa.p_value, "survivors": spa.survivors, "strategies": names_order},
    }
    metadata_payload = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "config": {
            "trading_days_per_year": config.trading_days_per_year,
            "minutes_per_trading_day": config.minutes_per_trading_day,
            "session_minutes": config.session_minutes,
            "embargo_days": config.embargo_days,
            "label_lookahead_bars": config.label_lookahead_bars,
            "rl_fast_smoke": config.rl_fast_smoke,
            "reality_checks": {
                "n_bootstrap": config.rc_bootstrap,
                "block_len_bars": config.rc_block_len,
            },
        },
        "library_versions": _collect_versions(),
        "data_hashes": data_hashes,
        "strategy_seeds": {
            strat.name: (strat.rl.get("seed") if strat.rl else None)
            for strat in strategy_objs
        },
        "vecnormalize_stats": {k: v for k, v in rl_vecnorm_paths.items()},
        "session_minutes_used": resolved_minutes,
    }
    extras["run_metadata"] = metadata_payload


    write_reports(output_dir, per_cycle_records, summary, extras)

    if dry_run:
        os.environ.setdefault("DRY_RUN", "1")
        os.environ.setdefault("ALLOW_ORDERS", "0")

    if config.go_no_go:
        summary["go_no_go"] = _evaluate_go_no_go(summary, dsr, white, spa, config.go_no_go)

    return {
        "summary": summary,
        "white_rc_p": summary.get("white_rc_p"),
        "spa_p": summary.get("spa_p"),
        "go_no_go": summary.get("go_no_go"),
        "output_dir": str(output_dir),
        "dsr": extras["dsr"],
        "per_cycle_records": per_cycle_records,
        "run_metadata": metadata_payload,
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
    strategies: Sequence[StrategyConfig],
    cpcv_folds: int,
    label_lookahead: int,
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

    results: List[Dict[str, Any]] = []
    for strat in strategies:
        if strat.type == "rl_policy" or strat.model == "logistic":
            continue
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
        dsr = deflated_sharpe_ratio(stacked, trials_effective=len(strategies))
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
        params = strategy.params or {}
        fast = params.get("fast", 10)
        slow = params.get("slow", 40)
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
        params = strategy.params or {}
        fast = params.get("fast", 10)
        slow = params.get("slow", 40)
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


def _evaluate_go_no_go(
    summary: Dict[str, Any],
    dsr,
    white,
    spa,
    thresholds: Dict[str, Any],
) -> bool:
    sharpe_min = thresholds.get("sharpe_min")
    dsr_p_max = thresholds.get("dsr_p_max")
    slippage_max = thresholds.get("slippage_budget_max")
    white_p_min = thresholds.get("white_p_min")
    spa_p_min = thresholds.get("spa_p_min")
    sharpe_ok = sharpe_min is None or summary.get("sharpe_median", 0) >= sharpe_min
    dsr_ok = dsr_p_max is None or dsr.p_value <= dsr_p_max
    slippage_ok = slippage_max is None or summary.get("slippage_budget_used", 0) <= slippage_max
    white_ok = white_p_min is None or white.p_value >= white_p_min
    spa_ok = spa_p_min is None or spa.p_value >= spa_p_min
    return bool(sharpe_ok and dsr_ok and slippage_ok and white_ok and spa_ok)


def _build_cycle_record(
    cycle_idx: int,
    symbol: str,
    strategy_name: str,
    returns: np.ndarray,
    signals: Optional[np.ndarray],
) -> Dict[str, Any]:
    returns = np.asarray(returns, dtype=float)
    equity = np.cumprod(1 + returns)
    turnover_val = turnover(np.asarray(signals, dtype=float)) if signals is not None else 0.0
    return {
        "cycle": cycle_idx,
        "symbol": symbol,
        "config": strategy_name,
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(returns, equity),
        "turnover": turnover_val,
        "expectancy": expectancy(returns),
        "slippage_usage": 0.0,
    }


__all__ = ["run_wfo", "load_config"]
def _hash_dataframe(df: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _collect_versions() -> Dict[str, str]:
    libs = [
        "stable-baselines3",
        "sb3-contrib",
        "gymnasium",
        "torch",
        "numpy",
        "pandas",
    ]
    versions: Dict[str, str] = {}
    for lib in libs:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:  # pragma: no cover - optional deps
            versions[lib] = "not installed"
    return versions
