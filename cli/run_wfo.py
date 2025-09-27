#!/usr/bin/env python3
"""CLI entry point for the walk-forward optimisation runner."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("DRY_RUN", "1")
os.environ.setdefault("ALLOW_ORDERS", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wfo.runner import run_wfo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward optimisation with CPCV model selection")
    parser.add_argument("--symbols", type=str, default="ES,NQ", help="Comma-separated symbols (ES,NQ,GC,6E,6B,6A)")
    parser.add_argument("--is_days", type=int, default=120, help="Number of trading days in the in-sample window")
    parser.add_argument("--oos_days", type=int, default=20, help="Number of trading days in the out-of-sample window")
    parser.add_argument("--step_days", type=int, default=20, help="Step size between WFO cycles in trading days")
    parser.add_argument("--cycles_min", type=int, default=12, help="Minimum number of cycles required per symbol")
    parser.add_argument("--embargo_days", type=int, default=1, help="Embargo in trading days between IS and OOS")
    parser.add_argument("--label_lookahead_bars", type=int, default=1, help="Number of bars the label looks ahead (purge width)")
    parser.add_argument("--cpcv_folds", type=int, default=12, help="Maximum CPCV splits to evaluate")
    parser.add_argument("--config", type=str, default=str(Path("wfo/wfo_config.yaml")), help="Path to YAML config")
    parser.add_argument("--dry_run", type=int, default=1, help="Set 1 to enforce DRY_RUN/ALLOW_ORDERS guards")
    parser.add_argument("--output", type=str, default=str(Path("artifacts/wfo")), help="Output directory root")
    parser.add_argument("--rl_fast_smoke", type=int, default=None, help="Override rl_fast_smoke (1/0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    symbols = [sym.strip() for sym in args.symbols.split(",") if sym.strip()]
    rl_fast_smoke = None if args.rl_fast_smoke is None else bool(args.rl_fast_smoke)
    result = run_wfo(
        symbols=symbols,
        is_days=args.is_days,
        oos_days=args.oos_days,
        step_days=args.step_days,
        cycles_min=args.cycles_min,
        embargo_days=args.embargo_days,
        label_lookahead_bars=args.label_lookahead_bars,
        cpcv_folds=args.cpcv_folds,
        config_path=args.config,
        dry_run=bool(args.dry_run),
        output_root=Path(args.output),
        rl_fast_smoke=rl_fast_smoke,
    )
    print("\nWFO completed: ")
    print(result)


if __name__ == "__main__":
    main()
