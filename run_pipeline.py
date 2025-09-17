from pathlib import Path
import sys
import runpy

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _run_module_main():
    """Execute the canonical pipeline module's __main__ block."""
    try:
        runpy.run_module("src.rl_trading_pipeline", run_name="__main__")
    except Exception:
        # Fallback to legacy root pipeline
        runpy.run_module("rl_trading_pipeline", run_name="__main__")


if __name__ == "__main__":
    _run_module_main()
