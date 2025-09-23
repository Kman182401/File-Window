from pathlib import Path
import sys, importlib
import logging

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    force=True,
)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _resolve_main():
    """
    Import-time should be side-effect free.
    Resolve and return a callable main() without executing heavy module code.
    """
    try:
        mod = importlib.import_module("src.rl_trading_pipeline")
        return getattr(mod, "main")
    except Exception:
        mod = importlib.import_module("rl_trading_pipeline")
        return getattr(mod, "main")


if __name__ == "__main__":
    _resolve_main()()
