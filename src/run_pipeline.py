#!/usr/bin/env python3
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    module = importlib.import_module("run_pipeline")
    module.main()
