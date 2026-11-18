"""Run the lightweight test suite without requiring pytest."""

from __future__ import annotations

import importlib
import pathlib
import sys
import traceback

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

modules = ["test_model", "test_data_eval", "test_transfer"]
failed = 0
for name in modules:
    module = importlib.import_module(name)
    for attr in sorted(dir(module)):
        if not attr.startswith("test_"):
            continue
        try:
            getattr(module, attr)()
            print(f"PASS {name}.{attr}")
        except Exception:
            failed += 1
            print(f"FAIL {name}.{attr}")
            traceback.print_exc()
print(f"\n{len(modules)} modules, {failed} failures")
raise SystemExit(1 if failed else 0)
