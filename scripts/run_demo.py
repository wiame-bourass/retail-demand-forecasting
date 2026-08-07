from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("\n$", " ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    demo_script = ROOT / "scripts" / "00_make_demo_data.py"
    spec = spec_from_file_location("demo_data", demo_script)
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    module.make_demo_data(ROOT / "data" / "demo_raw")

    config = "config/demo.yaml"
    steps = [
        "01_audit_raw_data.py",
        "02_select_scope.py",
        "03_build_curated_panel.py",
        "04_statistical_analysis.py",
        "05_build_feature_dataset.py",
        "05b_feature_diagnostics.py",
        "05c_build_family_weights.py",
        "06_backtest_and_select.py",
        "06b_statistical_models_benchmark.py",
        "07_evaluate_holdout.py",
        "08_train_final_and_predict.py",
        "09_inventory_simulation.py",
        "10_explain_model.py",
        "11_generate_executive_report.py",
        "12_monitoring_snapshot.py",
    ]
    for step in steps:
        run([sys.executable, f"scripts/{step}", "--config", config])
    run([sys.executable, "-m", "pytest"])
    print("\nDemo pipeline completed successfully.")


if __name__ == "__main__":
    main()
