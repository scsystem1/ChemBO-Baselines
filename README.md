# ChemBO Baselines

This repository packages the baseline methods and helper scripts extracted from the local `ChemBO-Agent` workspace into a standalone release layout.

Included:

- `baseline/BO-ICL`
- `baseline/Pref-BO`
- `baseline/Reasoning-BO`
- `baseline/bora`
- `baseline/gollum`
- `baseline/common/tabular_benchmarks.py`
- `run_tabular_baseline.py` as a unified experiment launcher
- `scripts/` with baseline-focused run scripts
- `data/DAR.csv`, `data/OCM.csv`, and `data/suzuki.csv` for the tabular benchmark runs

Not included:

- ChemBO Agent main optimization code
- experiment outputs
- local caches and nested git metadata

Notes:

- Some baseline subprojects preserve their own upstream layout and dependency files.
- The run scripts assume this repository root contains `baseline/`, `scripts/`, and `data/`.
- The unified launcher can forward dataset-specific and baseline-specific CLI arguments while writing each run into its own timestamped output directory.
- `scripts/setup_baseline_envs.sh` bootstraps the per-baseline conda environments that are used in this repo.
- `scripts/run_suzuki_reasoningbo_gollum_3x.sh` runs the requested Suzuki benchmark suite for Reasoning-BO and GOLLuM.
