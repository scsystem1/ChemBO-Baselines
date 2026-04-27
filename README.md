# ChemBO Baselines

This repository packages the baseline methods and helper scripts extracted from the local `ChemBO-Agent` workspace into a standalone release layout.

Included:

- `baseline/BO-ICL`
- `baseline/Pref-BO`
- `baseline/Reasoning-BO`
- `baseline/gollum`
- `baseline/common/tabular_benchmarks.py`
- `scripts/` with baseline-focused run scripts
- `data/DAR.csv` and `data/OCM.csv` for the tabular benchmark runs

Not included:

- ChemBO Agent main optimization code
- experiment outputs
- local caches and nested git metadata

Notes:

- Some baseline subprojects preserve their own upstream layout and dependency files.
- The run scripts assume this repository root contains `baseline/`, `scripts/`, and `data/`.
