# Plot Organization

All plotting scripts in this folder read experiment tables from `../Results` and write paper-ready figures only inside `plots/figures`.

## Output layout

- `plots/figures/method_comparison`: combined method comparison panels
- `plots/figures/ablation_study`: ablation figures

Each script exports `.png` figures with normalized lowercase snake_case file names so the figures are easier to reference in the paper source.

## Scripts

- `plot_common.py`: shared plotting utilities for method comparisons
- `plot_all_methods_panels.py`: combined method comparison panels
- `plot_ablation.py`: ablation figures

## Usage

Run from the project root:

```bash
python plots/plot_all_methods_panels.py
python plots/plot_ablation.py
```

## Execution Notes

- Run these plotting scripts as plain script files from the project root.
- Avoid `python -m ...` execution, since these scripts use local sibling imports such as `plot_config` and `plot_common`.
- Open `test.ipynb` from the repository root, where `run.py` is located.
