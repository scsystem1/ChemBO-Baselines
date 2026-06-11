# Reviewer Run Guide

This repository keeps the original experiment logic unchanged.
For the main experiments, use `run.py` as the top-level entry point.
It dispatches directly to the original classes used for:

- `boost`
- `fixed`
- `static`
- `random_search`
- `best_utility`
- `botorch_default`
- `boost_fixedhp`
- `random_kernel_af`

The `No-PASt-BO` and `HEBO` baselines are not routed through `run.py`.
They rely on different dependency stacks (`GPyOpt` and `HEBO` respectively),
so each must be run in its own separate environment using the original scripts.

If you prefer an interactive workflow, open `test.ipynb`.
That notebook exposes a small set of editable variables at the top and then calls `run.py` with the corresponding options.
Open the notebook from the repository root, where `run.py` is located.

## What This Repository Reproduces

- All experiments were executed on CPU **except `botorch_default`**, which was executed on GPU (`--device cuda`). Its discrete acquisition optimization over ~1.87M candidates per iteration on synthetic benchmarks is roughly 10× slower on CPU. CPU still works for `botorch_default` if GPU is unavailable, but the published `botorch_default` numbers were obtained on GPU. Every other baseline in this paper used CPU.
- Each full experiment uses 30 trials with seeds `0` to `29`.
- Each run uses 10 initial points followed by 90 BO iterations, for 100 total evaluations.
- HPO-B data are already included under `Code/benchmarks/hpob_data/`.

## Main Requirements

Use any environment name you prefer, but match the following versions used for the main experiments:

- Python `3.11.11`
- PyTorch `2.7.1+cu128` (used on CPU for most methods and on GPU for `botorch_default`)
- torchvision `0.22.1+cu128`
- torchaudio `2.7.1+cu128`
- GPyTorch `1.14`
- BoTorch `0.14.0`
- NumPy `2.1.2`
- SciPy `1.15.3`
- scikit-learn `1.6.1`
- pandas `2.2.3`
- openpyxl `3.1.5`
- tqdm `4.67.1`
- joblib `1.5.1`

## No-PASt-BO Requirements

`No-PASt-BO` follows the official codebase with only minimal compatibility-oriented changes.
Because `GPyOpt` is not compatible with the newer numerical stack used by the main experiments, this baseline should be run in a separate environment.

Use versions compatible with the paper:

- Python `3.11.11`
- PyTorch `2.7.1+cu128`
- GPy `1.13.2`
- GPyOpt `1.2.6`
- NumPy `1.26.4`
- SciPy `1.12.0`
- scikit-learn `1.6.1`
- pandas `2.3.2`
- openpyxl `3.1.5`

## HEBO Requirements

`HEBO` uses the official `hebo` package. Because `hebo 0.3.6` pulls in different `torch` / `numpy` / `scipy` versions than the main BoTorch/GPyTorch stack when freshly installed, this baseline should be run in a separate environment.

Use versions compatible with the paper:

- Python `3.11.11`
- HEBO `0.3.6`
- PyTorch `2.7.1+cu128`
- NumPy `1.24.4`
- SciPy `1.12.0`
- pandas `2.3.3`
- openpyxl `3.1.5`
- tqdm `4.67.1`

## Quick Sanity Check

Activate the main environment first.
Even `--list` imports the project modules, so the required packages must already be installed.

Interactive notebook option:

- Open `test.ipynb`
- Edit the configuration cell near the top
- Run the notebook cells in order

List supported methods and tasks:

```bash
python run.py --list
```

Preview a run without writing any files:

```bash
python run.py --method boost --space synthetic --objective ackley --trials 1 --max-iter 12 --n-init-points 10 --dry-run
```

## Main Experiments via `run.py`

Run these commands from the repository root.

BOOST on all synthetic benchmarks:

```bash
python run.py --method boost --space synthetic --objective all --trials 30 --max-iter 100 --n-init-points 10 --device cpu
```

BOOST on all HPO-B tasks:

```bash
python run.py --method boost --space hpob --hpob-task all --trials 30 --max-iter 100 --n-init-points 10 --device cpu
```

Fixed kernel-acquisition BO on one synthetic benchmark:

```bash
python run.py --method fixed --space synthetic --objective levy --kernel Matern32 --acquisition EI --trials 30 --max-iter 100 --n-init-points 10 --device cpu
```

Static baseline:

```bash
python run.py --method static --space synthetic --objective rosenbrock --trials 30 --max-iter 100 --n-init-points 10 --device cpu
```

Random search baseline on one HPO-B task:

```bash
python run.py --method random_search --space hpob --hpob-task 5891_8D --trials 30 --max-iter 100 --n-init-points 10 --device cpu
```

Best Utility baseline on one HPO-B task:

```bash
python run.py --method best_utility --space hpob --hpob-task 5964_9D --trials 30 --max-iter 100 --n-init-points 10 --device cpu
```

BoTorch default (qLogNEI, q=1) on all synthetic benchmarks (GPU strongly recommended):

```bash
python run.py --method botorch_default --space synthetic --objective all --trials 30 --max-iter 100 --n-init-points 10 --device cuda
```

BoTorch default on all HPO-B tasks (GPU strongly recommended):

```bash
python run.py --method botorch_default --space hpob --hpob-task all --trials 30 --max-iter 100 --n-init-points 10 --device cuda
```

BOOST with fixed full-data GP hyperparameters on all synthetic benchmarks:

```bash
python run.py --method boost_fixedhp --space synthetic --objective all --trials 30 --max-iter 100 --n-init-points 10 --device cpu
```

BOOST fixed-HP on all HPO-B tasks:

```bash
python run.py --method boost_fixedhp --space hpob --hpob-task all --trials 30 --max-iter 100 --n-init-points 10 --device cpu
```

Random-KA baseline (uniform random pick from the 16-pair portfolio at every iteration) on all synthetic benchmarks:

```bash
python run.py --method random_kernel_af --space synthetic --objective all --trials 30 --max-iter 100 --n-init-points 10 --device cpu
```

Random-KA baseline on all HPO-B tasks:

```bash
python run.py --method random_kernel_af --space hpob --hpob-task all --trials 30 --max-iter 100 --n-init-points 10 --device cpu
```

## No-PASt-BO Baseline

Activate the separate `No-PASt-BO` environment, then run the original scripts from the repository root:

Synthetic benchmarks:

```bash
python Code/Adaptive_Acquisition/nopastbo_synthetic_benchmark.py
```

HPO-B tasks:

```bash
python Code/Adaptive_Acquisition/nopastbo_hpob.py
```

These scripts retain their original structure and their own output-directory logic.

## HEBO Baseline

Activate the separate `HEBO` environment, then run the original scripts from the repository root:

Synthetic benchmarks:

```bash
python Code/HEBO/Test_Benchmark_Functions_hebo.py
```

HPO-B tasks:

```bash
python Code/HEBO/Test_HPOB_hebo.py
```

These scripts retain their original structure and write outputs under `results/results_HEBO_<YYYYMMDD>_trial30/` (synthetic) and `results/results_HPOB_HEBO_<YYYYMMDD>_trial30/` (HPO-B).

## Output Files

By default, `run.py` writes outputs under:

```text
reviewer_runs/<timestamp>_<method>_<space>/
```

Each `run.py` output directory contains:

- `run_config.json` with the exact command and resolved configuration
- `.xlsx` files generated by the original code

You can override the output location:

```bash
python run.py --method boost --space synthetic --objective ackley --output-dir reviewer_runs/my_ackley_run
```

## Figure Regeneration

The repository also includes plotting scripts under `plots/` that regenerate paper-ready figures from the saved result tables in `Results/`.

Run these commands from the repository root:

```bash
python plots/plot_all_methods_panels.py
python plots/plot_ablation.py
```

These scripts write figures only under:

```text
plots/figures/
```

For additional plotting notes, see `plots/README.md`.

## Notes

- `--method fixed` requires both `--kernel` and `--acquisition`.
- For `boost`, `static`, `random_search`, `botorch_default`, `boost_fixedhp`, and `random_kernel_af`, the default `TBD` kernel-acquisition placeholders match the behavior of the original scripts.
- For `best_utility`, the default acquisition is `EI`, matching the original implementation in this repository.
- For `botorch_default`, GPU (`--device cuda`) is strongly recommended; the published `botorch_default` results were obtained on GPU. CPU still works but is roughly 10× slower for synthetic benchmarks.
- The main `run.py` entry point does not cover `No-PASt-BO` or `HEBO`; both baselines remain separate because of their dependency stacks.
